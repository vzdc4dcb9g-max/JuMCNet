from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ConfusionGuidedClassificationLoss(nn.Module):
    """Cross entropy plus compactness/separation for validated hard pairs."""

    requires_features = True

    def __init__(
        self,
        label_smoothing: float,
        confusion_pairs: list[tuple[int, int]],
        auxiliary_weight: float = 0.15,
        max_pair_similarity: float = 0.20,
        compactness_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.confusion_pairs = confusion_pairs
        self.auxiliary_weight = float(auxiliary_weight)
        self.max_pair_similarity = float(max_pair_similarity)
        self.compactness_weight = float(compactness_weight)
        self.last_components: dict[str, float] = {}

    def _confusion_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(features.float(), dim=1)
        pair_classes = sorted(
            {class_index for pair in self.confusion_pairs for class_index in pair}
        )
        prototypes: dict[int, torch.Tensor] = {}
        compactness_terms: list[torch.Tensor] = []

        for class_index in pair_classes:
            class_mask = targets == class_index
            if not torch.any(class_mask):
                continue
            class_features = normalized[class_mask]
            prototype = F.normalize(
                class_features.mean(dim=0, keepdim=True), dim=1
            ).squeeze(0)
            prototypes[class_index] = prototype
            compactness_terms.append(
                (1.0 - class_features @ prototype).mean()
            )

        separation_terms: list[torch.Tensor] = []
        for first_class, second_class in self.confusion_pairs:
            if first_class not in prototypes or second_class not in prototypes:
                continue
            similarity = torch.sum(
                prototypes[first_class] * prototypes[second_class]
            )
            separation_terms.append(
                F.relu(similarity - self.max_pair_similarity)
            )

        zero = features.float().sum() * 0.0
        separation = (
            torch.stack(separation_terms).mean()
            if separation_terms
            else zero
        )
        compactness = (
            torch.stack(compactness_terms).mean()
            if compactness_terms
            else zero
        )
        return separation + self.compactness_weight * compactness

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        if features is None or self.auxiliary_weight <= 0.0:
            auxiliary = classification.detach() * 0.0
            total = classification
        else:
            auxiliary = self._confusion_loss(features, targets)
            total = classification + self.auxiliary_weight * auxiliary

        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "confusion_loss": float(auxiliary.detach().item()),
        }
        return total


class DifficultyWeightedCrossEntropyLoss(nn.Module):
    """Mild dynamic reweighting based only on training-set class losses.

    A per-class loss EMA estimates optimization difficulty. The normalized
    difficulty is clipped to a narrow interval so hard classes receive a
    modest boost without allowing a noisy class to dominate training.
    Evaluation always falls back to ordinary cross entropy and never updates
    the running statistics, preventing validation/test leakage.
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        ema_beta: float = 0.90,
        warmup_epochs: int = 5,
        min_weight: float = 0.80,
        max_weight: float = 1.20,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if not 0.0 < min_weight <= 1.0 <= max_weight:
            raise ValueError(
                "difficulty weights must satisfy 0 < min <= 1 <= max"
            )

        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)
        self.ema_beta = float(ema_beta)
        self.warmup_epochs = int(warmup_epochs)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.current_epoch = 0
        self.register_buffer(
            "class_loss_ema", torch.ones(self.num_classes)
        )
        self.register_buffer(
            "class_seen", torch.zeros(self.num_classes, dtype=torch.bool)
        )
        self.last_components: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @torch.no_grad()
    def _update_statistics(
        self,
        per_sample_loss: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        for class_index in targets.unique():
            index = int(class_index.item())
            class_mean = per_sample_loss[targets == index].mean().float()
            if not bool(self.class_seen[index]):
                self.class_loss_ema[index].copy_(class_mean)
                self.class_seen[index] = True
            else:
                self.class_loss_ema[index].mul_(self.ema_beta).add_(
                    class_mean, alpha=1.0 - self.ema_beta
                )

    @torch.no_grad()
    def current_weights(self) -> torch.Tensor:
        weights = torch.ones_like(self.class_loss_ema)
        if not torch.any(self.class_seen):
            return weights
        reference = self.class_loss_ema[self.class_seen].mean().clamp_min(1e-8)
        weights[self.class_seen] = self.class_loss_ema[self.class_seen] / reference
        weights.clamp_(self.min_weight, self.max_weight)
        weights.div_(weights.mean().clamp_min(1e-8))
        return weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        per_sample = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        training_step = torch.is_grad_enabled()
        if training_step:
            self._update_statistics(per_sample.detach(), targets)

        use_weights = (
            training_step and self.current_epoch > self.warmup_epochs
        )
        weights = (
            self.current_weights()
            if use_weights
            else torch.ones_like(self.class_loss_ema)
        )
        sample_weights = weights[targets]
        weighted = (per_sample * sample_weights).sum() / sample_weights.sum()
        plain = per_sample.mean()
        self.last_components = {
            "classification_loss": float(plain.detach().item()),
            "weighted_classification_loss": float(weighted.detach().item()),
            "difficulty_weight_min": float(weights.min().item()),
            "difficulty_weight_max": float(weights.max().item()),
        }
        return weighted


def _resolve_confusion_pairs(
    pair_names: list[list[str]],
    class_names: list[str],
) -> list[tuple[int, int]]:
    name_to_index = {name: index for index, name in enumerate(class_names)}
    resolved: list[tuple[int, int]] = []
    for pair in pair_names:
        if len(pair) != 2:
            raise ValueError(f"Each confusion pair must contain 2 names: {pair}")
        missing = [name for name in pair if name not in name_to_index]
        if missing:
            raise ValueError(
                f"Unknown classes in confusion pair {pair}; missing={missing}"
            )
        first, second = (name_to_index[pair[0]], name_to_index[pair[1]])
        if first == second:
            raise ValueError(f"Confusion pair must use different classes: {pair}")
        resolved.append((first, second))
    return resolved


class BinaryAuxClassificationLoss(nn.Module):
    """Main CE + healthy/diseased binary auxiliary CE (FAF-style)."""

    requires_aux = True

    def __init__(
        self,
        label_smoothing: float,
        healthy_class_index: int,
        auxiliary_weight: float = 0.30,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        # Binary aux uses hard 0/1 targets; keep smoothing at 0 for stability.
        self.aux_cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.healthy_class_index = int(healthy_class_index)
        self.auxiliary_weight = float(auxiliary_weight)
        self.last_components: dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        if aux_logits is None or self.auxiliary_weight <= 0.0:
            auxiliary = classification.detach() * 0.0
            total = classification
        else:
            binary_targets = (targets == self.healthy_class_index).long()
            auxiliary = self.aux_cross_entropy(aux_logits, binary_targets)
            total = classification + self.auxiliary_weight * auxiliary

        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "binary_aux_loss": float(auxiliary.detach().item()),
        }
        return total


class GlobalAuxClassificationLoss(nn.Module):
    """Main CE + Stage4 global-feat multi-class auxiliary CE."""

    requires_aux = True

    def __init__(
        self,
        label_smoothing: float,
        auxiliary_weight: float = 0.30,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.aux_cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.auxiliary_weight = float(auxiliary_weight)
        self.last_components: dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        if aux_logits is None or self.auxiliary_weight <= 0.0:
            auxiliary = classification.detach() * 0.0
            total = classification
        else:
            auxiliary = self.aux_cross_entropy(aux_logits, targets)
            total = classification + self.auxiliary_weight * auxiliary

        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "global_aux_loss": float(auxiliary.detach().item()),
        }
        return total


class LGRCClassificationLoss(nn.Module):
    """Main CE plus scheduled local-global relation consistency losses."""

    requires_lgrc = True

    def __init__(
        self,
        label_smoothing: float,
        max_weight: float = 0.10,
        relation_fraction: float = 0.50,
        start_epoch: int = 10,
        warmup_end_epoch: int = 30,
        decay_start_epoch: int = 180,
        end_epoch: int = 200,
        smooth_l1_beta: float = 0.10,
    ) -> None:
        super().__init__()
        if max_weight < 0.0:
            raise ValueError("lgrc.max_weight must be >= 0")
        if not 0.0 <= relation_fraction <= 1.0:
            raise ValueError("lgrc.relation_fraction must be in [0, 1]")
        if not (
            1 <= start_epoch <= warmup_end_epoch <= decay_start_epoch <= end_epoch
        ):
            raise ValueError(
                "LGRC epochs must satisfy 1 <= start <= warmup_end <= "
                "decay_start <= end"
            )
        if smooth_l1_beta <= 0.0:
            raise ValueError("lgrc.smooth_l1_beta must be > 0")

        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.max_weight = float(max_weight)
        self.relation_fraction = float(relation_fraction)
        self.start_epoch = int(start_epoch)
        self.warmup_end_epoch = int(warmup_end_epoch)
        self.decay_start_epoch = int(decay_start_epoch)
        self.end_epoch = int(end_epoch)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.current_epoch = 1
        self.last_components: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def effective_weight(self) -> float:
        epoch = self.current_epoch
        if epoch < self.start_epoch or epoch > self.end_epoch:
            return 0.0
        if epoch <= self.warmup_end_epoch:
            span = max(self.warmup_end_epoch - self.start_epoch + 1, 1)
            progress = (epoch - self.start_epoch + 1) / span
            return self.max_weight * min(max(progress, 0.0), 1.0)
        if epoch < self.decay_start_epoch:
            return self.max_weight
        span = max(self.end_epoch - self.decay_start_epoch + 1, 1)
        remaining = (self.end_epoch - epoch + 1) / span
        return self.max_weight * min(max(remaining, 0.0), 1.0)

    def auxiliary_active(self) -> bool:
        return self.effective_weight() > 0.0

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lgrc_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        weight = self.effective_weight()
        if lgrc_state is None or weight <= 0.0:
            relation = classification.detach() * 0.0
            position = classification.detach() * 0.0
            auxiliary = classification.detach() * 0.0
            total = classification
        else:
            required = {
                "local_relation",
                "global_relation",
                "position_prediction",
                "position_target",
            }
            missing = required.difference(lgrc_state)
            if missing:
                raise KeyError(f"LGRC state is missing keys: {sorted(missing)}")
            relation = F.smooth_l1_loss(
                lgrc_state["global_relation"].float(),
                lgrc_state["local_relation"].float(),
                beta=self.smooth_l1_beta,
            )
            position = F.smooth_l1_loss(
                lgrc_state["position_prediction"].float(),
                lgrc_state["position_target"].float(),
                beta=self.smooth_l1_beta,
            )
            auxiliary = (
                self.relation_fraction * relation
                + (1.0 - self.relation_fraction) * position
            )
            total = classification + weight * auxiliary

        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "lgrc_relation_loss": float(relation.detach().item()),
            "lgrc_position_loss": float(position.detach().item()),
            "lgrc_auxiliary_loss": float(auxiliary.detach().item()),
            "lgrc_effective_weight": float(weight),
        }
        return total


class RCAuxClassificationLoss(nn.Module):
    """Main CE plus a scheduled residual-codebook texture auxiliary CE."""

    requires_rcaux = True

    def __init__(
        self,
        label_smoothing: float,
        max_weight: float = 0.15,
        warmup_end_epoch: int = 20,
        decay_start_epoch: int = 160,
        end_epoch: int = 200,
    ) -> None:
        super().__init__()
        if max_weight < 0.0:
            raise ValueError("rcaux.max_weight must be >= 0")
        if not (
            1 <= warmup_end_epoch < decay_start_epoch < end_epoch
        ):
            raise ValueError(
                "RCAux epochs must satisfy 1 <= warmup_end < "
                "decay_start < end"
            )
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.aux_cross_entropy = nn.CrossEntropyLoss()
        self.max_weight = float(max_weight)
        self.warmup_end_epoch = int(warmup_end_epoch)
        self.decay_start_epoch = int(decay_start_epoch)
        self.end_epoch = int(end_epoch)
        self.current_epoch = 1
        self.last_components: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def effective_weight(self) -> float:
        epoch = self.current_epoch
        if epoch <= 0 or epoch >= self.end_epoch:
            return 0.0
        if epoch <= self.warmup_end_epoch:
            return self.max_weight * epoch / self.warmup_end_epoch
        if epoch <= self.decay_start_epoch:
            return self.max_weight
        decay_span = self.end_epoch - self.decay_start_epoch
        return self.max_weight * (
            self.end_epoch - epoch
        ) / decay_span

    def auxiliary_active(self) -> bool:
        return self.effective_weight() > 0.0

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        rcaux_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        weight = self.effective_weight()
        if rcaux_state is None or weight <= 0.0:
            auxiliary = classification.detach() * 0.0
            entropy = 0.0
            usage_values: list[float] = []
            total = classification
        else:
            required = {
                "auxiliary_logits",
                "codebook_usage",
                "assignment_entropy",
            }
            missing = required.difference(rcaux_state)
            if missing:
                raise KeyError(
                    f"RCAux state is missing keys: {sorted(missing)}"
                )
            auxiliary = self.aux_cross_entropy(
                rcaux_state["auxiliary_logits"].float(), targets
            )
            usage = rcaux_state["codebook_usage"].detach().float()
            usage_values = usage.cpu().tolist()
            entropy = float(
                rcaux_state["assignment_entropy"].detach().item()
            )
            total = classification + weight * auxiliary

        components = {
            "classification_loss": float(classification.detach().item()),
            "rcaux_auxiliary_loss": float(auxiliary.detach().item()),
            "rcaux_effective_weight": float(weight),
            "rcaux_assignment_entropy": float(entropy),
            "rcaux_usage_min": min(usage_values) if usage_values else 0.0,
            "rcaux_usage_max": max(usage_values) if usage_values else 0.0,
        }
        components.update(
            {
                f"rcaux_codeword_{index}_usage": float(value)
                for index, value in enumerate(usage_values)
            }
        )
        self.last_components = components
        return total


class RCAuxV4ClassificationLoss(RCAuxClassificationLoss):
    """Variant-aware RCAux criterion used by the independent Scheme-4 models."""

    SUPPORTED_VARIANTS = {
        "mha",
        "pn",
        "paw",
        "sa",
        "hnm",
        "kexp",
        "m2g",
        "ardrop",
    }

    def __init__(
        self,
        *args: Any,
        variant: str,
        mha_temperature: float = 2.0,
        mha_beta: float = 0.30,
        mha_confidence_threshold: float = 0.65,
        regularizer_start_epoch: int = 20,
        hnm_margin: float = 0.20,
        hnm_weight: float = 0.05,
        hnm_target_probability: float = 0.80,
        ardrop_kl_weight: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        normalized = str(variant).strip().lower()
        if normalized not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported RCAux-v4 variant: {variant!r}")
        if mha_temperature <= 0.0:
            raise ValueError("mha_temperature must be > 0")
        if not 0.0 <= mha_beta <= 1.0:
            raise ValueError("mha_beta must be in [0, 1]")
        if not 0.0 <= mha_confidence_threshold <= 1.0:
            raise ValueError("mha_confidence_threshold must be in [0, 1]")
        if regularizer_start_epoch < 0:
            raise ValueError("regularizer_start_epoch must be >= 0")
        if hnm_margin < 0.0 or hnm_weight < 0.0:
            raise ValueError("HNM margin and weight must be >= 0")
        if not 0.0 <= hnm_target_probability <= 1.0:
            raise ValueError("hnm_target_probability must be in [0, 1]")
        if ardrop_kl_weight < 0.0:
            raise ValueError("ardrop_kl_weight must be >= 0")
        self.variant = normalized
        self.mha_temperature = float(mha_temperature)
        self.mha_beta = float(mha_beta)
        self.mha_confidence_threshold = float(mha_confidence_threshold)
        self.regularizer_start_epoch = int(regularizer_start_epoch)
        self.hnm_margin = float(hnm_margin)
        self.hnm_weight = float(hnm_weight)
        self.hnm_target_probability = float(hnm_target_probability)
        self.ardrop_kl_weight = float(ardrop_kl_weight)

    @staticmethod
    def _symmetric_kl(
        first_logits: torch.Tensor,
        second_logits: torch.Tensor,
    ) -> torch.Tensor:
        first_log = F.log_softmax(first_logits.float(), dim=-1)
        second_log = F.log_softmax(second_logits.float(), dim=-1)
        first_prob = first_log.exp()
        second_prob = second_log.exp()
        first_to_second = F.kl_div(
            first_log, second_prob.detach(), reduction="batchmean"
        )
        second_to_first = F.kl_div(
            second_log, first_prob.detach(), reduction="batchmean"
        )
        return 0.5 * (first_to_second + second_to_first)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        rcaux_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        weight = self.effective_weight()
        zero = classification.detach() * 0.0
        auxiliary_ce = zero
        regularizer = zero
        active_rate = 0.0
        entropy = 0.0
        usage_values: list[float] = []
        diagnostic_values: dict[str, float] = {}

        if rcaux_state is None or weight <= 0.0:
            total = classification
            auxiliary_objective = zero
        else:
            required = {
                "auxiliary_logits",
                "codebook_usage",
                "assignment_entropy",
            }
            if self.variant == "ardrop":
                required.add("auxiliary_logits_2")
            missing = required.difference(rcaux_state)
            if missing:
                raise KeyError(
                    f"RCAux-v4 state is missing keys: {sorted(missing)}"
                )

            auxiliary_logits = rcaux_state["auxiliary_logits"].float()
            auxiliary_ce = self.aux_cross_entropy(auxiliary_logits, targets)
            regularizer_active = (
                self.current_epoch > self.regularizer_start_epoch
            )

            if self.variant == "mha" and regularizer_active:
                temperature = self.mha_temperature
                with torch.no_grad():
                    confidence = torch.softmax(
                        logits.detach().float(), dim=-1
                    ).amax(dim=-1)
                    active = confidence >= self.mha_confidence_threshold
                    teacher = torch.softmax(
                        logits.detach().float() / temperature, dim=-1
                    )
                per_sample = F.kl_div(
                    F.log_softmax(auxiliary_logits / temperature, dim=-1),
                    teacher,
                    reduction="none",
                ).sum(dim=-1)
                if bool(active.any()):
                    regularizer = per_sample[active].mean() * (
                        temperature * temperature
                    )
                active_rate = float(active.float().mean().item())
                auxiliary_objective = (
                    (1.0 - self.mha_beta) * auxiliary_ce
                    + self.mha_beta * regularizer
                )
            elif self.variant == "hnm" and regularizer_active:
                target_logits = auxiliary_logits.gather(
                    1, targets.unsqueeze(1)
                ).squeeze(1)
                wrong_logits = auxiliary_logits.clone()
                wrong_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                hardest_wrong = wrong_logits.max(dim=1).values
                target_probability = torch.softmax(
                    auxiliary_logits.detach(), dim=-1
                ).gather(1, targets.unsqueeze(1)).squeeze(1)
                active = target_probability < self.hnm_target_probability
                margins = F.relu(
                    self.hnm_margin - (target_logits - hardest_wrong)
                )
                if bool(active.any()):
                    regularizer = margins[active].mean()
                active_rate = float(active.float().mean().item())
                auxiliary_objective = (
                    auxiliary_ce + self.hnm_weight * regularizer
                )
            elif self.variant == "ardrop":
                second_logits = rcaux_state["auxiliary_logits_2"].float()
                second_ce = self.aux_cross_entropy(second_logits, targets)
                auxiliary_ce = 0.5 * (auxiliary_ce + second_ce)
                if regularizer_active:
                    regularizer = self._symmetric_kl(
                        auxiliary_logits, second_logits
                    )
                    active_rate = 1.0
                auxiliary_objective = (
                    auxiliary_ce + self.ardrop_kl_weight * regularizer
                )
            else:
                auxiliary_objective = auxiliary_ce

            usage = rcaux_state["codebook_usage"].detach().float()
            usage_values = usage.cpu().tolist()
            entropy = float(
                rcaux_state["assignment_entropy"].detach().item()
            )
            for state_name in (
                "power_norm_mean_abs",
                "saliency_entropy",
                "second_order_gate",
            ):
                state_value = rcaux_state.get(state_name)
                if state_value is not None:
                    diagnostic_values[f"rcaux_{state_name}"] = float(
                        state_value.detach().float().mean().item()
                    )
            total = classification + weight * auxiliary_objective

        components = {
            "classification_loss": float(classification.detach().item()),
            "rcaux_auxiliary_loss": float(auxiliary_ce.detach().item()),
            "rcaux_auxiliary_objective": float(
                auxiliary_objective.detach().item()
            ),
            "rcaux_variant_regularizer": float(regularizer.detach().item()),
            "rcaux_variant_active_rate": float(active_rate),
            "rcaux_effective_weight": float(weight),
            "rcaux_assignment_entropy": float(entropy),
            "rcaux_usage_min": min(usage_values) if usage_values else 0.0,
            "rcaux_usage_max": max(usage_values) if usage_values else 0.0,
        }
        components.update(diagnostic_values)
        components.update(
            {
                f"rcaux_codeword_{index}_usage": float(value)
                for index, value in enumerate(usage_values)
            }
        )
        self.last_components = components
        return total


class Optimization3AuxClassificationLoss(RCAuxClassificationLoss):
    """Main CE plus one scheduled, removable Optimization-3.0 auxiliary."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        auxiliary_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        weight = self.effective_weight()
        if auxiliary_state is None or weight <= 0.0:
            auxiliary = classification.detach() * 0.0
            entropy = 0.0
            usage_values: list[float] = []
            total = classification
        else:
            required = {
                "auxiliary_logits",
                "component_usage",
                "component_entropy",
            }
            missing = required.difference(auxiliary_state)
            if missing:
                raise KeyError(
                    "Optimization-3.0 auxiliary state is missing keys: "
                    f"{sorted(missing)}"
                )
            auxiliary = self.aux_cross_entropy(
                auxiliary_state["auxiliary_logits"].float(), targets
            )
            usage = auxiliary_state["component_usage"].detach().float()
            usage_values = usage.cpu().tolist()
            entropy = float(
                auxiliary_state["component_entropy"].detach().item()
            )
            total = classification + weight * auxiliary

        components = {
            "classification_loss": float(classification.detach().item()),
            "opt3_auxiliary_loss": float(auxiliary.detach().item()),
            "opt3_effective_weight": float(weight),
            "opt3_component_entropy": float(entropy),
            "opt3_usage_min": min(usage_values) if usage_values else 0.0,
            "opt3_usage_max": max(usage_values) if usage_values else 0.0,
        }
        components.update(
            {
                f"opt3_component_{index}_usage": float(value)
                for index, value in enumerate(usage_values)
            }
        )
        self.last_components = components
        return total


class UFBRCAuxClassificationLoss(RCAuxClassificationLoss):
    """RCAux plus a soft floor for persistently underused codewords.

    The floor is half of the uniform marginal usage, tau=1/(2K).  It is a
    one-sided penalty: codewords above the floor receive no balancing force,
    so the loss does not impose an artificial uniform distribution.
    """

    def __init__(
        self,
        label_smoothing: float,
        max_weight: float = 0.15,
        warmup_end_epoch: int = 20,
        decay_start_epoch: int = 160,
        end_epoch: int = 200,
        floor_weight: float = 0.02,
        floor_start_epoch: int = 10,
    ) -> None:
        super().__init__(
            label_smoothing=label_smoothing,
            max_weight=max_weight,
            warmup_end_epoch=warmup_end_epoch,
            decay_start_epoch=decay_start_epoch,
            end_epoch=end_epoch,
        )
        if floor_weight < 0.0:
            raise ValueError("rcaux.floor_weight must be >= 0")
        if not 1 <= floor_start_epoch < end_epoch:
            raise ValueError(
                "rcaux.floor_start_epoch must satisfy 1 <= start < end"
            )
        self.floor_weight = float(floor_weight)
        self.floor_start_epoch = int(floor_start_epoch)

    def floor_effective_weight(self) -> float:
        epoch = self.current_epoch
        if epoch < self.floor_start_epoch or epoch >= self.end_epoch:
            return 0.0
        if epoch <= self.decay_start_epoch:
            return self.floor_weight
        decay_span = self.end_epoch - self.decay_start_epoch
        return self.floor_weight * (self.end_epoch - epoch) / decay_span

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        rcaux_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        auxiliary_weight = self.effective_weight()
        floor_weight = self.floor_effective_weight()

        if rcaux_state is None or auxiliary_weight <= 0.0:
            auxiliary = classification.detach() * 0.0
            floor_loss = classification.detach() * 0.0
            entropy = 0.0
            usage_values: list[float] = []
            floor_threshold = 0.0
            starved_codewords = 0
            total = classification
        else:
            required = {
                "auxiliary_logits",
                "codebook_usage",
                "assignment_entropy",
            }
            missing = required.difference(rcaux_state)
            if missing:
                raise KeyError(
                    f"UFB-RCAux state is missing keys: {sorted(missing)}"
                )
            auxiliary = self.aux_cross_entropy(
                rcaux_state["auxiliary_logits"].float(), targets
            )
            usage = rcaux_state["codebook_usage"].float()
            if usage.ndim != 1 or usage.numel() < 2:
                raise ValueError(
                    "UFB-RCAux codebook_usage must be a 1D tensor with K>=2"
                )
            floor_threshold = 0.5 / float(usage.numel())
            normalized_deficit = F.relu(floor_threshold - usage) / (
                floor_threshold + 1e-12
            )
            floor_loss = normalized_deficit.square().mean()
            usage_detached = usage.detach()
            usage_values = usage_detached.cpu().tolist()
            starved_codewords = int(
                (usage_detached < floor_threshold).sum().item()
            )
            entropy = float(
                rcaux_state["assignment_entropy"].detach().item()
            )
            total = (
                classification
                + auxiliary_weight * auxiliary
                + floor_weight * floor_loss
            )

        components = {
            "classification_loss": float(classification.detach().item()),
            "rcaux_auxiliary_loss": float(auxiliary.detach().item()),
            "rcaux_effective_weight": float(auxiliary_weight),
            "rcaux_assignment_entropy": float(entropy),
            "rcaux_usage_min": min(usage_values) if usage_values else 0.0,
            "rcaux_usage_max": max(usage_values) if usage_values else 0.0,
            "ufb_floor_loss": float(floor_loss.detach().item()),
            "ufb_effective_weight": float(floor_weight),
            "ufb_floor_threshold": float(floor_threshold),
            "ufb_starved_codewords": float(starved_codewords),
        }
        components.update(
            {
                f"rcaux_codeword_{index}_usage": float(value)
                for index, value in enumerate(usage_values)
            }
        )
        self.last_components = components
        return total


class CGRHClassificationLoss(nn.Module):
    """Refined multi-class CE plus explicit hard-pair binary supervision."""

    requires_cgrh = True

    def __init__(
        self,
        label_smoothing: float,
        pair_indices: list[tuple[int, int]],
        auxiliary_weight: float = 0.20,
    ) -> None:
        super().__init__()
        if not pair_indices:
            raise ValueError("CGRHClassificationLoss requires hard pairs")
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.pair_indices = [(int(a), int(b)) for a, b in pair_indices]
        self.auxiliary_weight = float(auxiliary_weight)
        self.last_components: dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(logits, targets)
        if cgrh_state is None or self.auxiliary_weight <= 0.0:
            pair_loss = classification.detach() * 0.0
            route_rate = 0.0
        else:
            pair_logits = cgrh_state["pair_logits"]
            state_pairs = cgrh_state["pair_indices"]
            expected_pairs = torch.tensor(
                self.pair_indices,
                dtype=state_pairs.dtype,
                device=state_pairs.device,
            )
            if state_pairs.shape != expected_pairs.shape or not torch.equal(
                state_pairs, expected_pairs
            ):
                raise RuntimeError(
                    "CGRH model/loss pair definitions do not match"
                )
            pair_loss_sum = classification.float().sum() * 0.0
            pair_sample_count = 0
            for pair_index, (first, second) in enumerate(self.pair_indices):
                mask = (targets == first) | (targets == second)
                count = int(mask.sum().item())
                if count == 0:
                    continue
                binary_targets = (targets[mask] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    pair_logits[mask, pair_index].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count
            pair_loss = (
                pair_loss_sum / pair_sample_count
                if pair_sample_count > 0
                else pair_loss_sum
            )
            route_mask = cgrh_state["route_mask"]
            route_rate = float(route_mask.any(dim=1).float().mean().item())

        total = classification + self.auxiliary_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "cgrh_pair_loss": float(pair_loss.detach().item()),
            "cgrh_route_rate": route_rate,
        }
        return total


class CGRHAuxiliaryTrainingLoss(nn.Module):
    """Main CE plus full-sample supervision for fixed hard-pair experts.

    Unlike the original CGRH loss, the pair experts never refine inference
    logits.  Every sample whose ground-truth label belongs to a configured
    pair supervises that pair's binary expert during training.
    """

    requires_cgrh_aux = True

    def __init__(
        self,
        label_smoothing: float,
        pair_indices: list[tuple[int, int]],
        auxiliary_weight: float = 0.20,
    ) -> None:
        super().__init__()
        if not pair_indices:
            raise ValueError("CGRH-Aux requires fixed hard pairs")
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.pair_indices = [(int(a), int(b)) for a, b in pair_indices]
        self.auxiliary_weight = float(auxiliary_weight)
        self.last_components: dict[str, float] = {}

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_aux_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        supervised_mask = torch.zeros_like(targets, dtype=torch.bool)

        if cgrh_aux_state is None or self.auxiliary_weight <= 0.0:
            pair_loss = classification.detach() * 0.0
        else:
            pair_logits = cgrh_aux_state["pair_logits"]
            state_pairs = cgrh_aux_state["pair_indices"]
            expected_pairs = torch.tensor(
                self.pair_indices,
                dtype=state_pairs.dtype,
                device=state_pairs.device,
            )
            if state_pairs.shape != expected_pairs.shape or not torch.equal(
                state_pairs, expected_pairs
            ):
                raise RuntimeError(
                    "CGRH-Aux model/loss pair definitions do not match"
                )

            pair_loss_sum = classification.float().sum() * 0.0
            pair_sample_count = 0
            for pair_index, (first, second) in enumerate(self.pair_indices):
                mask = (targets == first) | (targets == second)
                supervised_mask |= mask
                count = int(mask.sum().item())
                if count == 0:
                    continue
                binary_targets = (targets[mask] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    pair_logits[mask, pair_index].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count
            pair_loss = (
                pair_loss_sum / pair_sample_count
                if pair_sample_count > 0
                else pair_loss_sum
            )

        total = classification + self.auxiliary_weight * pair_loss
        supervision_rate = float(supervised_mask.float().mean().item())
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "cgrh_aux_pair_loss": float(pair_loss.detach().item()),
            "cgrh_aux_supervision_rate": supervision_rate,
        }
        return total


class APDAuxiliaryClassificationLoss(nn.Module):
    """Main multi-class CE plus training-only all-pair binary supervision."""

    requires_apd_aux = True

    def __init__(
        self,
        label_smoothing: float,
        pair_indices: list[tuple[int, int]],
        auxiliary_weight: float = 0.20,
    ) -> None:
        super().__init__()
        if not pair_indices:
            raise ValueError("APD auxiliary loss requires class pairs")
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.pair_indices = [(int(a), int(b)) for a, b in pair_indices]
        self.auxiliary_weight = float(auxiliary_weight)
        self.last_components: dict[str, float] = {}

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        apd_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        if apd_state is None or self.auxiliary_weight <= 0.0:
            pair_loss = classification.detach() * 0.0
        else:
            pair_logits = apd_state["pair_logits"]
            state_pairs = apd_state["pair_indices"]
            expected_pairs = torch.tensor(
                self.pair_indices,
                dtype=state_pairs.dtype,
                device=state_pairs.device,
            )
            if state_pairs.shape != expected_pairs.shape or not torch.equal(
                state_pairs, expected_pairs
            ):
                raise RuntimeError("APD model/loss pair definitions do not match")
            pair_loss_sum = classification.float().sum() * 0.0
            pair_sample_count = 0
            for pair_index, (first, second) in enumerate(self.pair_indices):
                mask = (targets == first) | (targets == second)
                count = int(mask.sum().item())
                if count == 0:
                    continue
                binary_targets = (targets[mask] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    pair_logits[mask, pair_index].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count
            pair_loss = (
                pair_loss_sum / pair_sample_count
                if pair_sample_count > 0
                else pair_loss_sum
            )
        total = classification + self.auxiliary_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "apd_pair_loss": float(pair_loss.detach().item()),
        }
        return total


class AdaptiveOnlineConfusionGraphLoss(nn.Module):
    """Main CE plus stable epoch-level, EMA-observed hard-pair supervision.

    Pair discovery is deliberately decoupled from per-sample routing.  An EMA
    teacher contributes an epoch confusion matrix; a row-normalized symmetric
    graph is smoothed across epochs and updated only at a coarse interval.
    The selected pairs supervise a shared K-way auxiliary head during training.
    """

    requires_aocg = True

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        pair_weight: float = 0.05,
        warmup_epochs: int = 20,
        update_interval: int = 5,
        active_pair_count: int = 3,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 150,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("AOCG requires at least two classes")
        if len(class_names) != num_classes:
            raise ValueError("AOCG class_names must match num_classes")
        if pair_weight < 0.0:
            raise ValueError("AOCG pair_weight must be >= 0")
        if warmup_epochs < 0 or update_interval < 1:
            raise ValueError("Invalid AOCG warmup/update interval")
        if not 0.0 <= confusion_decay < 1.0:
            raise ValueError("AOCG confusion_decay must be in [0, 1)")
        if active_pair_count < 1 or entry_persistence < 1:
            raise ValueError("AOCG pair count/persistence must be >= 1")
        if exit_rank_multiplier < 1:
            raise ValueError("AOCG exit_rank_multiplier must be >= 1")

        self.num_classes = int(num_classes)
        self.class_names = [str(name) for name in class_names]
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.pair_weight = float(pair_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.active_pair_count = min(
            int(active_pair_count),
            self.num_classes * (self.num_classes - 1) // 2,
        )
        self.confusion_decay = float(confusion_decay)
        self.freeze_pairs_epoch = int(freeze_pairs_epoch)
        self.min_pair_score = float(min_pair_score)
        self.min_class_observations = int(min_class_observations)
        self.entry_persistence = int(entry_persistence)
        self.exit_rank_multiplier = int(exit_rank_multiplier)
        self.current_epoch = 0
        self.last_components: dict[str, float] = {}
        self._active_pairs: list[tuple[int, int]] = []
        self._selection_history: list[dict[str, Any]] = []
        self._has_smoothed_confusion = False
        self._observed_samples = 0

        shape = (self.num_classes, self.num_classes)
        self.register_buffer("epoch_confusion", torch.zeros(shape))
        self.register_buffer("smoothed_confusion", torch.zeros(shape))
        self.register_buffer(
            "candidate_streaks", torch.zeros(shape, dtype=torch.long)
        )

    @torch.no_grad()
    def observe_teacher(
        self,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate one EMA-teacher confusion matrix on training samples."""
        if self.current_epoch > self.freeze_pairs_epoch:
            return
        predictions = teacher_logits.detach().argmax(dim=1)
        flat_indices = targets.detach() * self.num_classes + predictions
        counts = torch.bincount(
            flat_indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.epoch_confusion.add_(counts.to(self.epoch_confusion.dtype))
        self._observed_samples += int(targets.numel())

    def should_observe_teacher(self) -> bool:
        return self.current_epoch <= self.freeze_pairs_epoch

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        scored_pairs: list[tuple[float, int, int]] = []
        for first in range(self.num_classes):
            if int(class_counts[first].item()) < self.min_class_observations:
                continue
            for second in range(first + 1, self.num_classes):
                if int(class_counts[second].item()) < self.min_class_observations:
                    continue
                score = float(score_matrix[first, second].item())
                if score >= self.min_pair_score:
                    scored_pairs.append((score, first, second))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        entry_pairs = {
            (first, second)
            for _, first, second in scored_pairs[: self.active_pair_count]
        }
        exit_limit = self.active_pair_count * self.exit_rank_multiplier
        exit_pairs = {
            (first, second)
            for _, first, second in scored_pairs[:exit_limit]
        }

        new_streaks = torch.zeros_like(self.candidate_streaks)
        for first, second in entry_pairs:
            streak = int(self.candidate_streaks[first, second].item()) + 1
            new_streaks[first, second] = streak
            new_streaks[second, first] = streak
        self.candidate_streaks.copy_(new_streaks)

        retained = [pair for pair in self._active_pairs if pair in exit_pairs]
        for _, first, second in scored_pairs:
            pair = (first, second)
            if pair in retained or pair not in entry_pairs:
                continue
            if int(self.candidate_streaks[first, second].item()) < (
                self.entry_persistence
            ):
                continue
            retained.append(pair)
            if len(retained) >= self.active_pair_count:
                break
        self._active_pairs = sorted(
            retained,
            key=lambda pair: float(score_matrix[pair[0], pair[1]].item()),
            reverse=True,
        )[: self.active_pair_count]

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "active_pairs": self.active_pair_records(score_matrix),
                "top_candidates": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "entry_streak": int(
                            self.candidate_streaks[first, second].item()
                        ),
                    }
                    for score, first, second in scored_pairs[
                        : self.active_pair_count
                    ]
                ],
            }
        )

    @torch.no_grad()
    def _finalize_previous_epoch(self, source_epoch: int) -> None:
        if self._observed_samples <= 0:
            return
        class_counts = self.epoch_confusion.sum(dim=1)
        normalized = self.epoch_confusion / class_counts.clamp_min(1.0).unsqueeze(1)
        symmetric_scores = 0.5 * (normalized + normalized.transpose(0, 1))
        symmetric_scores.fill_diagonal_(0.0)

        if self._has_smoothed_confusion:
            self.smoothed_confusion.mul_(self.confusion_decay).add_(
                symmetric_scores,
                alpha=1.0 - self.confusion_decay,
            )
        else:
            self.smoothed_confusion.copy_(symmetric_scores)
            self._has_smoothed_confusion = True

        should_update = (
            source_epoch >= self.warmup_epochs
            and source_epoch < self.freeze_pairs_epoch
            and (source_epoch - self.warmup_epochs) % self.update_interval == 0
        )
        if should_update:
            self._select_active_pairs(
                self.smoothed_confusion,
                class_counts,
                source_epoch,
            )

        self.epoch_confusion.zero_()
        self._observed_samples = 0

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if self.current_epoch > 0 and epoch != self.current_epoch:
            self._finalize_previous_epoch(self.current_epoch)
        self.current_epoch = epoch

    def active_pair_records(
        self,
        score_matrix: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.smoothed_confusion if score_matrix is None else score_matrix
        return [
            {
                "class_i": self.class_names[first],
                "class_j": self.class_names[second],
                "class_i_index": int(first),
                "class_j_index": int(second),
                "score": float(scores[first, second].item()),
            }
            for first, second in self._active_pairs
        ]

    def graph_state(self) -> dict[str, Any]:
        return {
            "current_epoch": int(self.current_epoch),
            "active_pairs": self.active_pair_records(),
            "selection_history": list(self._selection_history),
        }

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        aocg_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        pair_loss_sum = classification.float().sum() * 0.0
        pair_sample_count = 0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        guidance_enabled = (
            aocg_state is not None
            and self.current_epoch > self.warmup_epochs
            and self.pair_weight > 0.0
            and bool(self._active_pairs)
        )
        if guidance_enabled:
            aux_logits = aocg_state["aux_logits"]
            for first, second in self._active_pairs:
                mask = (targets == first) | (targets == second)
                count = int(mask.sum().item())
                if count == 0:
                    continue
                supervised_samples |= mask
                binary_logits = aux_logits[mask][:, [first, second]]
                binary_targets = (targets[mask] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    binary_logits.float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count

        pair_loss = (
            pair_loss_sum / pair_sample_count
            if pair_sample_count > 0
            else pair_loss_sum
        )
        total = classification + self.pair_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "aocg_pair_loss": float(pair_loss.detach().item()),
            "aocg_active_pair_count": float(len(self._active_pairs)),
            "aocg_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
        }
        return total


class OnlineDynamicPairCGRHTrainOnlyLoss(AdaptiveOnlineConfusionGraphLoss):
    """Online pair mining with train-time CGRH logit refinement.

    Pair discovery reuses the stable EMA-teacher confusion graph from AOCG.
    For active pairs, all ground-truth pair samples train the corresponding
    binary experts.  The refined multiclass logits supply an additional CE
    term during training, but validation and deployment select/use the main
    classifier only.
    """

    requires_odpm_cgrh = True
    requires_aocg = False
    graph_history_filename = "odpm_cgrh_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        pair_weight: float = 0.15,
        refine_weight: float = 0.25,
        warmup_epochs: int = 30,
        update_interval: int = 5,
        active_pair_count: int = 3,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 150,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=pair_weight,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=freeze_pairs_epoch,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
        )
        if refine_weight < 0.0:
            raise ValueError("ODPM-CGRH refine_weight must be >= 0")
        self.refine_weight = float(refine_weight)

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state["variant"] = "odpm_cgrh_trainonly"
        state["pair_weight"] = float(self.pair_weight)
        state["refine_weight"] = float(self.refine_weight)
        return state

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        odpm_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if odpm_state is None:
            main_logits = refined_logits
        else:
            main_logits = odpm_state["main_logits"]
        classification = self.cross_entropy(main_logits, targets)
        pair_loss_sum = classification.float().sum() * 0.0
        pair_sample_count = 0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        guidance_enabled = (
            odpm_state is not None
            and self.current_epoch > self.warmup_epochs
            and bool(self._active_pairs)
        )
        if guidance_enabled and self.pair_weight > 0.0:
            pair_logits = odpm_state["pair_logits"]
            pair_indices = odpm_state["pair_indices"]
            pair_to_slot = {
                tuple(int(value) for value in pair): slot
                for slot, pair in enumerate(pair_indices.tolist())
            }
            for first, second in self._active_pairs:
                slot = pair_to_slot[tuple(sorted((first, second)))]
                mask = (targets == first) | (targets == second)
                count = int(mask.sum().item())
                if count == 0:
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count

        pair_loss = (
            pair_loss_sum / pair_sample_count
            if pair_sample_count > 0
            else pair_loss_sum
        )
        routed = (
            odpm_state["route_mask"].any(dim=1)
            if odpm_state is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        if guidance_enabled and self.refine_weight > 0.0 and bool(routed.any()):
            # Refined CE is limited to actually routed samples; main CE still
            # supervises every sample and is the checkpoint-selection target.
            refine_loss = F.cross_entropy(
                refined_logits[routed].float(),
                targets[routed],
            )
        else:
            refine_loss = classification.float().sum() * 0.0

        total = (
            classification
            + self.pair_weight * pair_loss
            + self.refine_weight * refine_loss
        )
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "odpm_cgrh_pair_loss": float(pair_loss.detach().item()),
            "odpm_cgrh_refine_loss": float(refine_loss.detach().item()),
            "odpm_cgrh_active_pair_count": float(len(self._active_pairs)),
            "odpm_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "odpm_cgrh_route_rate": float(
                routed.float().mean().detach().item()
            ),
        }
        return total


class OnlineDynamicPairCGRHHardV2Loss(AdaptiveOnlineConfusionGraphLoss):
    """Sparse hard-sample pair experts with main-head checkpoint selection.

    Online pair discovery is identical to ODPM-CGRH V1.  Unlike V1, a sample
    is eligible only when its ground-truth belongs to an active pair, the
    deployable main head's Top-2 exactly matches that pair, and its probability
    gap is small.  Within every (pair, true-class) group, only the configured
    hardest fraction is retained.  Both binary-expert CE and refined-logit CE
    use precisely this sparse mask.
    """

    requires_odpm_cgrh_hard_v2 = True
    requires_aocg = False
    graph_history_filename = "odpm_cgrh_hard_v2_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        pair_weight: float = 0.05,
        refine_weight: float = 0.10,
        probability_margin_threshold: float = 0.20,
        hard_fraction: float = 0.50,
        min_samples_per_group: int = 1,
        max_samples_per_group: int = 0,
        warmup_epochs: int = 30,
        update_interval: int = 10,
        active_pair_count: int = 3,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 90,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=pair_weight,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=freeze_pairs_epoch,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
        )
        if refine_weight < 0.0:
            raise ValueError("Hard-V2 refine_weight must be >= 0")
        if not 0.0 < probability_margin_threshold <= 1.0:
            raise ValueError(
                "Hard-V2 probability_margin_threshold must be in (0, 1]"
            )
        if not 0.0 < hard_fraction <= 1.0:
            raise ValueError("Hard-V2 hard_fraction must be in (0, 1]")
        if min_samples_per_group < 0 or max_samples_per_group < 0:
            raise ValueError("Hard-V2 sample limits must be >= 0")
        self.refine_weight = float(refine_weight)
        self.probability_margin_threshold = float(
            probability_margin_threshold
        )
        self.hard_fraction = float(hard_fraction)
        self.min_samples_per_group = int(min_samples_per_group)
        self.max_samples_per_group = int(max_samples_per_group)

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "odpm_cgrh_hard_v2",
                "pair_weight": float(self.pair_weight),
                "refine_weight": float(self.refine_weight),
                "probability_margin_threshold": float(
                    self.probability_margin_threshold
                ),
                "hard_fraction": float(self.hard_fraction),
                "min_samples_per_group": int(self.min_samples_per_group),
                "max_samples_per_group": int(self.max_samples_per_group),
            }
        )
        return state

    def _hard_group_mask(
        self,
        candidate_mask: torch.Tensor,
        probability_gaps: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the lowest-gap samples from one pair/true-class group."""
        candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(1)
        count = int(candidate_indices.numel())
        selected = torch.zeros_like(candidate_mask)
        if count == 0:
            return selected
        keep = max(
            self.min_samples_per_group,
            int(torch.ceil(torch.tensor(count * self.hard_fraction)).item()),
        )
        keep = min(keep, count)
        if self.max_samples_per_group > 0:
            keep = min(keep, self.max_samples_per_group)
        if keep <= 0:
            return selected
        local_order = torch.topk(
            probability_gaps[candidate_indices],
            k=keep,
            largest=False,
        ).indices
        selected[candidate_indices[local_order]] = True
        return selected

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        odpm_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            odpm_state["main_logits"]
            if odpm_state is not None
            else refined_logits
        )
        classification = self.cross_entropy(main_logits, targets)
        zero = classification.float().sum() * 0.0
        pair_loss_sum = zero
        pair_sample_count = 0
        active_class_mask = torch.zeros_like(targets, dtype=torch.bool)
        top2_match_mask = torch.zeros_like(targets, dtype=torch.bool)
        low_margin_mask = torch.zeros_like(targets, dtype=torch.bool)
        selected_mask = torch.zeros_like(targets, dtype=torch.bool)

        guidance_enabled = (
            odpm_state is not None
            and self.current_epoch > self.warmup_epochs
            and bool(self._active_pairs)
        )
        if guidance_enabled:
            with torch.no_grad():
                probabilities = torch.softmax(main_logits.detach().float(), dim=1)
                top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
                probability_gaps = top_probabilities[:, 0] - top_probabilities[:, 1]
                global_low_margin = (
                    probability_gaps <= self.probability_margin_threshold
                )

            pair_logits = odpm_state["pair_logits"]
            pair_indices = odpm_state["pair_indices"]
            pair_to_slot = {
                tuple(int(value) for value in pair): slot
                for slot, pair in enumerate(pair_indices.tolist())
            }
            for first, second in self._active_pairs:
                target_in_pair = (targets == first) | (targets == second)
                active_class_mask |= target_in_pair
                top2_matches = (
                    ((top_indices[:, 0] == first) & (top_indices[:, 1] == second))
                    | ((top_indices[:, 0] == second) & (top_indices[:, 1] == first))
                )
                top2_match_mask |= target_in_pair & top2_matches
                pair_candidates = target_in_pair & top2_matches & global_low_margin
                low_margin_mask |= pair_candidates

                pair_selected = torch.zeros_like(targets, dtype=torch.bool)
                # Per-class quotas prevent a dominant class from consuming
                # every hard sample retained for this pair.
                for true_class in (first, second):
                    group = pair_candidates & (targets == true_class)
                    pair_selected |= self._hard_group_mask(
                        group,
                        probability_gaps,
                    )
                if not bool(pair_selected.any()):
                    continue
                selected_mask |= pair_selected
                slot = pair_to_slot[tuple(sorted((first, second)))]
                binary_targets = (targets[pair_selected] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    pair_logits[pair_selected, slot].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += int(pair_selected.sum().item())

        pair_loss = (
            pair_loss_sum / pair_sample_count
            if pair_sample_count > 0
            else pair_loss_sum
        )
        if (
            guidance_enabled
            and self.refine_weight > 0.0
            and bool(selected_mask.any())
        ):
            refine_loss = F.cross_entropy(
                refined_logits[selected_mask].float(),
                targets[selected_mask],
            )
        else:
            refine_loss = zero

        total = (
            classification
            + self.pair_weight * pair_loss
            + self.refine_weight * refine_loss
        )
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "odpm_hard_v2_pair_loss": float(pair_loss.detach().item()),
            "odpm_hard_v2_refine_loss": float(refine_loss.detach().item()),
            "odpm_hard_v2_active_pair_count": float(len(self._active_pairs)),
            "odpm_hard_v2_active_class_rate": float(
                active_class_mask.float().mean().detach().item()
            ),
            "odpm_hard_v2_top2_match_rate": float(
                top2_match_mask.float().mean().detach().item()
            ),
            "odpm_hard_v2_low_margin_rate": float(
                low_margin_mask.float().mean().detach().item()
            ),
            "odpm_hard_v2_supervision_rate": float(
                selected_mask.float().mean().detach().item()
            ),
            "odpm_hard_v2_selected_samples": float(
                selected_mask.sum().detach().item()
            ),
        }
        return total


class OnlineDynamicPairCGRHCoverageV3Loss(
    AdaptiveOnlineConfusionGraphLoss
):
    """Dynamic discovery followed by CGRH-style high-coverage supervision.

    The EMA main head discovers stable training-set confusion pairs.  Every
    sample whose ground-truth belongs to an active pair supervises that pair's
    binary expert; Top-2 matching and low-margin filtering are deliberately not
    used for the pair loss.  If one class participates in several pairs, its
    per-sample auxiliary loss is averaged across those pairs so repeated edges
    cannot amplify that class's backbone gradient.

    Validation, checkpoint selection, testing, and deployment use main logits
    only.  All experts therefore remain a removable training-time regularizer.
    """

    requires_odpm_cgrh_coverage_v3 = True
    requires_aocg = False
    graph_history_filename = "odpm_cgrh_coverage_v3_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        dynamic_pair_weight: float = 0.05,
        max_pair_weight: float = 0.10,
        final_pair_weight: float = 0.02,
        warmup_epochs: int = 30,
        dynamic_ramp_end_epoch: int = 60,
        update_interval: int = 10,
        active_pair_count: int = 3,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 90,
        frozen_ramp_end_epoch: int = 140,
        decay_start_epoch: int = 141,
        decay_end_epoch: int = 200,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=max_pair_weight,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=freeze_pairs_epoch,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
        )
        if not (
            0.0 <= final_pair_weight
            <= dynamic_pair_weight
            <= max_pair_weight
        ):
            raise ValueError(
                "Coverage-V3 weights must satisfy "
                "0 <= final <= dynamic <= max"
            )
        if not (
            warmup_epochs < dynamic_ramp_end_epoch <= freeze_pairs_epoch
            < frozen_ramp_end_epoch < decay_start_epoch <= decay_end_epoch
        ):
            raise ValueError("Coverage-V3 epoch schedule is inconsistent")
        self.dynamic_pair_weight = float(dynamic_pair_weight)
        self.max_pair_weight = float(max_pair_weight)
        self.final_pair_weight = float(final_pair_weight)
        self.dynamic_ramp_end_epoch = int(dynamic_ramp_end_epoch)
        self.frozen_ramp_end_epoch = int(frozen_ramp_end_epoch)
        self.decay_start_epoch = int(decay_start_epoch)
        self.decay_end_epoch = int(decay_end_epoch)

    @torch.no_grad()
    def _finalize_previous_epoch(self, source_epoch: int) -> None:
        """Include epoch ``freeze_pairs_epoch`` in the final graph update."""
        if int(source_epoch) != self.freeze_pairs_epoch:
            super()._finalize_previous_epoch(source_epoch)
            return
        original_freeze_epoch = self.freeze_pairs_epoch
        self.freeze_pairs_epoch = original_freeze_epoch + 1
        try:
            super()._finalize_previous_epoch(source_epoch)
        finally:
            self.freeze_pairs_epoch = original_freeze_epoch

    def effective_pair_weight(self) -> float:
        epoch = self.current_epoch
        if epoch <= self.warmup_epochs:
            return 0.0
        if epoch <= self.dynamic_ramp_end_epoch:
            span = max(self.dynamic_ramp_end_epoch - self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / span
            return self.dynamic_pair_weight * min(max(progress, 0.0), 1.0)
        if epoch <= self.freeze_pairs_epoch:
            return self.dynamic_pair_weight
        if epoch <= self.frozen_ramp_end_epoch:
            span = max(
                self.frozen_ramp_end_epoch - self.freeze_pairs_epoch,
                1,
            )
            progress = (epoch - self.freeze_pairs_epoch) / span
            return self.dynamic_pair_weight + progress * (
                self.max_pair_weight - self.dynamic_pair_weight
            )
        if epoch < self.decay_start_epoch:
            return self.max_pair_weight
        if epoch <= self.decay_end_epoch:
            span = max(self.decay_end_epoch - self.decay_start_epoch, 1)
            progress = (epoch - self.decay_start_epoch) / span
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_pair_weight + cosine * (
                self.max_pair_weight - self.final_pair_weight
            )
        return self.final_pair_weight

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "odpm_cgrh_coverage_v3",
                "effective_pair_weight": float(self.effective_pair_weight()),
                "dynamic_pair_weight": float(self.dynamic_pair_weight),
                "max_pair_weight": float(self.max_pair_weight),
                "final_pair_weight": float(self.final_pair_weight),
                "dynamic_ramp_end_epoch": int(self.dynamic_ramp_end_epoch),
                "freeze_pairs_epoch": int(self.freeze_pairs_epoch),
                "frozen_ramp_end_epoch": int(self.frozen_ramp_end_epoch),
                "decay_start_epoch": int(self.decay_start_epoch),
                "decay_end_epoch": int(self.decay_end_epoch),
                "pairs_frozen": bool(
                    self.current_epoch > self.freeze_pairs_epoch
                ),
            }
        )
        return state

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        odpm_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        zero = classification.float().sum() * 0.0
        pair_loss_sum = zero
        pair_instance_count = 0
        sample_pair_counts = torch.zeros_like(targets, dtype=torch.long)
        effective_weight = self.effective_pair_weight()

        guidance_enabled = (
            odpm_state is not None
            and self.current_epoch > self.warmup_epochs
            and effective_weight > 0.0
            and bool(self._active_pairs)
        )
        active_masks: list[tuple[int, int, torch.Tensor]] = []
        if guidance_enabled:
            for first, second in self._active_pairs:
                mask = (targets == first) | (targets == second)
                if not bool(mask.any()):
                    continue
                active_masks.append((first, second, mask))
                sample_pair_counts[mask] += 1
                pair_instance_count += int(mask.sum().item())

            if active_masks:
                pair_logits = odpm_state["pair_logits"]
                pair_indices = odpm_state["pair_indices"]
                pair_to_slot = {
                    tuple(int(value) for value in pair): slot
                    for slot, pair in enumerate(pair_indices.tolist())
                }
                supervised_sample_count = int(
                    (sample_pair_counts > 0).sum().item()
                )
                for first, second, mask in active_masks:
                    slot = pair_to_slot[tuple(sorted((first, second)))]
                    binary_targets = (targets[mask] == second).long()
                    per_instance = F.cross_entropy(
                        pair_logits[mask, slot].float(),
                        binary_targets,
                        reduction="none",
                    )
                    # A sample shared by multiple graph edges retains total
                    # auxiliary mass one instead of being over-weighted.
                    normalizer = sample_pair_counts[mask].to(
                        per_instance.dtype
                    )
                    pair_loss_sum = pair_loss_sum + (
                        per_instance / normalizer
                    ).sum()
                pair_loss = pair_loss_sum / max(supervised_sample_count, 1)
            else:
                pair_loss = zero
        else:
            pair_loss = zero

        supervised_mask = sample_pair_counts > 0
        supervised_count = int(supervised_mask.sum().item())
        batch_size = max(int(targets.shape[0]), 1)
        mean_multiplicity = (
            float(
                sample_pair_counts[supervised_mask]
                .float()
                .mean()
                .detach()
                .item()
            )
            if supervised_count > 0
            else 0.0
        )
        total = classification + effective_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "coverage_v3_pair_loss": float(pair_loss.detach().item()),
            "coverage_v3_effective_pair_weight": float(effective_weight),
            "coverage_v3_active_pair_count": float(len(self._active_pairs)),
            "coverage_v3_supervision_rate": float(
                supervised_count / batch_size
            ),
            "coverage_v3_supervision_instance_rate": float(
                pair_instance_count / batch_size
            ),
            "coverage_v3_mean_pair_multiplicity": mean_multiplicity,
            "coverage_v3_pairs_frozen": float(
                self.current_epoch > self.freeze_pairs_epoch
            ),
        }
        return total


class DynamicConfusionMarginGuidanceLoss(nn.Module):
    """Online soft-confusion discovery with main-logit margin guidance.

    An EMA teacher estimates a class-level soft confusion graph using only
    training samples.  Stable Top-K pairs are selected at epoch boundaries.
    For every supervised sample, at most one graph neighbour is used as the
    hard competitor, and the margin is applied directly to the deployable
    main classifier logits.  No auxiliary model branch is required.
    """

    requires_dcmg = True
    graph_history_filename = "dcmg_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        pair_weight: float = 0.05,
        margin: float = 0.10,
        warmup_epochs: int = 20,
        update_interval: int = 5,
        active_pair_count: int = 3,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 150,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
        score_weighted: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("DCMG requires at least two classes")
        if len(class_names) != num_classes:
            raise ValueError("DCMG class_names must match num_classes")
        if pair_weight < 0.0 or margin < 0.0:
            raise ValueError("DCMG pair_weight and margin must be >= 0")
        if warmup_epochs < 0 or update_interval < 1:
            raise ValueError("Invalid DCMG warmup/update interval")
        if not 0.0 <= confusion_decay < 1.0:
            raise ValueError("DCMG confusion_decay must be in [0, 1)")
        if active_pair_count < 1 or entry_persistence < 1:
            raise ValueError("DCMG pair count/persistence must be >= 1")
        if exit_rank_multiplier < 1:
            raise ValueError("DCMG exit_rank_multiplier must be >= 1")

        self.num_classes = int(num_classes)
        self.class_names = [str(name) for name in class_names]
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.pair_weight = float(pair_weight)
        self.margin = float(margin)
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.active_pair_count = min(
            int(active_pair_count),
            self.num_classes * (self.num_classes - 1) // 2,
        )
        self.confusion_decay = float(confusion_decay)
        self.freeze_pairs_epoch = int(freeze_pairs_epoch)
        self.min_pair_score = float(min_pair_score)
        self.min_class_observations = int(min_class_observations)
        self.entry_persistence = int(entry_persistence)
        self.exit_rank_multiplier = int(exit_rank_multiplier)
        self.score_weighted = bool(score_weighted)
        self.current_epoch = 0
        self.last_components: dict[str, float] = {}
        self._active_pairs: list[tuple[int, int]] = []
        self._selection_history: list[dict[str, Any]] = []
        self._has_smoothed_confusion = False

        shape = (self.num_classes, self.num_classes)
        self.register_buffer("epoch_probability_sums", torch.zeros(shape))
        self.register_buffer("epoch_class_counts", torch.zeros(self.num_classes))
        self.register_buffer("smoothed_confusion", torch.zeros(shape))
        self.register_buffer(
            "candidate_streaks", torch.zeros(shape, dtype=torch.long)
        )

    def should_observe_teacher(self) -> bool:
        return (
            self.freeze_pairs_epoch <= 0
            or self.current_epoch < self.freeze_pairs_epoch
        )

    @torch.no_grad()
    def observe_teacher(
        self,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate EMA-teacher class probabilities by ground-truth row."""
        if not self.should_observe_teacher():
            return
        probabilities = torch.softmax(teacher_logits.detach().float(), dim=1)
        detached_targets = targets.detach().long()
        self.epoch_probability_sums.index_add_(
            0, detached_targets, probabilities
        )
        self.epoch_class_counts.add_(
            torch.bincount(
                detached_targets,
                minlength=self.num_classes,
            ).to(self.epoch_class_counts.dtype)
        )

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        scored_pairs: list[tuple[float, int, int]] = []
        for first in range(self.num_classes):
            if int(class_counts[first].item()) < self.min_class_observations:
                continue
            for second in range(first + 1, self.num_classes):
                if int(class_counts[second].item()) < self.min_class_observations:
                    continue
                score = float(score_matrix[first, second].item())
                if score >= self.min_pair_score:
                    scored_pairs.append((score, first, second))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        entry_pairs = {
            (first, second)
            for _, first, second in scored_pairs[: self.active_pair_count]
        }
        exit_limit = self.active_pair_count * self.exit_rank_multiplier
        exit_pairs = {
            (first, second)
            for _, first, second in scored_pairs[:exit_limit]
        }

        new_streaks = torch.zeros_like(self.candidate_streaks)
        for first, second in entry_pairs:
            streak = int(self.candidate_streaks[first, second].item()) + 1
            new_streaks[first, second] = streak
            new_streaks[second, first] = streak
        self.candidate_streaks.copy_(new_streaks)

        retained = [pair for pair in self._active_pairs if pair in exit_pairs]
        for _, first, second in scored_pairs:
            pair = (first, second)
            if pair in retained or pair not in entry_pairs:
                continue
            if int(self.candidate_streaks[first, second].item()) < (
                self.entry_persistence
            ):
                continue
            retained.append(pair)
            if len(retained) >= self.active_pair_count:
                break
        self._active_pairs = sorted(
            retained,
            key=lambda pair: float(score_matrix[pair[0], pair[1]].item()),
            reverse=True,
        )[: self.active_pair_count]

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "active_pairs": self.active_pair_records(score_matrix),
                "top_candidates": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "entry_streak": int(
                            self.candidate_streaks[first, second].item()
                        ),
                    }
                    for score, first, second in scored_pairs[
                        : self.active_pair_count
                    ]
                ],
            }
        )

    @torch.no_grad()
    def _finalize_previous_epoch(self, source_epoch: int) -> None:
        if float(self.epoch_class_counts.sum().item()) <= 0.0:
            return
        mean_probabilities = self.epoch_probability_sums / (
            self.epoch_class_counts.clamp_min(1.0).unsqueeze(1)
        )
        symmetric_scores = 0.5 * (
            mean_probabilities + mean_probabilities.transpose(0, 1)
        )
        symmetric_scores.fill_diagonal_(0.0)

        if self._has_smoothed_confusion:
            self.smoothed_confusion.mul_(self.confusion_decay).add_(
                symmetric_scores,
                alpha=1.0 - self.confusion_decay,
            )
        else:
            self.smoothed_confusion.copy_(symmetric_scores)
            self._has_smoothed_confusion = True

        should_update = (
            source_epoch >= self.warmup_epochs
            and (
                self.freeze_pairs_epoch <= 0
                or source_epoch < self.freeze_pairs_epoch
            )
            and (source_epoch - self.warmup_epochs) % self.update_interval == 0
        )
        if should_update:
            self._select_active_pairs(
                self.smoothed_confusion,
                self.epoch_class_counts,
                source_epoch,
            )

        self.epoch_probability_sums.zero_()
        self.epoch_class_counts.zero_()

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if self.current_epoch > 0 and epoch != self.current_epoch:
            self._finalize_previous_epoch(self.current_epoch)
        self.current_epoch = epoch

    def active_pair_records(
        self,
        score_matrix: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.smoothed_confusion if score_matrix is None else score_matrix
        return [
            {
                "class_i": self.class_names[first],
                "class_j": self.class_names[second],
                "class_i_index": int(first),
                "class_j_index": int(second),
                "score": float(scores[first, second].item()),
            }
            for first, second in self._active_pairs
        ]

    def graph_state(self) -> dict[str, Any]:
        return {
            "current_epoch": int(self.current_epoch),
            "margin": float(self.margin),
            "pair_weight": float(self.pair_weight),
            "active_pairs": self.active_pair_records(),
            "selection_history": list(self._selection_history),
        }

    def _class_competitors(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        competitors = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=device
        )
        competitor_scores = torch.zeros(
            self.num_classes,
            dtype=self.smoothed_confusion.dtype,
            device=device,
        )
        scores = self.smoothed_confusion.to(device=device)
        for first, second in self._active_pairs:
            score = scores[first, second]
            if bool(score > competitor_scores[first]):
                competitors[first] = second
                competitor_scores[first] = score
            if bool(score > competitor_scores[second]):
                competitors[second] = first
                competitor_scores[second] = score
        return competitors, competitor_scores

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        margin_loss = classification.float().sum() * 0.0
        supervision_rate = 0.0
        violation_rate = 0.0
        mean_logit_gap = 0.0

        guidance_enabled = (
            self.current_epoch > self.warmup_epochs
            and self.pair_weight > 0.0
            and bool(self._active_pairs)
        )
        if guidance_enabled:
            competitors, competitor_scores = self._class_competitors(
                main_logits.device
            )
            sample_competitors = competitors[targets]
            supervised_mask = sample_competitors >= 0
            supervision_rate = float(
                supervised_mask.float().mean().detach().item()
            )
            if bool(supervised_mask.any()):
                row_indices = torch.arange(
                    targets.shape[0], device=targets.device
                )[supervised_mask]
                true_indices = targets[supervised_mask]
                hard_indices = sample_competitors[supervised_mask]
                true_logits = main_logits[row_indices, true_indices].float()
                hard_logits = main_logits[row_indices, hard_indices].float()
                logit_gaps = true_logits - hard_logits
                per_sample_loss = F.softplus(self.margin - logit_gaps)
                if self.score_weighted:
                    sample_scores = competitor_scores[true_indices].float()
                    normalized_weights = sample_scores / (
                        sample_scores.mean().clamp_min(1e-8)
                    )
                    margin_loss = (
                        per_sample_loss * normalized_weights
                    ).sum() / normalized_weights.sum().clamp_min(1e-8)
                else:
                    margin_loss = per_sample_loss.mean()
                violation_rate = float(
                    (logit_gaps < self.margin).float().mean().detach().item()
                )
                mean_logit_gap = float(logit_gaps.mean().detach().item())

        total = classification + self.pair_weight * margin_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "dcmg_margin_loss": float(margin_loss.detach().item()),
            "dcmg_active_pair_count": float(len(self._active_pairs)),
            "dcmg_supervision_rate": supervision_rate,
            "dcmg_violation_rate": violation_rate,
            "dcmg_mean_logit_gap": mean_logit_gap,
        }
        return total


class AutoCGRHClassificationLoss(DynamicConfusionMarginGuidanceLoss):
    """EMA soft-confusion discovery followed by frozen CGRH training.

    Discovery is observational: before ``discovery_epochs`` the deployable
    classifier receives only its normal multi-class CE. At the phase boundary,
    temporal votes select a variable-size consensus graph. The graph is frozen
    and the corresponding pre-allocated CGRH experts receive the same
    full-pair supervision as the original fixed-pair CGRH.
    """

    requires_auto_cgrh = True
    requires_dcmg = False
    graph_history_filename = "auto_cgrh_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        discovery_epochs: int = 60,
        discovery_warmup_epochs: int = 20,
        update_interval: int = 5,
        max_active_pairs: int = 3,
        min_active_pairs: int = 2,
        confusion_decay: float = 0.90,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
        min_consensus_rate: float = 0.60,
        auxiliary_weight: float = 0.20,
        pair_ramp_epochs: int = 0,
        stable_pair_gating: bool = False,
    ) -> None:
        if discovery_epochs < 1:
            raise ValueError("Auto-CGRH discovery_epochs must be >= 1")
        if not 0 <= discovery_warmup_epochs < discovery_epochs:
            raise ValueError(
                "Auto-CGRH discovery_warmup_epochs must be in "
                "[0, discovery_epochs)"
            )
        if not 0.0 <= min_consensus_rate <= 1.0:
            raise ValueError("Auto-CGRH min_consensus_rate must be in [0, 1]")
        if not 1 <= min_active_pairs <= max_active_pairs:
            raise ValueError(
                "Auto-CGRH requires 1 <= min_active_pairs <= max_active_pairs"
            )
        if auxiliary_weight < 0.0:
            raise ValueError("Auto-CGRH auxiliary_weight must be >= 0")
        if pair_ramp_epochs < 0:
            raise ValueError("Auto-CGRH pair_ramp_epochs must be >= 0")

        # freeze_pairs_epoch is exclusive in the DCMG observer. Setting it one
        # epoch after the discovery boundary includes all of epoch N.
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=0.0,
            margin=0.0,
            warmup_epochs=discovery_warmup_epochs,
            update_interval=update_interval,
            active_pair_count=max_active_pairs,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=discovery_epochs + 1,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
            score_weighted=False,
        )
        self.discovery_epochs = int(discovery_epochs)
        self.discovery_warmup_epochs = int(discovery_warmup_epochs)
        self.min_active_pairs = int(min_active_pairs)
        self.min_consensus_rate = float(min_consensus_rate)
        self.auxiliary_weight = float(auxiliary_weight)
        self.pair_ramp_epochs = int(pair_ramp_epochs)
        self.stable_pair_gating = bool(stable_pair_gating)
        self._pairs_frozen = False
        self._frozen_epoch: int | None = None
        self.register_buffer(
            "pair_vote_counts",
            torch.zeros(
                (self.num_classes, self.num_classes),
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "selection_update_count",
            torch.zeros((), dtype=torch.long),
        )

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        super()._select_active_pairs(
            score_matrix,
            class_counts,
            source_epoch,
        )
        if not self._selection_history:
            return
        self.selection_update_count.add_(1)
        for record in self._selection_history[-1]["top_candidates"]:
            first = int(record["class_i_index"])
            second = int(record["class_j_index"])
            self.pair_vote_counts[first, second].add_(1)
            self.pair_vote_counts[second, first].add_(1)

    @torch.no_grad()
    def _freeze_consensus_pairs(self) -> None:
        updates = max(int(self.selection_update_count.item()), 1)
        candidates: list[tuple[float, float, int, int]] = []
        fallbacks: list[tuple[float, float, int, int]] = []
        stable_pairs = set(self._active_pairs)
        for first in range(self.num_classes):
            for second in range(first + 1, self.num_classes):
                score = float(self.smoothed_confusion[first, second].item())
                vote_rate = (
                    float(self.pair_vote_counts[first, second].item()) / updates
                )
                item = (vote_rate, score, first, second)
                if score >= self.min_pair_score:
                    fallbacks.append(item)
                    stable_enough = (
                        not self.stable_pair_gating
                        or (first, second) in stable_pairs
                    )
                    if (
                        stable_enough
                        and vote_rate >= self.min_consensus_rate
                    ):
                        candidates.append(item)

        candidates.sort(reverse=True)
        fallbacks.sort(reverse=True)
        selected = [
            (first, second)
            for _, _, first, second in candidates[: self.active_pair_count]
        ]
        if self.stable_pair_gating:
            # Prefer pairs that passed entry persistence and survived the
            # configured exit-rank hysteresis.  Global score fallbacks are
            # used only when the stable graph cannot satisfy min_active_pairs.
            stable_fallbacks = [
                item
                for item in fallbacks
                if (item[2], item[3]) in stable_pairs
            ]
            fallback_groups = (stable_fallbacks, fallbacks)
        else:
            fallback_groups = (fallbacks,)
        for fallback_group in fallback_groups:
            for _, _, first, second in fallback_group:
                pair = (first, second)
                if pair in selected:
                    continue
                selected.append(pair)
                if len(selected) >= self.min_active_pairs:
                    break
            if len(selected) >= self.min_active_pairs:
                break

        self._active_pairs = selected[: self.active_pair_count]
        if not self._active_pairs:
            raise RuntimeError(
                "Auto-CGRH could not freeze any pair; lower min_pair_score"
            )
        self._pairs_frozen = True
        self._frozen_epoch = self.discovery_epochs

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if (
            int(epoch) > self.discovery_epochs
            and not self._pairs_frozen
        ):
            self._freeze_consensus_pairs()

    def should_observe_teacher(self) -> bool:
        return not self._pairs_frozen and super().should_observe_teacher()

    def checkpoint_eligible(self) -> bool:
        """Discovery-only epochs must not become deployable checkpoints."""
        return bool(self._pairs_frozen)

    def consensus_records(self) -> list[dict[str, Any]]:
        updates = max(int(self.selection_update_count.item()), 1)
        records: list[dict[str, Any]] = []
        for first in range(self.num_classes):
            for second in range(first + 1, self.num_classes):
                votes = int(self.pair_vote_counts[first, second].item())
                if votes <= 0:
                    continue
                records.append(
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "votes": votes,
                        "vote_rate": float(votes / updates),
                        "score": float(
                            self.smoothed_confusion[first, second].item()
                        ),
                    }
                )
        return sorted(
            records,
            key=lambda record: (
                float(record["vote_rate"]),
                float(record["score"]),
            ),
            reverse=True,
        )

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "auto_cgrh",
                "discovery_epochs": int(self.discovery_epochs),
                "discovery_warmup_epochs": int(
                    self.discovery_warmup_epochs
                ),
                "pairs_frozen": bool(self._pairs_frozen),
                "frozen_epoch": self._frozen_epoch,
                "min_consensus_rate": float(self.min_consensus_rate),
                "min_active_pairs": int(self.min_active_pairs),
                "pair_ramp_epochs": int(self.pair_ramp_epochs),
                "stable_pair_gating": bool(self.stable_pair_gating),
                "selection_update_count": int(
                    self.selection_update_count.item()
                ),
                "consensus_candidates": self.consensus_records(),
            }
        )
        return state

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            cgrh_state["main_logits"]
            if cgrh_state is not None
            else refined_logits
        )
        main_classification = self.cross_entropy(main_logits, targets)
        if not self._pairs_frozen:
            ramp_frac = 0.0
        elif self.pair_ramp_epochs <= 0:
            ramp_frac = 1.0
        else:
            epochs_after_freeze = (
                self.current_epoch - (self.discovery_epochs + 1)
            )
            ramp_frac = min(
                1.0,
                max(0.0, epochs_after_freeze / self.pair_ramp_epochs),
            )
        if self._pairs_frozen and ramp_frac > 0.0:
            refined_classification = self.cross_entropy(
                refined_logits,
                targets,
            )
            classification = (
                (1.0 - ramp_frac) * main_classification
                + ramp_frac * refined_classification
            )
        else:
            classification = main_classification
        pair_loss_sum = classification.float().sum() * 0.0
        pair_sample_count = 0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        if (
            self._pairs_frozen
            and cgrh_state is not None
            and self.auxiliary_weight > 0.0
        ):
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            pair_to_slot = {
                tuple(int(value) for value in pair): slot
                for slot, pair in enumerate(pair_indices.tolist())
            }
            for first, second in self._active_pairs:
                slot = pair_to_slot[tuple(sorted((first, second)))]
                mask = (targets == first) | (targets == second)
                count = int(mask.sum().item())
                if count == 0:
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == second).long()
                pair_loss_sum = pair_loss_sum + F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count

        pair_loss = (
            pair_loss_sum / pair_sample_count
            if pair_sample_count > 0
            else pair_loss_sum
        )
        routed = (
            cgrh_state["route_mask"].any(dim=1)
            if cgrh_state is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        effective_pair_weight = self.auxiliary_weight * ramp_frac
        total = classification + effective_pair_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "auto_cgrh_main_classification_loss": float(
                main_classification.detach().item()
            ),
            "auto_cgrh_pair_loss": float(pair_loss.detach().item()),
            "auto_cgrh_active_pair_count": float(len(self._active_pairs)),
            "auto_cgrh_pairs_frozen": float(self._pairs_frozen),
            "auto_cgrh_refine_blend": float(ramp_frac),
            "auto_cgrh_effective_pair_weight": float(
                effective_pair_weight
            ),
            "auto_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "auto_cgrh_route_rate": float(
                routed.float().mean().detach().item()
            ),
        }
        return total


class LMCClassificationLoss(nn.Module):
    """LMC-Net: hard-confusion discovery, stability freeze, then Static CGRH.

    Protocol (single 200-epoch run, no external best.pth):
      1. Warmup: train the main head only; do not mine pairs.
      2. Observe: accumulate EMA-teacher hard confusion C[y, yhat],
         symmetrize + EMA-smooth, and record Top-K undirected pairs.
      3. Freeze at the first selection update on/after ``min_discovery_epochs``
         (or at ``max_discovery_epochs`` as a hard cap).
      4. Refine: activate Top-``refine_pair_count`` pairs (can exceed the
         core Top-K used for logging). Low-margin samples whose
         Top-2 hits any frozen refine pair (e.g. rank-4) still get residual
         correction. Experts stay active at validation/test.

    Optional ablations:
      - ``use_temperature_weights``: softmax scores within frozen refine pool.
      - ``low_margin_only`` (default True): when True, accumulate only on
        low-margin teacher predictions; when False, full hard confusion
        C[y, yhat]. Inference routing still uses low margin separately.
    """

    requires_auto_cgrh = True
    requires_lmc = True
    graph_history_filename = "lmc_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        warmup_epochs: int = 20,
        update_interval: int = 5,
        min_discovery_epochs: int = 40,
        max_discovery_epochs: int = 100,
        active_pair_count: int = 3,
        refine_pair_count: int | None = None,
        min_active_pairs: int = 2,
        confusion_decay: float = 0.90,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        auxiliary_weight: float = 0.20,
        pair_ramp_epochs: int = 20,
        observation_margin_threshold: float = 0.20,
        low_margin_only: bool = True,
        use_temperature_weights: bool = False,
        temperature: float = 0.05,
        never_freeze: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("LMC requires at least two classes")
        if len(class_names) != num_classes:
            raise ValueError("LMC class_names must match num_classes")
        if warmup_epochs < 0 or update_interval < 1:
            raise ValueError("Invalid LMC warmup/update interval")
        never_freeze = bool(never_freeze)
        if never_freeze:
            # Freeze-schedule knobs are unused; keep them ordered for logging.
            if min_discovery_epochs < warmup_epochs:
                min_discovery_epochs = int(warmup_epochs)
            if max_discovery_epochs < min_discovery_epochs:
                max_discovery_epochs = int(min_discovery_epochs)
        else:
            if not 0 <= warmup_epochs < min_discovery_epochs:
                raise ValueError(
                    "LMC warmup_epochs must be in [0, min_discovery_epochs)"
                )
            if min_discovery_epochs > max_discovery_epochs:
                raise ValueError(
                    "LMC min_discovery_epochs must be <= max_discovery_epochs"
                )
        if not 1 <= min_active_pairs <= active_pair_count:
            raise ValueError(
                "LMC requires 1 <= min_active_pairs <= active_pair_count"
            )
        maximum_pairs = int(num_classes) * (int(num_classes) - 1) // 2
        core_pair_count = min(int(active_pair_count), maximum_pairs)
        if refine_pair_count is None:
            refine_pair_count = core_pair_count
        refine_pair_count = min(int(refine_pair_count), maximum_pairs)
        if refine_pair_count < core_pair_count:
            raise ValueError(
                "LMC refine_pair_count must be >= active_pair_count"
            )
        if not 0.0 <= confusion_decay < 1.0:
            raise ValueError("LMC confusion_decay must be in [0, 1)")
        if auxiliary_weight < 0.0 or pair_ramp_epochs < 0:
            raise ValueError("Invalid LMC auxiliary/ramp settings")
        if not 0.0 <= observation_margin_threshold <= 1.0:
            raise ValueError(
                "LMC observation_margin_threshold must be in [0, 1]"
            )
        if temperature <= 0.0:
            raise ValueError("LMC temperature must be > 0")

        self.num_classes = int(num_classes)
        self.class_names = [str(name) for name in class_names]
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.min_discovery_epochs = int(min_discovery_epochs)
        self.max_discovery_epochs = int(max_discovery_epochs)
        self.active_pair_count = core_pair_count
        self.refine_pair_count = refine_pair_count
        self.min_active_pairs = int(min_active_pairs)
        self.confusion_decay = float(confusion_decay)
        self.min_pair_score = float(min_pair_score)
        self.min_class_observations = int(min_class_observations)
        self.auxiliary_weight = float(auxiliary_weight)
        self.pair_ramp_epochs = int(pair_ramp_epochs)
        self.observation_margin_threshold = float(
            observation_margin_threshold
        )
        self.low_margin_only = bool(low_margin_only)
        self.use_temperature_weights = bool(use_temperature_weights)
        self.temperature = float(temperature)
        self.never_freeze = never_freeze

        self.current_epoch = 0
        self.last_components: dict[str, float] = {}
        self._active_pairs: list[tuple[int, int]] = []
        self._core_pairs: list[tuple[int, int]] = []
        self._pair_weights: dict[tuple[int, int], float] = {}
        self._selection_history: list[dict[str, Any]] = []
        self._has_smoothed_confusion = False
        self._observed_samples = 0
        self._pairs_frozen = False
        self._frozen_epoch: int | None = None
        self._freeze_reason: str | None = None
        self._refine_start_epoch: int | None = None
        self._eligible_hard_pair_count = 0
        self._core_hard_pair_count = 0

        shape = (self.num_classes, self.num_classes)
        self.register_buffer("epoch_confusion", torch.zeros(shape))
        self.register_buffer("smoothed_confusion", torch.zeros(shape))

    def should_observe_teacher(self) -> bool:
        if self.current_epoch <= self.warmup_epochs:
            return False
        if self.never_freeze:
            return True
        return (
            not self._pairs_frozen
            and self.current_epoch <= self.max_discovery_epochs
        )

    def checkpoint_eligible(self) -> bool:
        if self.never_freeze:
            return self.current_epoch > self.warmup_epochs
        return bool(self._pairs_frozen)

    def _refinement_ready(self) -> bool:
        if not self._active_pairs:
            return False
        if self.never_freeze:
            return self.current_epoch > self.warmup_epochs
        return bool(self._pairs_frozen)

    def _ramp_fraction(self) -> float:
        if not self._refinement_ready():
            return 0.0
        if self.pair_ramp_epochs <= 0:
            return 1.0
        if self.never_freeze:
            start = (
                int(self._refine_start_epoch)
                if self._refine_start_epoch is not None
                else self.warmup_epochs + 1
            )
            epochs_after = self.current_epoch - start
        else:
            if self._frozen_epoch is None:
                return 1.0
            epochs_after = self.current_epoch - (
                int(self._frozen_epoch) + 1
            )
        return min(1.0, max(0.0, epochs_after / self.pair_ramp_epochs))

    def _publish_active_pairs(
        self,
        scored_pairs: list[tuple[float, int, int]],
        score_matrix: torch.Tensor,
        source_epoch: int,
    ) -> None:
        core_pairs = [
            (first, second)
            for _, first, second in scored_pairs[: self.active_pair_count]
        ]
        refine_pairs = [
            (first, second)
            for _, first, second in scored_pairs[: self.refine_pair_count]
        ]
        if len(refine_pairs) < self.min_active_pairs:
            return
        self._core_pairs = core_pairs
        self._active_pairs = refine_pairs
        self._assign_pair_weights(refine_pairs, score_matrix)
        if self._refine_start_epoch is None:
            self._refine_start_epoch = int(source_epoch) + 1
        # Qualifying hard pairs before Top-K truncation (for model-side stats).
        self._eligible_hard_pair_count = int(len(scored_pairs))
        self._core_hard_pair_count = int(len(core_pairs))

    @torch.no_grad()
    def observe_teacher(
        self,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not self.should_observe_teacher():
            return
        detached_logits = teacher_logits.detach()
        detached_targets = targets.detach().long()
        probabilities = torch.softmax(detached_logits.float(), dim=1)
        top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
        margins = top_probabilities[:, 0] - top_probabilities[:, 1]

        if self.low_margin_only:
            keep = margins <= self.observation_margin_threshold
            if not bool(keep.any()):
                return
            top_indices = top_indices[keep]
            detached_targets = detached_targets[keep]
            # Hesitation mining: credit the unordered Top-2 pair. Using only
            # y→argmax leaves correct-but-uncertain samples on the diagonal,
            # which disappears after symmetrization and yields an empty graph.
            first = top_indices[:, 0]
            second = top_indices[:, 1]
            lo = torch.minimum(first, second)
            hi = torch.maximum(first, second)
            flat_indices = lo * self.num_classes + hi
            counts = torch.bincount(
                flat_indices,
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)
            # Mirror so the matrix stays usable by the symmetric scorer.
            counts = counts + counts.transpose(0, 1)
            counts.fill_diagonal_(0)
            self.epoch_confusion.add_(counts.to(self.epoch_confusion.dtype))
            self._observed_samples += int(detached_targets.numel())
            return

        predictions = detached_logits.argmax(dim=1)
        flat_indices = detached_targets * self.num_classes + predictions
        counts = torch.bincount(
            flat_indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.epoch_confusion.add_(counts.to(self.epoch_confusion.dtype))
        self._observed_samples += int(detached_targets.numel())

    def _score_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
    ) -> list[tuple[float, int, int]]:
        scored_pairs: list[tuple[float, int, int]] = []
        for first in range(self.num_classes):
            if int(class_counts[first].item()) < self.min_class_observations:
                continue
            for second in range(first + 1, self.num_classes):
                if (
                    int(class_counts[second].item())
                    < self.min_class_observations
                ):
                    continue
                score = float(score_matrix[first, second].item())
                if score >= self.min_pair_score:
                    scored_pairs.append((score, first, second))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        return scored_pairs

    def _assign_pair_weights(
        self,
        pairs: list[tuple[int, int]],
        score_matrix: torch.Tensor,
    ) -> None:
        if not pairs:
            self._pair_weights = {}
            return
        if not self.use_temperature_weights:
            weight = 1.0 / float(len(pairs))
            self._pair_weights = {pair: weight for pair in pairs}
            return
        scores = torch.tensor(
            [float(score_matrix[first, second].item()) for first, second in pairs],
            dtype=torch.float32,
        )
        weights = torch.softmax(scores / self.temperature, dim=0)
        self._pair_weights = {
            pair: float(weight.item())
            for pair, weight in zip(pairs, weights)
        }

    def _freeze_pairs(
        self,
        scored_pairs: list[tuple[float, int, int]],
        source_epoch: int,
        reason: str,
        score_matrix: torch.Tensor,
    ) -> None:
        if self.never_freeze:
            raise RuntimeError("LMC never_freeze mode cannot freeze pairs")
        self._publish_active_pairs(scored_pairs, score_matrix, source_epoch)
        if len(self._core_pairs) < self.min_active_pairs:
            raise RuntimeError(
                "LMC could not freeze enough pairs; "
                "lower min_pair_score or extend discovery"
            )
        self._pairs_frozen = True
        self._frozen_epoch = int(source_epoch)
        self._freeze_reason = str(reason)

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        if self._pairs_frozen:
            return
        scored_pairs = self._score_pairs(score_matrix, class_counts)
        top_pairs = [
            (first, second)
            for _, first, second in scored_pairs[: self.active_pair_count]
        ]

        if self.never_freeze:
            self._publish_active_pairs(
                scored_pairs,
                score_matrix,
                source_epoch,
            )

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "never_freeze": bool(self.never_freeze),
                "core_pairs": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "rank": rank,
                    }
                    for rank, (score, first, second) in enumerate(
                        scored_pairs[: self.active_pair_count],
                        start=1,
                    )
                ],
                "refine_candidates": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "rank": rank,
                    }
                    for rank, (score, first, second) in enumerate(
                        scored_pairs[: self.refine_pair_count],
                        start=1,
                    )
                ],
                "pairs_frozen": False,
                "active_pairs": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                    }
                    for first, second in self._active_pairs
                ],
            }
        )

        if self.never_freeze:
            return

        can_freeze = (
            source_epoch >= self.min_discovery_epochs
            and len(top_pairs) >= self.min_active_pairs
        )
        if can_freeze:
            self._freeze_pairs(
                scored_pairs,
                source_epoch,
                reason="min_discovery_epochs",
                score_matrix=score_matrix,
            )
            self._selection_history[-1]["pairs_frozen"] = True
            self._selection_history[-1]["freeze_reason"] = (
                "min_discovery_epochs"
            )

    @torch.no_grad()
    def _finalize_previous_epoch(self, source_epoch: int) -> None:
        if self._pairs_frozen:
            self.epoch_confusion.zero_()
            self._observed_samples = 0
            return

        if self._observed_samples <= 0:
            if (
                not self.never_freeze
                and source_epoch >= self.max_discovery_epochs
                and self._has_smoothed_confusion
            ):
                class_counts = self.smoothed_confusion.new_full(
                    (self.num_classes,),
                    float(self.min_class_observations),
                )
                scored = self._score_pairs(
                    self.smoothed_confusion,
                    class_counts,
                )
                if scored:
                    self._freeze_pairs(
                        scored,
                        source_epoch,
                        reason="max_discovery_epochs",
                        score_matrix=self.smoothed_confusion,
                    )
            self.epoch_confusion.zero_()
            self._observed_samples = 0
            return

        class_counts = self.epoch_confusion.sum(dim=1)
        normalized = self.epoch_confusion / class_counts.clamp_min(
            1.0
        ).unsqueeze(1)
        symmetric_scores = 0.5 * (
            normalized + normalized.transpose(0, 1)
        )
        symmetric_scores.fill_diagonal_(0.0)

        if self._has_smoothed_confusion:
            self.smoothed_confusion.mul_(self.confusion_decay).add_(
                symmetric_scores,
                alpha=1.0 - self.confusion_decay,
            )
        else:
            self.smoothed_confusion.copy_(symmetric_scores)
            self._has_smoothed_confusion = True

        should_update = (
            source_epoch >= self.warmup_epochs
            and (source_epoch - self.warmup_epochs) % self.update_interval
            == 0
        )
        if should_update:
            self._select_active_pairs(
                self.smoothed_confusion,
                class_counts,
                source_epoch,
            )

        if (
            not self.never_freeze
            and not self._pairs_frozen
            and source_epoch >= self.max_discovery_epochs
        ):
            scored = self._score_pairs(
                self.smoothed_confusion,
                class_counts,
            )
            if scored:
                self._freeze_pairs(
                    scored,
                    source_epoch,
                    reason="max_discovery_epochs",
                    score_matrix=self.smoothed_confusion,
                )

        self.epoch_confusion.zero_()
        self._observed_samples = 0

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if self.current_epoch > 0 and epoch != self.current_epoch:
            self._finalize_previous_epoch(self.current_epoch)
        self.current_epoch = epoch

    def active_pair_records(
        self,
        score_matrix: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        scores = (
            self.smoothed_confusion if score_matrix is None else score_matrix
        )
        core_set = set(self._core_pairs)
        return [
            {
                "class_i": self.class_names[first],
                "class_j": self.class_names[second],
                "class_i_index": int(first),
                "class_j_index": int(second),
                "score": float(scores[first, second].item()),
                "weight": float(
                    self._pair_weights.get((first, second), 0.0)
                ),
                "rank": rank,
                "is_core": (first, second) in core_set,
            }
            for rank, (first, second) in enumerate(
                self._active_pairs,
                start=1,
            )
        ]

    def graph_state(self) -> dict[str, Any]:
        return {
            "variant": "lmc",
            "current_epoch": int(self.current_epoch),
            "warmup_epochs": int(self.warmup_epochs),
            "min_discovery_epochs": int(self.min_discovery_epochs),
            "max_discovery_epochs": int(self.max_discovery_epochs),
            "active_pair_count": int(self.active_pair_count),
            "refine_pair_count": int(self.refine_pair_count),
            "never_freeze": bool(self.never_freeze),
            "pairs_frozen": bool(self._pairs_frozen),
            "frozen_epoch": self._frozen_epoch,
            "freeze_reason": self._freeze_reason,
            "refine_start_epoch": self._refine_start_epoch,
            "low_margin_only": bool(self.low_margin_only),
            "use_temperature_weights": bool(self.use_temperature_weights),
            "temperature": float(self.temperature),
            "core_pairs": [
                {
                    "class_i": self.class_names[first],
                    "class_j": self.class_names[second],
                    "class_i_index": int(first),
                    "class_j_index": int(second),
                }
                for first, second in self._core_pairs
            ],
            "active_pairs": self.active_pair_records(),
            "selection_history": list(self._selection_history),
        }

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            cgrh_state["main_logits"]
            if cgrh_state is not None
            else refined_logits
        )
        main_classification = self.cross_entropy(main_logits, targets)
        ramp_frac = self._ramp_fraction()
        refine_ready = self._refinement_ready()

        if refine_ready and ramp_frac > 0.0:
            refined_classification = self.cross_entropy(
                refined_logits,
                targets,
            )
            classification = (
                (1.0 - ramp_frac) * main_classification
                + ramp_frac * refined_classification
            )
        else:
            classification = main_classification

        pair_loss_sum = classification.float().sum() * 0.0
        pair_sample_count = 0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        if (
            refine_ready
            and cgrh_state is not None
            and self.auxiliary_weight > 0.0
            and ramp_frac > 0.0
        ):
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            pair_to_slot = {
                tuple(int(value) for value in pair): slot
                for slot, pair in enumerate(pair_indices.tolist())
            }
            for first, second in self._active_pairs:
                slot = pair_to_slot[tuple(sorted((first, second)))]
                mask = (targets == first) | (targets == second)
                count = int(mask.sum().item())
                if count == 0:
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == second).long()
                pair_weight = float(
                    self._pair_weights.get((first, second), 1.0)
                )
                pair_loss_sum = pair_loss_sum + pair_weight * F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                    reduction="sum",
                )
                pair_sample_count += count

        pair_loss = (
            pair_loss_sum / pair_sample_count
            if pair_sample_count > 0
            else pair_loss_sum
        )
        routed = (
            cgrh_state["route_mask"].any(dim=1)
            if cgrh_state is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        effective_pair_weight = self.auxiliary_weight * ramp_frac
        total = classification + effective_pair_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "lmc_main_classification_loss": float(
                main_classification.detach().item()
            ),
            "lmc_pair_loss": float(pair_loss.detach().item()),
            "lmc_active_pair_count": float(len(self._active_pairs)),
            "lmc_core_hard_pair_count": float(self._core_hard_pair_count),
            "lmc_eligible_hard_pair_count": float(
                self._eligible_hard_pair_count
            ),
            "lmc_pairs_frozen": float(self._pairs_frozen),
            "lmc_never_freeze": float(self.never_freeze),
            "lmc_refine_blend": float(ramp_frac),
            "lmc_effective_pair_weight": float(effective_pair_weight),
            "lmc_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "lmc_route_rate": float(
                routed.float().mean().detach().item()
            ),
        }
        return total


class SoftDynamicCGRHClassificationLoss(DynamicConfusionMarginGuidanceLoss):
    """Bounded all-pair supervision driven by an EMA soft-confusion graph.

    The graph never freezes and no hard Top-K switch is used. Pair weights sum
    to one, can increase or decrease at epoch boundaries, and are multiplied by
    a capped global strength that decays as total EMA confusion is resolved.
    Validation and deployment use the main classifier only.
    """

    requires_soft_dynamic_cgrh = True
    requires_dcmg = False
    graph_history_filename = "soft_dynamic_cgrh_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        warmup_epochs: int = 20,
        ramp_end_epoch: int = 60,
        update_interval: int = 1,
        confusion_decay: float = 0.90,
        temperature: float = 0.02,
        max_pair_share: float = 0.45,
        max_pair_weight: float = 0.10,
        max_refine_weight: float = 0.05,
        min_difficulty_scale: float = 0.20,
        min_class_observations: int = 8,
    ) -> None:
        pair_count = int(num_classes) * (int(num_classes) - 1) // 2
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=0.0,
            margin=0.0,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=max(pair_count, 1),
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=0,
            min_pair_score=0.0,
            min_class_observations=min_class_observations,
            entry_persistence=1,
            exit_rank_multiplier=1,
            score_weighted=False,
        )
        if ramp_end_epoch <= warmup_epochs:
            raise ValueError(
                "Soft Dynamic CGRH ramp_end_epoch must exceed warmup_epochs"
            )
        if temperature <= 0.0:
            raise ValueError("Soft Dynamic CGRH temperature must be > 0")
        if not 0.0 < max_pair_share <= 1.0:
            raise ValueError("max_pair_share must be in (0, 1]")
        if max_pair_share * pair_count < 1.0:
            raise ValueError("max_pair_share is too small for the pair count")
        if max_pair_weight < 0.0 or max_refine_weight < 0.0:
            raise ValueError("Soft Dynamic CGRH weights must be >= 0")
        if not 0.0 <= min_difficulty_scale <= 1.0:
            raise ValueError("min_difficulty_scale must be in [0, 1]")

        self.ramp_end_epoch = int(ramp_end_epoch)
        self.temperature = float(temperature)
        self.max_pair_share = float(max_pair_share)
        self.max_pair_weight = float(max_pair_weight)
        self.max_refine_weight = float(max_refine_weight)
        self.min_difficulty_scale = float(min_difficulty_scale)
        self.register_buffer(
            "dynamic_pair_weights",
            torch.zeros(pair_count),
        )
        self.register_buffer("reference_difficulty", torch.zeros(()))
        self.register_buffer("current_difficulty", torch.zeros(()))

    def should_observe_teacher(self) -> bool:
        return True

    def _capped_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Project softmax mass onto a simplex with a per-pair upper bound."""
        raw = torch.softmax(logits, dim=0)
        output = torch.zeros_like(raw)
        remaining = torch.ones_like(raw, dtype=torch.bool)
        remaining_mass = 1.0
        for _ in range(int(raw.numel())):
            if not bool(remaining.any()):
                break
            candidate = (
                raw[remaining]
                / raw[remaining].sum().clamp_min(1e-12)
                * remaining_mass
            )
            over = candidate > self.max_pair_share
            if not bool(over.any()):
                output[remaining] = candidate
                break
            remaining_indices = remaining.nonzero(as_tuple=False).flatten()
            capped_indices = remaining_indices[over]
            output[capped_indices] = self.max_pair_share
            remaining[capped_indices] = False
            remaining_mass = max(
                0.0,
                remaining_mass
                - self.max_pair_share * int(capped_indices.numel()),
            )
        return output / output.sum().clamp_min(1e-12)

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        pair_scores: list[torch.Tensor] = []
        pair_records: list[tuple[int, int]] = []
        valid_pairs: list[bool] = []
        for first in range(self.num_classes):
            for second in range(first + 1, self.num_classes):
                pair_records.append((first, second))
                pair_scores.append(score_matrix[first, second])
                valid_pairs.append(
                    int(class_counts[first].item())
                    >= self.min_class_observations
                    and int(class_counts[second].item())
                    >= self.min_class_observations
                )
        scores = torch.stack(pair_scores)
        valid = torch.tensor(
            valid_pairs,
            dtype=torch.bool,
            device=scores.device,
        )
        if not bool(valid.any()):
            valid = torch.ones_like(valid)
        difficulty = scores[valid].sum()
        self.current_difficulty.copy_(difficulty)
        if float(self.reference_difficulty.item()) <= 0.0:
            self.reference_difficulty.copy_(difficulty.clamp_min(1e-8))

        logits = scores / self.temperature
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = self._capped_softmax(logits)
        self.dynamic_pair_weights.copy_(weights)

        ranked = sorted(
            (
                (
                    float(weights[slot].item()),
                    float(scores[slot].item()),
                    first,
                    second,
                )
                for slot, (first, second) in enumerate(pair_records)
            ),
            reverse=True,
        )
        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "difficulty": float(difficulty.item()),
                "effective_pair_weight": float(
                    self.effective_pair_weight()
                ),
                "top_pairs": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "weight": float(weight),
                        "score": float(score),
                    }
                    for weight, score, first, second in ranked[:5]
                ],
            }
        )

    def pair_weight_vector(self) -> torch.Tensor:
        return self.dynamic_pair_weights

    def difficulty_scale(self) -> float:
        reference = float(self.reference_difficulty.item())
        if reference <= 0.0:
            return 0.0
        ratio = float(self.current_difficulty.item()) / reference
        return min(1.0, max(self.min_difficulty_scale, ratio))

    def ramp_scale(self) -> float:
        if self.current_epoch <= self.warmup_epochs:
            return 0.0
        progress = (
            (self.current_epoch - self.warmup_epochs)
            / max(self.ramp_end_epoch - self.warmup_epochs, 1)
        )
        return min(1.0, max(0.0, progress))

    def effective_pair_weight(self) -> float:
        return (
            self.max_pair_weight
            * self.ramp_scale()
            * self.difficulty_scale()
        )

    def effective_refine_weight(self) -> float:
        return (
            self.max_refine_weight
            * self.ramp_scale()
            * self.difficulty_scale()
        )

    def pair_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        slot = 0
        for first in range(self.num_classes):
            for second in range(first + 1, self.num_classes):
                records.append(
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "weight": float(
                            self.dynamic_pair_weights[slot].item()
                        ),
                        "score": float(
                            self.smoothed_confusion[first, second].item()
                        ),
                    }
                )
                slot += 1
        return sorted(
            records,
            key=lambda record: float(record["weight"]),
            reverse=True,
        )

    def graph_state(self) -> dict[str, Any]:
        return {
            "current_epoch": int(self.current_epoch),
            "variant": "soft_dynamic_cgrh",
            "warmup_epochs": int(self.warmup_epochs),
            "ramp_end_epoch": int(self.ramp_end_epoch),
            "temperature": float(self.temperature),
            "max_pair_share": float(self.max_pair_share),
            "max_pair_weight": float(self.max_pair_weight),
            "max_refine_weight": float(self.max_refine_weight),
            "difficulty": float(self.current_difficulty.item()),
            "reference_difficulty": float(
                self.reference_difficulty.item()
            ),
            "difficulty_scale": float(self.difficulty_scale()),
            "effective_pair_weight": float(self.effective_pair_weight()),
            "effective_refine_weight": float(
                self.effective_refine_weight()
            ),
            "pairs": self.pair_records(),
            "selection_history": list(self._selection_history),
        }

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if cgrh_state is None:
            main_logits = refined_logits
            pair_logits = None
            pair_indices = None
            route_mask = torch.zeros_like(targets, dtype=torch.bool)
        else:
            main_logits = cgrh_state["main_logits"]
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            route_mask = cgrh_state["route_mask"].any(dim=1)

        classification = self.cross_entropy(main_logits, targets)
        pair_loss = classification.float().sum() * 0.0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)
        if (
            pair_logits is not None
            and pair_indices is not None
            and float(self.dynamic_pair_weights.sum().item()) > 0.0
        ):
            weighted_losses: list[torch.Tensor] = []
            for slot, (first, second) in enumerate(pair_indices.tolist()):
                mask = (targets == int(first)) | (targets == int(second))
                if not bool(mask.any()):
                    continue
                weight = self.dynamic_pair_weights[slot].to(
                    device=pair_logits.device,
                    dtype=pair_logits.dtype,
                )
                if float(weight.item()) <= 0.0:
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == int(second)).long()
                binary_loss = F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                )
                weighted_losses.append(weight.float() * binary_loss)
            if weighted_losses:
                pair_loss = torch.stack(weighted_losses).sum()

        if bool(route_mask.any()):
            refine_loss = F.cross_entropy(
                refined_logits[route_mask].float(),
                targets[route_mask],
            )
        else:
            refine_loss = classification.float().sum() * 0.0

        pair_strength = self.effective_pair_weight()
        refine_strength = self.effective_refine_weight()
        total = (
            classification
            + pair_strength * pair_loss
            + refine_strength * refine_loss
        )
        weights = self.dynamic_pair_weights.float()
        entropy = float(
            (-(weights * weights.clamp_min(1e-12).log()).sum())
            .detach()
            .item()
        )
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "soft_cgrh_pair_loss": float(pair_loss.detach().item()),
            "soft_cgrh_refine_loss": float(refine_loss.detach().item()),
            "soft_cgrh_effective_pair_weight": float(pair_strength),
            "soft_cgrh_effective_refine_weight": float(refine_strength),
            "soft_cgrh_difficulty_scale": float(self.difficulty_scale()),
            "soft_cgrh_pair_entropy": entropy,
            "soft_cgrh_max_pair_share": float(
                weights.max().detach().item()
                if weights.numel()
                else 0.0
            ),
            "soft_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "soft_cgrh_route_rate": float(
                route_mask.float().mean().detach().item()
            ),
        }
        return total


class ContinuousTop3CGRHClassificationLoss(DynamicConfusionMarginGuidanceLoss):
    """Continuous EMA Top-3 CGRH with progressive auxiliary strength.

    Unlike Auto-CGRH, the active Top-3 graph is never permanently frozen.
    Pair supervision ramps from zero after warmup, reaches ``auxiliary_weight``
    (default 0.20, matching fixed CGRH), and only the current Top-3 pairs
    receive non-zero mass.  Rank-3/4 swaps use two-update persistence on the
    score Top-3 and replace when the challenger score is strictly higher;
    weights then cross-fade over ``pair_handoff_epochs``.
    """

    requires_continuous_top3_cgrh = True
    requires_dcmg = False
    graph_history_filename = "continuous_top3_cgrh_history.json"
    _CONTINUOUS_FREEZE_EPOCH = 999_999

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        warmup_epochs: int = 20,
        auxiliary_weight: float = 0.20,
        auxiliary_ramp_epochs: int = 40,
        update_interval: int = 5,
        active_pair_count: int = 3,
        min_active_pairs: int = 2,
        confusion_decay: float = 0.90,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 1,
        exit_persistence: int = 2,
        replace_margin: float = 0.0,
        pair_handoff_epochs: int = 5,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("Continuous Top-3 CGRH warmup_epochs must be >= 0")
        if auxiliary_weight < 0.0:
            raise ValueError(
                "Continuous Top-3 CGRH auxiliary_weight must be >= 0"
            )
        if auxiliary_ramp_epochs < 0:
            raise ValueError(
                "Continuous Top-3 CGRH auxiliary_ramp_epochs must be >= 0"
            )
        if not 1 <= min_active_pairs <= active_pair_count:
            raise ValueError(
                "Continuous Top-3 CGRH requires "
                "1 <= min_active_pairs <= active_pair_count"
            )
        if exit_persistence < 1:
            raise ValueError(
                "Continuous Top-3 CGRH exit_persistence must be >= 1"
            )
        if replace_margin < 0.0:
            raise ValueError(
                "Continuous Top-3 CGRH replace_margin must be >= 0"
            )
        if pair_handoff_epochs < 1:
            raise ValueError(
                "Continuous Top-3 CGRH pair_handoff_epochs must be >= 1"
            )

        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=0.0,
            margin=0.0,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=self._CONTINUOUS_FREEZE_EPOCH,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
            score_weighted=False,
        )
        self.auxiliary_weight = float(auxiliary_weight)
        self.auxiliary_ramp_epochs = int(auxiliary_ramp_epochs)
        self.min_active_pairs = int(min_active_pairs)
        self.exit_persistence = int(exit_persistence)
        self.replace_margin = float(replace_margin)
        self.pair_handoff_epochs = int(pair_handoff_epochs)
        self._supervision_ready_epoch: int | None = None
        self._handoff_progress = 0

        pair_records: list[tuple[int, int]] = []
        for first in range(self.num_classes):
            for second in range(first + 1, self.num_classes):
                pair_records.append((first, second))
        self._pair_records = pair_records
        self._pair_to_slot = {
            pair: slot for slot, pair in enumerate(pair_records)
        }

        pair_count = len(pair_records)
        self.register_buffer(
            "supervision_weights",
            torch.zeros(pair_count),
        )
        self.register_buffer(
            "target_supervision_weights",
            torch.zeros(pair_count),
        )
        self.register_buffer(
            "exit_streaks",
            torch.zeros(
                (self.num_classes, self.num_classes),
                dtype=torch.long,
            ),
        )

    def should_observe_teacher(self) -> bool:
        return True

    @torch.no_grad()
    def _set_target_supervision_weights(self) -> None:
        target = torch.zeros_like(self.target_supervision_weights)
        if len(self._active_pairs) >= self.min_active_pairs:
            share = 1.0 / float(len(self._active_pairs))
            for first, second in self._active_pairs:
                slot = self._pair_to_slot[(first, second)]
                target[slot] = share
        if torch.allclose(
            target,
            self.target_supervision_weights,
            atol=1e-8,
            rtol=0.0,
        ):
            return
        self.target_supervision_weights.copy_(target)
        self._handoff_progress = 0

    @torch.no_grad()
    def _advance_supervision_blend(self) -> None:
        if self._handoff_progress >= self.pair_handoff_epochs:
            self.supervision_weights.copy_(self.target_supervision_weights)
            return
        alpha = 1.0 / float(self.pair_handoff_epochs)
        self.supervision_weights.lerp_(
            self.target_supervision_weights,
            alpha,
        )
        self._handoff_progress += 1

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        scored_pairs: list[tuple[float, int, int]] = []
        for first in range(self.num_classes):
            if int(class_counts[first].item()) < self.min_class_observations:
                continue
            for second in range(first + 1, self.num_classes):
                if int(class_counts[second].item()) < self.min_class_observations:
                    continue
                score = float(score_matrix[first, second].item())
                if score >= self.min_pair_score:
                    scored_pairs.append((score, first, second))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        entry_pairs = {
            (first, second)
            for _, first, second in scored_pairs[: self.active_pair_count]
        }
        exit_limit = self.active_pair_count * self.exit_rank_multiplier
        exit_pairs = {
            (first, second)
            for _, first, second in scored_pairs[:exit_limit]
        }

        new_streaks = torch.zeros_like(self.candidate_streaks)
        for first, second in entry_pairs:
            streak = int(self.candidate_streaks[first, second].item()) + 1
            new_streaks[first, second] = streak
            new_streaks[second, first] = streak
        self.candidate_streaks.copy_(new_streaks)

        new_exit_streaks = torch.zeros_like(self.exit_streaks)
        retained: list[tuple[int, int]] = []
        for first, second in self._active_pairs:
            pair = (first, second)
            if pair in exit_pairs:
                retained.append(pair)
                continue
            streak = int(self.exit_streaks[first, second].item()) + 1
            new_exit_streaks[first, second] = streak
            new_exit_streaks[second, first] = streak
            if streak < self.exit_persistence:
                retained.append(pair)
        self.exit_streaks.copy_(new_exit_streaks)

        for score, first, second in scored_pairs:
            pair = (first, second)
            if pair in retained:
                continue
            if pair not in entry_pairs:
                continue
            if int(self.candidate_streaks[first, second].item()) < (
                self.entry_persistence
            ):
                continue
            if len(retained) >= self.active_pair_count:
                lowest = min(
                    retained,
                    key=lambda item: float(
                        score_matrix[item[0], item[1]].item()
                    ),
                )
                lowest_score = float(
                    score_matrix[lowest[0], lowest[1]].item()
                )
                if float(score) < lowest_score + self.replace_margin:
                    continue
                retained.remove(lowest)
            retained.append(pair)
            if len(retained) >= self.active_pair_count:
                break

        self._active_pairs = sorted(
            retained,
            key=lambda pair: float(score_matrix[pair[0], pair[1]].item()),
            reverse=True,
        )[: self.active_pair_count]
        self._set_target_supervision_weights()

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "active_pairs": self.active_pair_records(score_matrix),
                "supervision_weights": self.pair_records(),
                "top_candidates": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "entry_streak": int(
                            self.candidate_streaks[first, second].item()
                        ),
                        "exit_streak": int(
                            self.exit_streaks[first, second].item()
                        ),
                    }
                    for score, first, second in scored_pairs[
                        : self.active_pair_count + 1
                    ]
                ],
            }
        )

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if (
            self._supervision_ready_epoch is None
            and self.current_epoch > self.warmup_epochs
            and len(self._active_pairs) >= self.min_active_pairs
        ):
            self._supervision_ready_epoch = int(self.current_epoch)
        self._advance_supervision_blend()

    def effective_auxiliary_weight(self) -> float:
        if self.current_epoch <= self.warmup_epochs:
            return 0.0
        if len(self._active_pairs) < self.min_active_pairs:
            return 0.0
        if float(self.supervision_weights.sum().item()) <= 0.0:
            return 0.0
        if self.auxiliary_ramp_epochs <= 0:
            return self.auxiliary_weight
        ready_epoch = self._supervision_ready_epoch
        if ready_epoch is None or self.current_epoch <= ready_epoch:
            return 0.0
        progress = min(
            1.0,
            (self.current_epoch - ready_epoch)
            / float(max(self.auxiliary_ramp_epochs, 1)),
        )
        return self.auxiliary_weight * progress

    def pair_weight_vector(self) -> torch.Tensor:
        return self.supervision_weights.clone()

    def pair_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for slot, (first, second) in enumerate(self._pair_records):
            records.append(
                {
                    "class_i": self.class_names[first],
                    "class_j": self.class_names[second],
                    "class_i_index": int(first),
                    "class_j_index": int(second),
                    "weight": float(self.supervision_weights[slot].item()),
                    "target_weight": float(
                        self.target_supervision_weights[slot].item()
                    ),
                    "score": float(
                        self.smoothed_confusion[first, second].item()
                    ),
                }
            )
        return sorted(
            records,
            key=lambda record: float(record["weight"]),
            reverse=True,
        )

    def graph_state(self) -> dict[str, Any]:
        return {
            "current_epoch": int(self.current_epoch),
            "variant": "continuous_top3_cgrh",
            "warmup_epochs": int(self.warmup_epochs),
            "auxiliary_weight": float(self.auxiliary_weight),
            "auxiliary_ramp_epochs": int(self.auxiliary_ramp_epochs),
            "effective_auxiliary_weight": float(
                self.effective_auxiliary_weight()
            ),
            "min_active_pairs": int(self.min_active_pairs),
            "entry_persistence": int(self.entry_persistence),
            "exit_persistence": int(self.exit_persistence),
            "exit_rank_multiplier": int(self.exit_rank_multiplier),
            "replace_margin": float(self.replace_margin),
            "pair_handoff_epochs": int(self.pair_handoff_epochs),
            "handoff_progress": int(self._handoff_progress),
            "supervision_ready_epoch": self._supervision_ready_epoch,
            "active_pairs": self.active_pair_records(),
            "pairs": self.pair_records(),
            "selection_history": list(self._selection_history),
        }

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            cgrh_state["main_logits"]
            if cgrh_state is not None
            else refined_logits
        )
        classification = self.cross_entropy(main_logits, targets)
        pair_loss = classification.float().sum() * 0.0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        weights = self.supervision_weights.to(
            device=main_logits.device,
            dtype=main_logits.dtype,
        )
        weight_sum = float(weights.sum().item())
        if (
            cgrh_state is not None
            and weight_sum > 0.0
            and self.effective_auxiliary_weight() > 0.0
        ):
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            weighted_losses: list[torch.Tensor] = []
            for slot, (first, second) in enumerate(pair_indices.tolist()):
                weight = weights[slot]
                if float(weight.item()) <= 0.0:
                    continue
                mask = (targets == int(first)) | (targets == int(second))
                if not bool(mask.any()):
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == int(second)).long()
                binary_loss = F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                )
                weighted_losses.append(weight.float() * binary_loss)
            if weighted_losses:
                pair_loss = torch.stack(weighted_losses).sum()

        auxiliary_strength = self.effective_auxiliary_weight()
        total = classification + auxiliary_strength * pair_loss
        route_mask = (
            cgrh_state["route_mask"].any(dim=1)
            if cgrh_state is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        top_records = self.pair_records()
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "ct3_cgrh_pair_loss": float(pair_loss.detach().item()),
            "ct3_cgrh_effective_auxiliary_weight": float(
                auxiliary_strength
            ),
            "ct3_cgrh_active_pair_count": float(len(self._active_pairs)),
            "ct3_cgrh_supervision_weight_sum": float(weight_sum),
            "ct3_cgrh_handoff_progress": float(self._handoff_progress),
            "ct3_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_route_rate": float(
                route_mask.float().mean().detach().item()
            ),
            "ct3_cgrh_top_pair_weight": float(
                top_records[0]["weight"] if top_records else 0.0
            ),
        }
        return total


class CALiteDualCEClassificationLoss(ContinuousTop3CGRHClassificationLoss):
    """Scheme A: EMA Top-k discovery + full CE + hard-pair CE on main head.

    Keeps confusion-graph discovery and equal-mass Top-k schedule from
    Continuous Top-3, but applies hard-pair supervision on the main-head
    2-class logits. No pair experts, residual scales, or refined path.
    Inference uses the single multiclass head only.
    """

    requires_continuous_top3_cgrh = False
    requires_competence_adaptive_top3_cgrh = False
    requires_dcmg = True
    graph_history_filename = "ca_lite_dual_ce_history.json"

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state["variant"] = "ca_lite_dual_ce"
        state["inference"] = "single_main_head"
        state["hard_pair_supervision"] = "main_logit_binary_ce"
        return state

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del cgrh_state
        main_logits = logits
        classification = self.cross_entropy(main_logits, targets)
        pair_loss = classification.float().sum() * 0.0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        weights = self.supervision_weights.to(
            device=main_logits.device,
            dtype=main_logits.dtype,
        )
        weight_sum = float(weights.sum().item())
        if weight_sum > 0.0 and self.effective_auxiliary_weight() > 0.0:
            weighted_losses: list[torch.Tensor] = []
            for slot, (first, second) in enumerate(self._pair_records):
                weight = weights[slot]
                if float(weight.item()) <= 0.0:
                    continue
                mask = (targets == int(first)) | (targets == int(second))
                if not bool(mask.any()):
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == int(second)).long()
                binary_logits = main_logits[mask][
                    :, [int(first), int(second)]
                ].float()
                binary_loss = F.cross_entropy(
                    binary_logits,
                    binary_targets,
                )
                weighted_losses.append(weight.float() * binary_loss)
            if weighted_losses:
                pair_loss = torch.stack(weighted_losses).sum()

        auxiliary_strength = self.effective_auxiliary_weight()
        total = classification + auxiliary_strength * pair_loss
        top_records = self.pair_records()
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "ca_lite_full_ce": float(classification.detach().item()),
            "ca_lite_hard_pair_ce": float(pair_loss.detach().item()),
            "ct3_cgrh_pair_loss": float(pair_loss.detach().item()),
            "ct3_cgrh_effective_auxiliary_weight": float(
                auxiliary_strength
            ),
            "ct3_cgrh_active_pair_count": float(len(self._active_pairs)),
            "ct3_cgrh_supervision_weight_sum": float(weight_sum),
            "ct3_cgrh_handoff_progress": float(self._handoff_progress),
            "ct3_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_route_rate": 0.0,
            "ct3_cgrh_top_pair_weight": float(
                top_records[0]["weight"] if top_records else 0.0
            ),
        }
        return total


class CompetenceAdaptiveTop3CGRHClassificationLoss(
    ContinuousTop3CGRHClassificationLoss
):
    """Continuous Top-3 discovery with competence-controlled residual CE.

    Pair-supervision weights retain the Swap recipe and sum to one.  A separate
    per-pair residual scale starts at zero and follows detached evidence that
    the binary expert beats the main head on samples that actually satisfy the
    CGRH route.  Inactive-pair evidence is discounted rather than discarded.
    """

    requires_continuous_top3_cgrh = False
    requires_competence_adaptive_top3_cgrh = True
    graph_history_filename = "competence_adaptive_top3_cgrh_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        warmup_epochs: int = 20,
        auxiliary_weight: float = 0.20,
        auxiliary_ramp_epochs: int = 40,
        update_interval: int = 5,
        active_pair_count: int = 3,
        min_active_pairs: int = 2,
        confusion_decay: float = 0.90,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 1,
        exit_persistence: int = 2,
        replace_margin: float = 0.0,
        pair_handoff_epochs: int = 5,
        max_residual_scale: float = 0.50,
        main_ce_weight: float = 0.25,
        refined_ce_weight: float = 0.75,
        advantage_decay: float = 0.95,
        evidence_decay: float = 0.995,
        inactive_evidence_decay: float = 0.90,
        minimum_effective_samples: float = 64.0,
        advantage_margin: float = 0.02,
        advantage_temperature: float = 0.02,
        scale_update_rate: float = 0.01,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs,
            auxiliary_weight=auxiliary_weight,
            auxiliary_ramp_epochs=auxiliary_ramp_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            min_active_pairs=min_active_pairs,
            confusion_decay=confusion_decay,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
            exit_persistence=exit_persistence,
            replace_margin=replace_margin,
            pair_handoff_epochs=pair_handoff_epochs,
        )
        if max_residual_scale < 0.0:
            raise ValueError("max_residual_scale must be >= 0")
        if main_ce_weight < 0.0 or refined_ce_weight < 0.0:
            raise ValueError("classification CE weights must be >= 0")
        if main_ce_weight + refined_ce_weight <= 0.0:
            raise ValueError("classification CE weights must have positive sum")
        for name, value in (
            ("advantage_decay", advantage_decay),
            ("evidence_decay", evidence_decay),
            ("inactive_evidence_decay", inactive_evidence_decay),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if minimum_effective_samples <= 0.0:
            raise ValueError("minimum_effective_samples must be > 0")
        if advantage_temperature <= 0.0:
            raise ValueError("advantage_temperature must be > 0")
        if not 0.0 < scale_update_rate <= 1.0:
            raise ValueError("scale_update_rate must be in (0, 1]")

        classification_weight_sum = main_ce_weight + refined_ce_weight
        self.max_residual_scale = float(max_residual_scale)
        self.main_ce_weight = float(main_ce_weight) / classification_weight_sum
        self.refined_ce_weight = (
            float(refined_ce_weight) / classification_weight_sum
        )
        self.advantage_decay = float(advantage_decay)
        self.evidence_decay = float(evidence_decay)
        self.inactive_evidence_decay = float(inactive_evidence_decay)
        self.minimum_effective_samples = float(minimum_effective_samples)
        self.advantage_margin = float(advantage_margin)
        self.advantage_temperature = float(advantage_temperature)
        self.scale_update_rate = float(scale_update_rate)

        pair_count = len(self._pair_records)
        self.register_buffer("residual_scales", torch.zeros(pair_count))
        self.register_buffer("expert_advantage_ema", torch.zeros(pair_count))
        self.register_buffer("effective_sample_counts", torch.zeros(pair_count))
        self.register_buffer("target_residual_scales", torch.zeros(pair_count))

    def residual_scale_vector(self) -> torch.Tensor:
        return self.residual_scales.clone()

    def set_epoch(self, epoch: int) -> None:
        previous_epoch = int(self.current_epoch)
        super().set_epoch(epoch)
        if previous_epoch <= 0 or int(epoch) == previous_epoch:
            return
        active_slots = {
            self._pair_to_slot[pair] for pair in self._active_pairs
        }
        with torch.no_grad():
            for slot in range(len(self._pair_records)):
                if slot in active_slots:
                    continue
                self.expert_advantage_ema[slot].mul_(
                    self.inactive_evidence_decay
                )
                self.effective_sample_counts[slot].mul_(
                    self.inactive_evidence_decay
                )

    @torch.no_grad()
    def _observe_competence(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        pair_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> None:
        for slot, (first, second) in enumerate(pair_indices.tolist()):
            if float(self.supervision_weights[slot].item()) <= 0.0:
                continue
            mask = route_mask[:, slot] & (
                (targets == int(first)) | (targets == int(second))
            )
            count = int(mask.sum().item())
            if count <= 0:
                continue
            binary_targets = (targets[mask] == int(second)).long()
            main_pair_logits = main_logits[mask][:, [int(first), int(second)]]
            main_loss = F.cross_entropy(
                main_pair_logits.float(),
                binary_targets,
            )
            expert_loss = F.cross_entropy(
                pair_logits[mask, slot].float(),
                binary_targets,
            )
            advantage = main_loss - expert_loss
            self.expert_advantage_ema[slot].mul_(
                self.advantage_decay
            ).add_(
                advantage,
                alpha=1.0 - self.advantage_decay,
            )
            self.effective_sample_counts[slot].mul_(
                self.evidence_decay
            ).add_(float(count))

        active_mass = self.supervision_weights * float(
            max(len(self._active_pairs), 1)
        )
        active_mass.clamp_(0.0, 1.0)
        confidence = self.effective_sample_counts / (
            self.effective_sample_counts + self.minimum_effective_samples
        )
        readiness = torch.sigmoid(
            (
                self.expert_advantage_ema - self.advantage_margin
            )
            / self.advantage_temperature
        )
        target = (
            self.max_residual_scale
            * readiness
            * confidence
            * active_mass
        )
        self.target_residual_scales.copy_(target)
        self.residual_scales.lerp_(target, self.scale_update_rate)

    def competence_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for slot, (first, second) in enumerate(self._pair_records):
            records.append(
                {
                    "class_i": self.class_names[first],
                    "class_j": self.class_names[second],
                    "class_i_index": int(first),
                    "class_j_index": int(second),
                    "active": bool(
                        float(self.supervision_weights[slot].item()) > 0.0
                    ),
                    "residual_scale": float(
                        self.residual_scales[slot].item()
                    ),
                    "target_residual_scale": float(
                        self.target_residual_scales[slot].item()
                    ),
                    "expert_advantage_ema": float(
                        self.expert_advantage_ema[slot].item()
                    ),
                    "effective_samples": float(
                        self.effective_sample_counts[slot].item()
                    ),
                }
            )
        return sorted(
            records,
            key=lambda record: float(record["residual_scale"]),
            reverse=True,
        )

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "competence_adaptive_continuous_top3_cgrh",
                "max_residual_scale": self.max_residual_scale,
                "main_ce_weight": self.main_ce_weight,
                "refined_ce_weight": self.refined_ce_weight,
                "advantage_decay": self.advantage_decay,
                "evidence_decay": self.evidence_decay,
                "inactive_evidence_decay": self.inactive_evidence_decay,
                "minimum_effective_samples": self.minimum_effective_samples,
                "advantage_margin": self.advantage_margin,
                "advantage_temperature": self.advantage_temperature,
                "scale_update_rate": self.scale_update_rate,
                "competence": self.competence_records(),
            }
        )
        return state

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            cgrh_state["main_logits"]
            if cgrh_state is not None
            else refined_logits
        )
        main_classification = self.cross_entropy(main_logits, targets)
        refined_classification = self.cross_entropy(refined_logits, targets)
        classification = (
            self.main_ce_weight * main_classification
            + self.refined_ce_weight * refined_classification
        )
        pair_loss = classification.float().sum() * 0.0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        weights = self.supervision_weights.to(
            device=main_logits.device,
            dtype=main_logits.dtype,
        )
        weight_sum = float(weights.sum().item())
        if cgrh_state is not None:
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            route_mask = cgrh_state["route_mask"]
            self._observe_competence(
                main_logits.detach(),
                targets.detach(),
                pair_logits.detach(),
                pair_indices,
                route_mask,
            )
        else:
            pair_logits = None
            pair_indices = None
            route_mask = None

        if (
            pair_logits is not None
            and pair_indices is not None
            and weight_sum > 0.0
            and self.effective_auxiliary_weight() > 0.0
        ):
            weighted_losses: list[torch.Tensor] = []
            for slot, (first, second) in enumerate(pair_indices.tolist()):
                weight = weights[slot]
                if float(weight.item()) <= 0.0:
                    continue
                mask = (targets == int(first)) | (targets == int(second))
                if not bool(mask.any()):
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == int(second)).long()
                binary_loss = F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                )
                weighted_losses.append(weight.float() * binary_loss)
            if weighted_losses:
                pair_loss = torch.stack(weighted_losses).sum()

        auxiliary_strength = self.effective_auxiliary_weight()
        total = classification + auxiliary_strength * pair_loss
        routed_samples = (
            route_mask.any(dim=1)
            if route_mask is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        active_scale_mask = self.supervision_weights > 0.0
        active_scales = self.residual_scales[active_scale_mask]
        mean_scale = (
            float(active_scales.mean().item())
            if active_scales.numel() > 0
            else 0.0
        )
        max_scale = (
            float(active_scales.max().item())
            if active_scales.numel() > 0
            else 0.0
        )
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "ca_cgrh_main_ce": float(main_classification.detach().item()),
            "ca_cgrh_refined_ce": float(refined_classification.detach().item()),
            "ct3_cgrh_pair_loss": float(pair_loss.detach().item()),
            "ct3_cgrh_effective_auxiliary_weight": float(
                auxiliary_strength
            ),
            "ct3_cgrh_active_pair_count": float(len(self._active_pairs)),
            "ct3_cgrh_supervision_weight_sum": float(weight_sum),
            "ct3_cgrh_handoff_progress": float(self._handoff_progress),
            "ct3_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_route_rate": float(
                routed_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_top_pair_weight": float(
                max(
                    (
                        float(weight)
                        for weight in self.supervision_weights.tolist()
                    ),
                    default=0.0,
                )
            ),
            "ca_cgrh_mean_scale": mean_scale,
            "ca_cgrh_max_scale": max_scale,
            "ca_cgrh_mean_advantage": float(
                self.expert_advantage_ema[active_scale_mask].mean().item()
            )
            if bool(active_scale_mask.any())
            else 0.0,
            "ca_cgrh_effective_samples": float(
                self.effective_sample_counts[active_scale_mask].sum().item()
            ),
        }
        return total


class StableCompetenceAdaptiveTop3CGRHClassificationLoss(
    CompetenceAdaptiveTop3CGRHClassificationLoss
):
    """Competence-adaptive CGRH with continuously voted Top-3 discovery.

    EMA-teacher confusion remains the only discovery signal.  A pair must
    appear repeatedly in the recent Top-3 window before it can enter the
    active graph, while an incumbent receives one-vote exit hysteresis.
    The graph is never frozen: qualified challengers can replace weaker
    incumbents throughout training after clearing a relative score margin.

    Pair-loss coverage/weights and competence-controlled residual scales are
    inherited unchanged.  In particular, competence never affects pair
    ranking or pair-supervision mass.
    """

    graph_history_filename = "stable_ca_cgrh_history.json"

    def __init__(
        self,
        *args: Any,
        vote_window: int = 4,
        minimum_votes: int = 3,
        replacement_margin_ratio: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if vote_window < 1:
            raise ValueError("vote_window must be >= 1")
        if not 1 <= minimum_votes <= vote_window:
            raise ValueError(
                "minimum_votes must satisfy 1 <= minimum_votes <= vote_window"
            )
        if replacement_margin_ratio < 0.0:
            raise ValueError("replacement_margin_ratio must be >= 0")
        self.vote_window = int(vote_window)
        self.minimum_votes = int(minimum_votes)
        self.replacement_margin_ratio = float(replacement_margin_ratio)
        self._candidate_vote_history: list[list[tuple[int, int]]] = []

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        scored_pairs: list[tuple[float, int, int]] = []
        for first in range(self.num_classes):
            if int(class_counts[first].item()) < self.min_class_observations:
                continue
            for second in range(first + 1, self.num_classes):
                if int(class_counts[second].item()) < self.min_class_observations:
                    continue
                score = float(score_matrix[first, second].item())
                if score >= self.min_pair_score:
                    scored_pairs.append((score, first, second))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        current_candidates = [
            (first, second)
            for _, first, second in scored_pairs[: self.active_pair_count]
        ]
        self._candidate_vote_history.append(current_candidates)
        if len(self._candidate_vote_history) > self.vote_window:
            self._candidate_vote_history.pop(0)

        vote_counts = {pair: 0 for pair in self._pair_records}
        for candidate_set in self._candidate_vote_history:
            for pair in candidate_set:
                vote_counts[pair] += 1
        score_lookup = {
            (first, second): score for score, first, second in scored_pairs
        }

        # New pairs require the full vote threshold.  Existing pairs receive
        # one-vote exit hysteresis so a single noisy update cannot evict them.
        incumbent_threshold = max(1, self.minimum_votes - 1)
        retained = [
            pair
            for pair in self._active_pairs
            if vote_counts.get(pair, 0) >= incumbent_threshold
            and pair in score_lookup
        ]
        retained.sort(
            key=lambda pair: score_lookup.get(pair, 0.0),
            reverse=True,
        )
        retained = retained[: self.active_pair_count]

        qualified = [
            pair
            for pair, votes in vote_counts.items()
            if votes >= self.minimum_votes and pair in score_lookup
        ]
        qualified.sort(
            key=lambda pair: (
                vote_counts[pair],
                score_lookup[pair],
            ),
            reverse=True,
        )

        for challenger in qualified:
            if challenger in retained:
                continue
            if len(retained) < self.active_pair_count:
                retained.append(challenger)
                continue
            weakest = min(retained, key=lambda pair: score_lookup[pair])
            weakest_score = score_lookup[weakest]
            challenger_score = score_lookup[challenger]
            clears_margin = (
                weakest_score <= 0.0
                or challenger_score
                >= weakest_score * (1.0 + self.replacement_margin_ratio)
            )
            if clears_margin:
                retained.remove(weakest)
                retained.append(challenger)

        self._active_pairs = sorted(
            retained,
            key=lambda pair: score_lookup.get(pair, 0.0),
            reverse=True,
        )[: self.active_pair_count]
        self._set_target_supervision_weights()

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "variant": "stable_competence_adaptive_top3_cgrh",
                "pair_locking": False,
                "vote_window": self.vote_window,
                "minimum_votes": self.minimum_votes,
                "replacement_margin_ratio": self.replacement_margin_ratio,
                "active_pairs": self.active_pair_records(score_matrix),
                "supervision_weights": self.pair_records(),
                "candidate_window": [
                    [
                        {
                            "class_i": self.class_names[first],
                            "class_j": self.class_names[second],
                            "class_i_index": int(first),
                            "class_j_index": int(second),
                        }
                        for first, second in candidate_set
                    ]
                    for candidate_set in self._candidate_vote_history
                ],
                "top_candidates": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "votes": int(vote_counts[(first, second)]),
                        "qualified": bool(
                            vote_counts[(first, second)] >= self.minimum_votes
                        ),
                    }
                    for score, first, second in scored_pairs[
                        : self.active_pair_count + 3
                    ]
                ],
            }
        )

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "stable_competence_adaptive_top3_cgrh",
                "pair_locking": False,
                "vote_window": self.vote_window,
                "minimum_votes": self.minimum_votes,
                "replacement_margin_ratio": self.replacement_margin_ratio,
            }
        )
        return state


class CompetenceAdaptiveTop3CGRHCDWClassificationLoss(
    CompetenceAdaptiveTop3CGRHClassificationLoss
):
    """CA-Top3-CGRH with online class-difficulty weighted main/refined CE.

    Per-class training CE EMA drives mild sample weights with no class-name
    priors. Pair discovery, competence residuals, and pair loss are unchanged.
    Statistics update only on training steps from detached main-head CE.
    """

    graph_history_filename = (
        "competence_adaptive_top3_cgrh_cdw_history.json"
    )

    def __init__(
        self,
        *args: Any,
        difficulty_ema_beta: float = 0.90,
        difficulty_warmup_epochs: int = 5,
        difficulty_min_weight: float = 0.80,
        difficulty_max_weight: float = 1.20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= difficulty_ema_beta < 1.0:
            raise ValueError("difficulty_ema_beta must be in [0, 1)")
        if difficulty_warmup_epochs < 0:
            raise ValueError("difficulty_warmup_epochs must be >= 0")
        if not (
            0.0
            < difficulty_min_weight
            <= 1.0
            <= difficulty_max_weight
        ):
            raise ValueError(
                "difficulty weights must satisfy 0 < min <= 1 <= max"
            )
        self.difficulty_ema_beta = float(difficulty_ema_beta)
        self.difficulty_warmup_epochs = int(difficulty_warmup_epochs)
        self.difficulty_min_weight = float(difficulty_min_weight)
        self.difficulty_max_weight = float(difficulty_max_weight)
        self.label_smoothing = float(
            getattr(self.cross_entropy, "label_smoothing", 0.0)
        )
        self.register_buffer(
            "class_loss_ema", torch.ones(self.num_classes)
        )
        self.register_buffer(
            "class_seen",
            torch.zeros(self.num_classes, dtype=torch.bool),
        )

    @torch.no_grad()
    def _update_class_difficulty(
        self,
        per_sample_loss: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        for class_index in targets.unique():
            index = int(class_index.item())
            class_mean = per_sample_loss[targets == index].mean().float()
            if not bool(self.class_seen[index]):
                self.class_loss_ema[index].copy_(class_mean)
                self.class_seen[index] = True
            else:
                self.class_loss_ema[index].mul_(
                    self.difficulty_ema_beta
                ).add_(
                    class_mean,
                    alpha=1.0 - self.difficulty_ema_beta,
                )

    @torch.no_grad()
    def current_class_weights(self) -> torch.Tensor:
        weights = torch.ones_like(self.class_loss_ema)
        if not torch.any(self.class_seen):
            return weights
        reference = (
            self.class_loss_ema[self.class_seen].mean().clamp_min(1e-8)
        )
        weights[self.class_seen] = (
            self.class_loss_ema[self.class_seen] / reference
        )
        weights.clamp_(
            self.difficulty_min_weight,
            self.difficulty_max_weight,
        )
        weights.div_(weights.mean().clamp_min(1e-8))
        return weights

    def _weighted_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(
            1e-8
        )

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        weights = self.current_class_weights()
        state.update(
            {
                "variant": "competence_adaptive_top3_cgrh_cdw",
                "difficulty_ema_beta": self.difficulty_ema_beta,
                "difficulty_warmup_epochs": self.difficulty_warmup_epochs,
                "difficulty_min_weight": self.difficulty_min_weight,
                "difficulty_max_weight": self.difficulty_max_weight,
                "class_difficulty": [
                    {
                        "class_name": self.class_names[index],
                        "class_index": int(index),
                        "loss_ema": float(self.class_loss_ema[index].item()),
                        "seen": bool(self.class_seen[index].item()),
                        "weight": float(weights[index].item()),
                    }
                    for index in range(self.num_classes)
                ],
            }
        )
        return state

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            cgrh_state["main_logits"]
            if cgrh_state is not None
            else refined_logits
        )
        main_per_sample = F.cross_entropy(
            main_logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        training_step = torch.is_grad_enabled()
        if training_step:
            self._update_class_difficulty(
                main_per_sample.detach(),
                targets.detach(),
            )
        use_weights = (
            training_step
            and self.current_epoch > self.difficulty_warmup_epochs
        )
        class_weights = (
            self.current_class_weights()
            if use_weights
            else torch.ones_like(self.class_loss_ema)
        ).to(device=main_logits.device, dtype=main_per_sample.dtype)
        sample_weights = class_weights[targets]
        main_classification = (
            main_per_sample * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-8)
        refined_classification = self._weighted_ce(
            refined_logits,
            targets,
            sample_weights,
        )
        plain_main = main_per_sample.mean()
        plain_refined = F.cross_entropy(
            refined_logits,
            targets,
            label_smoothing=self.label_smoothing,
        )
        classification = (
            self.main_ce_weight * main_classification
            + self.refined_ce_weight * refined_classification
        )
        pair_loss = classification.float().sum() * 0.0
        supervised_samples = torch.zeros_like(targets, dtype=torch.bool)

        weights = self.supervision_weights.to(
            device=main_logits.device,
            dtype=main_logits.dtype,
        )
        weight_sum = float(weights.sum().item())
        if cgrh_state is not None:
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            route_mask = cgrh_state["route_mask"]
            self._observe_competence(
                main_logits.detach(),
                targets.detach(),
                pair_logits.detach(),
                pair_indices,
                route_mask,
            )
        else:
            pair_logits = None
            pair_indices = None
            route_mask = None

        if (
            pair_logits is not None
            and pair_indices is not None
            and weight_sum > 0.0
            and self.effective_auxiliary_weight() > 0.0
        ):
            weighted_losses: list[torch.Tensor] = []
            for slot, (first, second) in enumerate(pair_indices.tolist()):
                weight = weights[slot]
                if float(weight.item()) <= 0.0:
                    continue
                mask = (targets == int(first)) | (targets == int(second))
                if not bool(mask.any()):
                    continue
                supervised_samples |= mask
                binary_targets = (targets[mask] == int(second)).long()
                binary_loss = F.cross_entropy(
                    pair_logits[mask, slot].float(),
                    binary_targets,
                )
                weighted_losses.append(weight.float() * binary_loss)
            if weighted_losses:
                pair_loss = torch.stack(weighted_losses).sum()

        auxiliary_strength = self.effective_auxiliary_weight()
        total = classification + auxiliary_strength * pair_loss
        routed_samples = (
            route_mask.any(dim=1)
            if route_mask is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        active_scale_mask = self.supervision_weights > 0.0
        active_scales = self.residual_scales[active_scale_mask]
        mean_scale = (
            float(active_scales.mean().item())
            if active_scales.numel() > 0
            else 0.0
        )
        max_scale = (
            float(active_scales.max().item())
            if active_scales.numel() > 0
            else 0.0
        )
        self.last_components = {
            "classification_loss": float(
                (
                    self.main_ce_weight * plain_main
                    + self.refined_ce_weight * plain_refined
                )
                .detach()
                .item()
            ),
            "ca_cgrh_main_ce": float(plain_main.detach().item()),
            "ca_cgrh_refined_ce": float(plain_refined.detach().item()),
            "cdw_weighted_main_ce": float(
                main_classification.detach().item()
            ),
            "cdw_weighted_refined_ce": float(
                refined_classification.detach().item()
            ),
            "cdw_weight_min": float(class_weights.min().item()),
            "cdw_weight_max": float(class_weights.max().item()),
            "ct3_cgrh_pair_loss": float(pair_loss.detach().item()),
            "ct3_cgrh_effective_auxiliary_weight": float(
                auxiliary_strength
            ),
            "ct3_cgrh_active_pair_count": float(len(self._active_pairs)),
            "ct3_cgrh_supervision_weight_sum": float(weight_sum),
            "ct3_cgrh_handoff_progress": float(self._handoff_progress),
            "ct3_cgrh_supervision_rate": float(
                supervised_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_route_rate": float(
                routed_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_top_pair_weight": float(
                max(
                    (
                        float(weight)
                        for weight in self.supervision_weights.tolist()
                    ),
                    default=0.0,
                )
            ),
            "ca_cgrh_mean_scale": mean_scale,
            "ca_cgrh_max_scale": max_scale,
            "ca_cgrh_mean_advantage": float(
                self.expert_advantage_ema[active_scale_mask].mean().item()
            )
            if bool(active_scale_mask.any())
            else 0.0,
            "ca_cgrh_effective_samples": float(
                self.effective_sample_counts[active_scale_mask].sum().item()
            ),
        }
        return total


class CompetenceAdaptiveTop3CGRHV2AClassificationLoss(
    CompetenceAdaptiveTop3CGRHClassificationLoss
):
    """V2-A: hardness- and competence-gated pair supervision.

    The original main/refined CE recipe is retained so this ablation isolates
    four auxiliary-path changes: continuous hard-sample weighting, competence
    gating of Pair Loss, hard-zero residuals for non-positive expert advantage,
    and per-sample normalisation when active pairs overlap.
    """

    graph_history_filename = (
        "competence_adaptive_top3_cgrh_v2a_history.json"
    )

    def __init__(
        self,
        *args: Any,
        hard_sample_quantile: float = 0.30,
        easy_sample_weight_floor: float = 0.10,
        competence_weight_floor: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < hard_sample_quantile < 1.0:
            raise ValueError("hard_sample_quantile must be in (0, 1)")
        for name, value in (
            ("easy_sample_weight_floor", easy_sample_weight_floor),
            ("competence_weight_floor", competence_weight_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.hard_sample_quantile = float(hard_sample_quantile)
        self.easy_sample_weight_floor = float(easy_sample_weight_floor)
        self.competence_weight_floor = float(competence_weight_floor)

    def _competence_gate_vector(self) -> torch.Tensor:
        active_mass = self.supervision_weights * float(
            max(len(self._active_pairs), 1)
        )
        active_mass = active_mass.clamp(0.0, 1.0)
        confidence = self.effective_sample_counts / (
            self.effective_sample_counts + self.minimum_effective_samples
        )
        readiness = torch.sigmoid(
            (self.expert_advantage_ema - self.advantage_margin)
            / self.advantage_temperature
        )
        positive_advantage = (self.expert_advantage_ema > 0.0).to(
            readiness.dtype
        )
        return readiness * confidence * active_mass * positive_advantage

    @torch.no_grad()
    def _observe_competence(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        pair_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> None:
        # Keep the original route-specific evidence pool. Enlarging that pool
        # would be a separate ablation rather than part of V2-A.
        for slot, (first, second) in enumerate(pair_indices.tolist()):
            if float(self.supervision_weights[slot].item()) <= 0.0:
                continue
            mask = route_mask[:, slot] & (
                (targets == int(first)) | (targets == int(second))
            )
            count = int(mask.sum().item())
            if count <= 0:
                continue
            binary_targets = (targets[mask] == int(second)).long()
            main_pair_logits = main_logits[mask][:, [int(first), int(second)]]
            main_loss = F.cross_entropy(
                main_pair_logits.float(),
                binary_targets,
            )
            expert_loss = F.cross_entropy(
                pair_logits[mask, slot].float(),
                binary_targets,
            )
            advantage = main_loss - expert_loss
            self.expert_advantage_ema[slot].mul_(
                self.advantage_decay
            ).add_(
                advantage,
                alpha=1.0 - self.advantage_decay,
            )
            self.effective_sample_counts[slot].mul_(
                self.evidence_decay
            ).add_(float(count))

        gate = self._competence_gate_vector()
        target = self.max_residual_scale * gate
        self.target_residual_scales.copy_(target)
        self.residual_scales.lerp_(target, self.scale_update_rate)

    def _pair_hardness_weights(
        self,
        main_pair_logits: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(
            main_pair_logits.detach().float(),
            dim=1,
        )
        probability_margin = (
            probabilities[:, 0] - probabilities[:, 1]
        ).abs()
        if probability_margin.numel() == 0:
            return probability_margin
        threshold = torch.quantile(
            probability_margin,
            self.hard_sample_quantile,
        )
        # Adaptive transition width avoids another tuned hyperparameter.
        temperature = max(0.05, 0.25 * float(threshold.item()))
        hard_weight = torch.sigmoid(
            (threshold - probability_margin) / temperature
        )
        return self.easy_sample_weight_floor + (
            1.0 - self.easy_sample_weight_floor
        ) * hard_weight

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "competence_adaptive_top3_cgrh_v2a",
                "hard_sample_quantile": self.hard_sample_quantile,
                "easy_sample_weight_floor": (
                    self.easy_sample_weight_floor
                ),
                "competence_weight_floor": (
                    self.competence_weight_floor
                ),
                "negative_advantage_residual": "hard_zero",
                "pair_weight_normalization": "per_sample_max_one",
            }
        )
        return state

    def forward(
        self,
        refined_logits: torch.Tensor,
        targets: torch.Tensor,
        cgrh_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        main_logits = (
            cgrh_state["main_logits"]
            if cgrh_state is not None
            else refined_logits
        )
        main_classification = self.cross_entropy(main_logits, targets)
        refined_classification = self.cross_entropy(refined_logits, targets)
        classification = (
            self.main_ce_weight * main_classification
            + self.refined_ce_weight * refined_classification
        )
        pair_loss = classification.float().sum() * 0.0
        candidate_samples = torch.zeros_like(targets, dtype=torch.bool)
        effective_samples = torch.zeros_like(targets, dtype=torch.bool)
        sample_pair_weight = torch.zeros(
            targets.shape[0],
            device=targets.device,
            dtype=torch.float32,
        )

        weights = self.supervision_weights.to(
            device=main_logits.device,
            dtype=torch.float32,
        )
        weight_sum = float(weights.sum().item())
        if cgrh_state is not None:
            pair_logits = cgrh_state["pair_logits"]
            pair_indices = cgrh_state["pair_indices"]
            route_mask = cgrh_state["route_mask"]
            self._observe_competence(
                main_logits.detach(),
                targets.detach(),
                pair_logits.detach(),
                pair_indices,
                route_mask,
            )
        else:
            pair_logits = None
            pair_indices = None
            route_mask = None

        auxiliary_strength = self.effective_auxiliary_weight()
        pair_specs: list[tuple[int, int, int, torch.Tensor]] = []
        raw_pair_weights: list[torch.Tensor] = []
        if (
            pair_logits is not None
            and pair_indices is not None
            and weight_sum > 0.0
            and auxiliary_strength > 0.0
        ):
            competence_gate = self._competence_gate_vector().to(
                device=main_logits.device,
                dtype=torch.float32,
            )
            active_pair_scale = float(max(len(self._active_pairs), 1))
            active_mass = (weights * active_pair_scale).clamp(0.0, 1.0)
            for slot, (first, second) in enumerate(pair_indices.tolist()):
                if float(active_mass[slot].item()) <= 0.0:
                    continue
                mask = (targets == int(first)) | (targets == int(second))
                if not bool(mask.any()):
                    continue
                candidate_samples |= mask
                hardness = self._pair_hardness_weights(
                    main_logits[mask][:, [int(first), int(second)]]
                )
                competence_weight = self.competence_weight_floor + (
                    1.0 - self.competence_weight_floor
                ) * competence_gate[slot]
                raw_weight = torch.zeros_like(sample_pair_weight)
                raw_weight[mask] = (
                    active_mass[slot]
                    * competence_weight
                    * hardness
                )
                pair_specs.append((slot, int(first), int(second), mask))
                raw_pair_weights.append(raw_weight)

            if raw_pair_weights:
                raw_weight_matrix = torch.stack(raw_pair_weights, dim=1)
                overlap_normalizer = raw_weight_matrix.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
                normalized_weight_matrix = (
                    raw_weight_matrix / overlap_normalizer
                )
                sample_pair_weight = normalized_weight_matrix.sum(dim=1)
                effective_samples = sample_pair_weight >= 0.50
                weighted_loss_sum = pair_loss
                for column, (slot, _, second, mask) in enumerate(pair_specs):
                    binary_targets = (targets[mask] == second).long()
                    binary_loss = F.cross_entropy(
                        pair_logits[mask, slot].float(),
                        binary_targets,
                        reduction="none",
                    )
                    weighted_loss_sum = weighted_loss_sum + (
                        normalized_weight_matrix[mask, column]
                        * binary_loss
                    ).sum()
                # Dividing by candidate count (not gated weight sum) preserves
                # the intended attenuation from hardness and competence.
                pair_loss = weighted_loss_sum / candidate_samples.sum().clamp_min(
                    1
                ).float()

        total = classification + auxiliary_strength * pair_loss
        routed_samples = (
            route_mask.any(dim=1)
            if route_mask is not None
            else torch.zeros_like(targets, dtype=torch.bool)
        )
        active_scale_mask = self.supervision_weights > 0.0
        active_scales = self.residual_scales[active_scale_mask]
        active_advantages = self.expert_advantage_ema[active_scale_mask]
        active_gates = self._competence_gate_vector()[active_scale_mask]
        candidate_count = int(candidate_samples.sum().item())
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "ca_cgrh_main_ce": float(main_classification.detach().item()),
            "ca_cgrh_refined_ce": float(
                refined_classification.detach().item()
            ),
            "ct3_cgrh_pair_loss": float(pair_loss.detach().item()),
            "ct3_cgrh_effective_auxiliary_weight": float(
                auxiliary_strength
            ),
            "ct3_cgrh_active_pair_count": float(len(self._active_pairs)),
            "ct3_cgrh_supervision_weight_sum": float(weight_sum),
            "ct3_cgrh_handoff_progress": float(self._handoff_progress),
            "ct3_cgrh_supervision_rate": float(
                candidate_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_route_rate": float(
                routed_samples.float().mean().detach().item()
            ),
            "ct3_cgrh_top_pair_weight": float(
                max(
                    (
                        float(weight)
                        for weight in self.supervision_weights.tolist()
                    ),
                    default=0.0,
                )
            ),
            "ca_cgrh_mean_scale": float(active_scales.mean().item())
            if active_scales.numel() > 0
            else 0.0,
            "ca_cgrh_max_scale": float(active_scales.max().item())
            if active_scales.numel() > 0
            else 0.0,
            "ca_cgrh_mean_advantage": float(active_advantages.mean().item())
            if active_advantages.numel() > 0
            else 0.0,
            "ca_cgrh_effective_samples": float(
                self.effective_sample_counts[active_scale_mask].sum().item()
            ),
            "v2a_candidate_supervision_rate": float(
                candidate_samples.float().mean().detach().item()
            ),
            "v2a_effective_supervision_rate": float(
                effective_samples.float().mean().detach().item()
            ),
            "v2a_mean_candidate_weight": float(
                sample_pair_weight[candidate_samples].mean().item()
            )
            if candidate_count > 0
            else 0.0,
            "v2a_competence_gate_mean": float(active_gates.mean().item())
            if active_gates.numel() > 0
            else 0.0,
            "v2a_positive_competence_pairs": float(
                (active_advantages > 0.0).sum().item()
            ),
        }
        return total


class ReliableDynamicConfusionMarginGuidanceLoss(
    DynamicConfusionMarginGuidanceLoss
):
    """Reliability-filtered DCMG with progressive main-logit guidance.

    DCMG-V2 keeps the training-only EMA soft-confusion graph from DCMG-V1,
    but no longer applies the pair loss to every sample whose label belongs
    to an active pair.  A sample is guided only when all of the following are
    true:

    * the EMA teacher predicts the ground-truth class as Top-1;
    * the graph-selected competitor is exactly the teacher's Top-2 class;
    * teacher confidence stays inside a reliable, non-trivial interval; and
    * the deployable main head is still close to the selected pair boundary.

    The auxiliary weight is linearly ramped and each sample loss is capped so
    that rare/noisy examples cannot dominate the main classification target.
    There is no extra inference branch or inference-time computation.
    """

    requires_dcmg = False
    requires_dcmg_v2 = True
    graph_history_filename = "dcmg_v2_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        max_pair_weight: float = 0.02,
        margin: float = 0.10,
        warmup_epochs: int = 20,
        update_interval: int = 10,
        active_pair_count: int = 3,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 150,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 3,
        exit_rank_multiplier: int = 2,
        ramp_start_epoch: int = 30,
        ramp_end_epoch: int = 80,
        min_teacher_true_prob: float = 0.45,
        max_teacher_true_prob: float = 0.95,
        min_teacher_competitor_prob: float = 0.05,
        max_teacher_margin: float = 0.70,
        max_main_logit_gap: float = 2.0,
        loss_cap: float = 1.5,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=max_pair_weight,
            margin=margin,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=freeze_pairs_epoch,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
            score_weighted=False,
        )
        if ramp_start_epoch < 0 or ramp_end_epoch < ramp_start_epoch:
            raise ValueError("Invalid DCMG-V2 ramp epoch range")
        if not 0.0 <= min_teacher_true_prob <= 1.0:
            raise ValueError("DCMG-V2 min_teacher_true_prob must be in [0, 1]")
        if not 0.0 <= max_teacher_true_prob <= 1.0:
            raise ValueError("DCMG-V2 max_teacher_true_prob must be in [0, 1]")
        if min_teacher_true_prob > max_teacher_true_prob:
            raise ValueError("DCMG-V2 teacher probability range is empty")
        if not 0.0 <= min_teacher_competitor_prob <= 1.0:
            raise ValueError(
                "DCMG-V2 min_teacher_competitor_prob must be in [0, 1]"
            )
        if max_teacher_margin < 0.0 or max_main_logit_gap < 0.0:
            raise ValueError("DCMG-V2 margin/gap thresholds must be >= 0")
        if loss_cap <= 0.0:
            raise ValueError("DCMG-V2 loss_cap must be > 0")

        self.max_pair_weight = float(max_pair_weight)
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.min_teacher_true_prob = float(min_teacher_true_prob)
        self.max_teacher_true_prob = float(max_teacher_true_prob)
        self.min_teacher_competitor_prob = float(
            min_teacher_competitor_prob
        )
        self.max_teacher_margin = float(max_teacher_margin)
        self.max_main_logit_gap = float(max_main_logit_gap)
        self.loss_cap = float(loss_cap)

    def effective_pair_weight(self) -> float:
        """Return the linearly ramped guidance weight for this epoch."""
        if self.current_epoch < self.ramp_start_epoch:
            return 0.0
        if self.ramp_end_epoch <= self.ramp_start_epoch:
            return self.max_pair_weight
        progress = (
            (self.current_epoch - self.ramp_start_epoch)
            / (self.ramp_end_epoch - self.ramp_start_epoch)
        )
        return self.max_pair_weight * min(max(progress, 0.0), 1.0)

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "dcmg_v2_reliable_progressive",
                "max_pair_weight": float(self.max_pair_weight),
                "effective_pair_weight": float(
                    self.effective_pair_weight()
                ),
                "ramp_start_epoch": int(self.ramp_start_epoch),
                "ramp_end_epoch": int(self.ramp_end_epoch),
                "min_teacher_true_prob": float(
                    self.min_teacher_true_prob
                ),
                "max_teacher_true_prob": float(
                    self.max_teacher_true_prob
                ),
                "min_teacher_competitor_prob": float(
                    self.min_teacher_competitor_prob
                ),
                "max_teacher_margin": float(self.max_teacher_margin),
                "max_main_logit_gap": float(self.max_main_logit_gap),
                "loss_cap": float(self.loss_cap),
            }
        )
        return state

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        margin_loss = classification.float().sum() * 0.0
        candidate_rate = 0.0
        teacher_reliable_rate = 0.0
        main_close_rate = 0.0
        supervision_rate = 0.0
        violation_rate = 0.0
        mean_logit_gap = 0.0
        effective_weight = self.effective_pair_weight()

        guidance_enabled = (
            teacher_logits is not None
            and self.current_epoch > self.warmup_epochs
            and effective_weight > 0.0
            and bool(self._active_pairs)
        )
        if guidance_enabled:
            competitors, _ = self._class_competitors(main_logits.device)
            sample_competitors = competitors[targets]
            candidate_mask = sample_competitors >= 0
            candidate_rate = float(
                candidate_mask.float().mean().detach().item()
            )

            safe_competitors = sample_competitors.clamp_min(0)
            row_indices = torch.arange(
                targets.shape[0], device=targets.device
            )
            true_logits = main_logits[row_indices, targets].float()
            hard_logits = main_logits[
                row_indices, safe_competitors
            ].float()
            logit_gaps = true_logits - hard_logits

            with torch.no_grad():
                teacher_probabilities = torch.softmax(
                    teacher_logits.detach().float(), dim=1
                )
                top2_indices = teacher_probabilities.topk(
                    k=2, dim=1
                ).indices
                teacher_true_probabilities = teacher_probabilities[
                    row_indices, targets
                ]
                teacher_competitor_probabilities = teacher_probabilities[
                    row_indices, safe_competitors
                ]
                teacher_margins = (
                    teacher_true_probabilities
                    - teacher_competitor_probabilities
                )

                teacher_reliable_mask = (
                    candidate_mask
                    & (top2_indices[:, 0] == targets)
                    & (top2_indices[:, 1] == sample_competitors)
                    & (
                        teacher_true_probabilities
                        >= self.min_teacher_true_prob
                    )
                    & (
                        teacher_true_probabilities
                        <= self.max_teacher_true_prob
                    )
                    & (
                        teacher_competitor_probabilities
                        >= self.min_teacher_competitor_prob
                    )
                    & (teacher_margins <= self.max_teacher_margin)
                )
                main_close_mask = (
                    candidate_mask
                    & (
                        logit_gaps.detach().abs()
                        <= self.max_main_logit_gap
                    )
                )
                supervised_mask = teacher_reliable_mask & main_close_mask

            teacher_reliable_rate = float(
                teacher_reliable_mask.float().mean().detach().item()
            )
            main_close_rate = float(
                main_close_mask.float().mean().detach().item()
            )
            supervision_rate = float(
                supervised_mask.float().mean().detach().item()
            )
            if bool(supervised_mask.any()):
                supervised_gaps = logit_gaps[supervised_mask]
                per_sample_loss = F.softplus(
                    self.margin - supervised_gaps
                ).clamp(max=self.loss_cap)
                margin_loss = per_sample_loss.mean()
                violation_rate = float(
                    (supervised_gaps < self.margin)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                mean_logit_gap = float(
                    supervised_gaps.mean().detach().item()
                )

        total = classification + effective_weight * margin_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "dcmg_v2_margin_loss": float(margin_loss.detach().item()),
            "dcmg_v2_active_pair_count": float(len(self._active_pairs)),
            "dcmg_v2_candidate_rate": candidate_rate,
            "dcmg_v2_teacher_reliable_rate": teacher_reliable_rate,
            "dcmg_v2_main_close_rate": main_close_rate,
            "dcmg_v2_supervision_rate": supervision_rate,
            "dcmg_v2_violation_rate": violation_rate,
            "dcmg_v2_mean_logit_gap": mean_logit_gap,
            "dcmg_v2_effective_pair_weight": float(effective_weight),
        }
        return total


class DynamicConfusionGraphPairRoutingLoss(
    DynamicConfusionMarginGuidanceLoss
):
    """Dynamic graph discovery plus CGRH-style main-logit pair routing.

    DCMG-CGR-V3 uses the EMA teacher only to estimate a training-set soft
    confusion graph.  Two stable pair slots preserve persistent relations and
    one exploration slot follows the strongest remaining relation.  Unlike
    DCMG-V2, per-sample routing is decided by the deployable main classifier:
    an active pair is supervised only when it exactly matches the main Top-2
    classes and their probability gap is small.  Pair cross entropy is applied
    directly to the corresponding two main logits, so no expert head survives
    (or even exists) at inference time.
    """

    requires_dcmg = True
    graph_history_filename = "dcmg_cgr_v3_pair_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        max_pair_weight: float = 0.10,
        warmup_epochs: int = 20,
        update_interval: int = 5,
        active_pair_count: int = 3,
        stable_pair_count: int = 2,
        candidate_pool_size: int = 6,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 150,
        min_auto_freeze_epoch: int = 120,
        auto_freeze_patience: int = 3,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        exit_rank_multiplier: int = 2,
        probability_margin_threshold: float = 0.20,
        minimum_difficulty_weight: float = 0.25,
        ramp_start_epoch: int = 25,
        ramp_end_epoch: int = 60,
        loss_cap: float = 2.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=max_pair_weight,
            margin=0.0,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            active_pair_count=active_pair_count,
            confusion_decay=confusion_decay,
            freeze_pairs_epoch=freeze_pairs_epoch,
            min_pair_score=min_pair_score,
            min_class_observations=min_class_observations,
            entry_persistence=entry_persistence,
            exit_rank_multiplier=exit_rank_multiplier,
            score_weighted=False,
        )
        if not 1 <= stable_pair_count < self.active_pair_count:
            raise ValueError(
                "DCMG-CGR-V3 stable_pair_count must be in "
                "[1, active_pair_count)"
            )
        if candidate_pool_size < self.active_pair_count:
            raise ValueError(
                "DCMG-CGR-V3 candidate_pool_size must be >= active_pair_count"
            )
        if min_auto_freeze_epoch < 0 or auto_freeze_patience < 1:
            raise ValueError("Invalid DCMG-CGR-V3 auto-freeze settings")
        if not 0.0 < probability_margin_threshold <= 1.0:
            raise ValueError(
                "DCMG-CGR-V3 probability_margin_threshold must be in (0, 1]"
            )
        if not 0.0 <= minimum_difficulty_weight <= 1.0:
            raise ValueError(
                "DCMG-CGR-V3 minimum_difficulty_weight must be in [0, 1]"
            )
        if ramp_start_epoch < 0 or ramp_end_epoch < ramp_start_epoch:
            raise ValueError("Invalid DCMG-CGR-V3 ramp epoch range")
        if loss_cap <= 0.0:
            raise ValueError("DCMG-CGR-V3 loss_cap must be > 0")

        maximum_pairs = self.num_classes * (self.num_classes - 1) // 2
        self.max_pair_weight = float(max_pair_weight)
        self.stable_pair_count = int(stable_pair_count)
        self.candidate_pool_size = min(
            int(candidate_pool_size), maximum_pairs
        )
        self.min_auto_freeze_epoch = int(min_auto_freeze_epoch)
        self.auto_freeze_patience = int(auto_freeze_patience)
        self.probability_margin_threshold = float(
            probability_margin_threshold
        )
        self.minimum_difficulty_weight = float(
            minimum_difficulty_weight
        )
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.loss_cap = float(loss_cap)
        self._stable_pairs: list[tuple[int, int]] = []
        self._exploration_pair: tuple[int, int] | None = None
        self._unchanged_update_count = 0
        self._auto_frozen = False
        self._auto_frozen_epoch: int | None = None

    def should_observe_teacher(self) -> bool:
        return not self._auto_frozen and super().should_observe_teacher()

    def effective_pair_weight(self) -> float:
        if self.current_epoch < self.ramp_start_epoch:
            return 0.0
        if self.ramp_end_epoch <= self.ramp_start_epoch:
            return self.max_pair_weight
        progress = (
            (self.current_epoch - self.ramp_start_epoch)
            / (self.ramp_end_epoch - self.ramp_start_epoch)
        )
        return self.max_pair_weight * min(max(progress, 0.0), 1.0)

    @torch.no_grad()
    def _select_active_pairs(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        scored_pairs: list[tuple[float, int, int]] = []
        for first in range(self.num_classes):
            if int(class_counts[first].item()) < self.min_class_observations:
                continue
            for second in range(first + 1, self.num_classes):
                if int(class_counts[second].item()) < self.min_class_observations:
                    continue
                score = float(score_matrix[first, second].item())
                if score >= self.min_pair_score:
                    scored_pairs.append((score, first, second))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        streak_pool_size = max(
            self.candidate_pool_size,
            self.active_pair_count * self.exit_rank_multiplier,
        )
        streak_pool = scored_pairs[:streak_pool_size]
        new_streaks = torch.zeros_like(self.candidate_streaks)
        for _, first, second in streak_pool:
            streak = int(self.candidate_streaks[first, second].item()) + 1
            new_streaks[first, second] = streak
            new_streaks[second, first] = streak
        self.candidate_streaks.copy_(new_streaks)

        eligible = [
            (score, first, second)
            for score, first, second in scored_pairs
            if int(self.candidate_streaks[first, second].item())
            >= self.entry_persistence
        ]
        # Stable slots use the strongest persistence-qualified candidates.
        # They may still be replaced when a consistently stronger relation
        # emerges, avoiding the early-lock problem of purely sticky slots.
        stable_pairs = [
            (first, second)
            for _, first, second in eligible[: self.stable_pair_count]
        ]

        exploration_pair = next(
            (
                (first, second)
                for _, first, second in eligible
                if (first, second) not in stable_pairs
            ),
            None,
        )
        new_active_pairs = list(stable_pairs)
        if exploration_pair is not None:
            new_active_pairs.append(exploration_pair)
        new_active_pairs = new_active_pairs[: self.active_pair_count]

        previous_set = set(self._active_pairs)
        current_set = set(new_active_pairs)
        if source_epoch < self.min_auto_freeze_epoch:
            self._unchanged_update_count = 0
        elif current_set and current_set == previous_set:
            self._unchanged_update_count += 1
        else:
            self._unchanged_update_count = 0

        self._stable_pairs = list(stable_pairs)
        self._exploration_pair = exploration_pair
        self._active_pairs = new_active_pairs

        if (
            source_epoch >= self.min_auto_freeze_epoch
            and self._unchanged_update_count >= self.auto_freeze_patience
            and len(self._active_pairs) == self.active_pair_count
        ):
            self._auto_frozen = True
            self._auto_frozen_epoch = int(source_epoch)

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "active_pairs": self.active_pair_records(score_matrix),
                "stable_pairs": [
                    self._pair_record(pair, score_matrix, "stable")
                    for pair in self._stable_pairs
                ],
                "exploration_pair": (
                    self._pair_record(
                        self._exploration_pair,
                        score_matrix,
                        "exploration",
                    )
                    if self._exploration_pair is not None
                    else None
                ),
                "unchanged_update_count": int(
                    self._unchanged_update_count
                ),
                "auto_frozen": bool(self._auto_frozen),
                "top_candidates": [
                    {
                        "class_i": self.class_names[first],
                        "class_j": self.class_names[second],
                        "class_i_index": int(first),
                        "class_j_index": int(second),
                        "score": float(score),
                        "entry_streak": int(
                            self.candidate_streaks[first, second].item()
                        ),
                    }
                    for score, first, second in scored_pairs[
                        : self.candidate_pool_size
                    ]
                ],
            }
        )

    def _pair_record(
        self,
        pair: tuple[int, int],
        score_matrix: torch.Tensor,
        role: str,
    ) -> dict[str, Any]:
        first, second = pair
        return {
            "class_i": self.class_names[first],
            "class_j": self.class_names[second],
            "class_i_index": int(first),
            "class_j_index": int(second),
            "score": float(score_matrix[first, second].item()),
            "role": str(role),
        }

    def active_pair_records(
        self,
        score_matrix: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.smoothed_confusion if score_matrix is None else score_matrix
        records: list[dict[str, Any]] = []
        stable_set = set(self._stable_pairs)
        for pair in self._active_pairs:
            role = "stable" if pair in stable_set else "exploration"
            records.append(self._pair_record(pair, scores, role))
        return records

    def graph_state(self) -> dict[str, Any]:
        state = super().graph_state()
        state.update(
            {
                "variant": "dcmg_cgr_v3_dynamic_pair_routing",
                "max_pair_weight": float(self.max_pair_weight),
                "effective_pair_weight": float(
                    self.effective_pair_weight()
                ),
                "stable_pair_count": int(self.stable_pair_count),
                "candidate_pool_size": int(self.candidate_pool_size),
                "probability_margin_threshold": float(
                    self.probability_margin_threshold
                ),
                "minimum_difficulty_weight": float(
                    self.minimum_difficulty_weight
                ),
                "auto_frozen": bool(self._auto_frozen),
                "auto_frozen_epoch": self._auto_frozen_epoch,
                "unchanged_update_count": int(
                    self._unchanged_update_count
                ),
                "stable_pairs": [
                    self._pair_record(
                        pair, self.smoothed_confusion, "stable"
                    )
                    for pair in self._stable_pairs
                ],
                "exploration_pair": (
                    self._pair_record(
                        self._exploration_pair,
                        self.smoothed_confusion,
                        "exploration",
                    )
                    if self._exploration_pair is not None
                    else None
                ),
            }
        )
        return state

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        pair_loss = classification.float().sum() * 0.0
        active_class_rate = 0.0
        top2_pair_match_rate = 0.0
        low_margin_rate = 0.0
        supervision_rate = 0.0
        mean_probability_gap = 0.0
        mean_difficulty_weight = 0.0
        effective_weight = self.effective_pair_weight()

        guidance_enabled = (
            self.current_epoch > self.warmup_epochs
            and effective_weight > 0.0
            and bool(self._active_pairs)
        )
        if guidance_enabled:
            with torch.no_grad():
                probabilities = torch.softmax(main_logits.detach().float(), dim=1)
                top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
                probability_gaps = top_probabilities[:, 0] - top_probabilities[:, 1]
                low_margin_mask = (
                    probability_gaps <= self.probability_margin_threshold
                )

            active_class_mask = torch.zeros_like(targets, dtype=torch.bool)
            pair_match_mask = torch.zeros_like(targets, dtype=torch.bool)
            supervised_mask = torch.zeros_like(targets, dtype=torch.bool)
            per_sample_losses = torch.zeros(
                targets.shape[0], device=main_logits.device, dtype=torch.float32
            )
            per_sample_weights = torch.zeros_like(per_sample_losses)

            for first, second in self._active_pairs:
                target_in_pair = (targets == first) | (targets == second)
                active_class_mask |= target_in_pair
                top2_matches = (
                    ((top_indices[:, 0] == first) & (top_indices[:, 1] == second))
                    | ((top_indices[:, 0] == second) & (top_indices[:, 1] == first))
                )
                matched = target_in_pair & top2_matches
                selected = matched & low_margin_mask
                pair_match_mask |= matched
                supervised_mask |= selected
                if not bool(selected.any()):
                    continue

                binary_targets = (targets[selected] == second).long()
                selected_pair_logits = main_logits[selected][:, [first, second]].float()
                selected_losses = F.cross_entropy(
                    selected_pair_logits,
                    binary_targets,
                    reduction="none",
                ).clamp(max=self.loss_cap)
                difficulty_weights = (
                    1.0
                    - probability_gaps[selected]
                    / self.probability_margin_threshold
                ).clamp(
                    min=self.minimum_difficulty_weight,
                    max=1.0,
                )
                per_sample_losses[selected] = selected_losses
                per_sample_weights[selected] = difficulty_weights

            active_class_rate = float(
                active_class_mask.float().mean().detach().item()
            )
            top2_pair_match_rate = float(
                pair_match_mask.float().mean().detach().item()
            )
            low_margin_rate = float(
                low_margin_mask.float().mean().detach().item()
            )
            supervision_rate = float(
                supervised_mask.float().mean().detach().item()
            )
            if bool(supervised_mask.any()):
                selected_weights = per_sample_weights[supervised_mask]
                pair_loss = (
                    per_sample_losses[supervised_mask] * selected_weights
                ).sum() / selected_weights.sum().clamp_min(1e-8)
                mean_probability_gap = float(
                    probability_gaps[supervised_mask].mean().item()
                )
                mean_difficulty_weight = float(
                    selected_weights.mean().detach().item()
                )

        total = classification + effective_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "dcmg_cgr_v3_pair_loss": float(pair_loss.detach().item()),
            "dcmg_cgr_v3_active_pair_count": float(len(self._active_pairs)),
            "dcmg_cgr_v3_active_class_rate": active_class_rate,
            "dcmg_cgr_v3_top2_pair_match_rate": top2_pair_match_rate,
            "dcmg_cgr_v3_low_margin_rate": low_margin_rate,
            "dcmg_cgr_v3_supervision_rate": supervision_rate,
            "dcmg_cgr_v3_mean_probability_gap": mean_probability_gap,
            "dcmg_cgr_v3_mean_difficulty_weight": mean_difficulty_weight,
            "dcmg_cgr_v3_effective_pair_weight": float(effective_weight),
            "dcmg_cgr_v3_auto_frozen": float(self._auto_frozen),
        }
        return total


class DirectionalCoverageDynamicConfusionLoss(nn.Module):
    """Direction-aware dynamic pair guidance with controlled coverage.

    The EMA teacher estimates directed confusion edges ``true -> competitor``
    from training samples. Two stable edges use hysteresis replacement while
    one exploration edge rotates through the remaining persistent candidate
    pool. Every sample whose label owns an active edge gets a light binary
    main-logit loss; samples whose main Top-2 exactly matches that edge and
    remains low-margin receive an additional boost. The weighted sum is
    divided by the full batch size so rare samples cannot dominate a batch.
    No auxiliary module is added to the deployable model.
    """

    requires_dcmg = True
    graph_history_filename = "dcmg_cgr_v4_graph_history.json"

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        label_smoothing: float,
        max_pair_weight: float = 0.06,
        warmup_epochs: int = 20,
        update_interval: int = 5,
        stable_pair_count: int = 2,
        candidate_pool_size: int = 6,
        confusion_decay: float = 0.90,
        freeze_pairs_epoch: int = 165,
        min_auto_freeze_epoch: int = 145,
        auto_freeze_patience: int = 4,
        freeze_score_tolerance: float = 0.05,
        min_edge_score: float = 0.005,
        min_class_observations: int = 8,
        entry_persistence: int = 2,
        stable_replacement_ratio: float = 1.12,
        exploration_tenure_updates: int = 2,
        base_sample_weight: float = 0.15,
        hard_sample_boost: float = 0.85,
        probability_margin_threshold: float = 0.25,
        ramp_start_epoch: int = 25,
        ramp_end_epoch: int = 65,
        loss_cap: float = 2.0,
    ) -> None:
        super().__init__()
        maximum_edges = num_classes * (num_classes - 1)
        if num_classes < 2 or len(class_names) != num_classes:
            raise ValueError("DCMG-CGR-V4 requires matching classes")
        if not 1 <= stable_pair_count < maximum_edges:
            raise ValueError("DCMG-CGR-V4 stable_pair_count is invalid")
        if candidate_pool_size <= stable_pair_count:
            raise ValueError(
                "DCMG-CGR-V4 candidate_pool_size must exceed stable slots"
            )
        if not 0.0 <= confusion_decay < 1.0:
            raise ValueError("DCMG-CGR-V4 confusion_decay must be in [0, 1)")
        if warmup_epochs < 0 or update_interval < 1:
            raise ValueError("Invalid DCMG-CGR-V4 warmup/update interval")
        if min_auto_freeze_epoch < 0 or auto_freeze_patience < 1:
            raise ValueError("Invalid DCMG-CGR-V4 auto-freeze settings")
        if not 0.0 <= freeze_score_tolerance <= 1.0:
            raise ValueError("freeze_score_tolerance must be in [0, 1]")
        if entry_persistence < 1 or stable_replacement_ratio < 1.0:
            raise ValueError("Invalid V4 persistence/replacement settings")
        if exploration_tenure_updates < 1:
            raise ValueError("exploration_tenure_updates must be >= 1")
        if base_sample_weight < 0.0 or hard_sample_boost < 0.0:
            raise ValueError("V4 sample weights must be non-negative")
        if base_sample_weight + hard_sample_boost <= 0.0:
            raise ValueError("V4 needs a positive sample weight")
        if not 0.0 < probability_margin_threshold <= 1.0:
            raise ValueError("V4 probability margin must be in (0, 1]")
        if ramp_start_epoch < 0 or ramp_end_epoch < ramp_start_epoch:
            raise ValueError("Invalid DCMG-CGR-V4 ramp range")
        if max_pair_weight < 0.0 or loss_cap <= 0.0:
            raise ValueError("Invalid DCMG-CGR-V4 loss settings")

        self.num_classes = int(num_classes)
        self.class_names = [str(name) for name in class_names]
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.max_pair_weight = float(max_pair_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.stable_pair_count = int(stable_pair_count)
        self.candidate_pool_size = min(
            int(candidate_pool_size), maximum_edges
        )
        self.confusion_decay = float(confusion_decay)
        self.freeze_pairs_epoch = int(freeze_pairs_epoch)
        self.min_auto_freeze_epoch = int(min_auto_freeze_epoch)
        self.auto_freeze_patience = int(auto_freeze_patience)
        self.freeze_score_tolerance = float(freeze_score_tolerance)
        self.min_edge_score = float(min_edge_score)
        self.min_class_observations = int(min_class_observations)
        self.entry_persistence = int(entry_persistence)
        self.stable_replacement_ratio = float(stable_replacement_ratio)
        self.exploration_tenure_updates = int(exploration_tenure_updates)
        self.base_sample_weight = float(base_sample_weight)
        self.hard_sample_boost = float(hard_sample_boost)
        self.probability_margin_threshold = float(
            probability_margin_threshold
        )
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.loss_cap = float(loss_cap)

        self.current_epoch = 0
        self.last_components: dict[str, float] = {}
        self._stable_edges: list[tuple[int, int]] = []
        self._exploration_edge: tuple[int, int] | None = None
        self._exploration_updates = 0
        self._exploration_cursor = 0
        self._selection_history: list[dict[str, Any]] = []
        self._unchanged_update_count = 0
        self._last_active_scores: dict[tuple[int, int], float] = {}
        self._auto_frozen = False
        self._auto_frozen_epoch: int | None = None
        self._has_smoothed_confusion = False

        shape = (self.num_classes, self.num_classes)
        self.register_buffer("epoch_probability_sums", torch.zeros(shape))
        self.register_buffer(
            "epoch_class_counts", torch.zeros(self.num_classes)
        )
        self.register_buffer("smoothed_confusion", torch.zeros(shape))
        self.register_buffer(
            "candidate_streaks", torch.zeros(shape, dtype=torch.long)
        )

    @property
    def _active_edges(self) -> list[tuple[int, int]]:
        edges = list(self._stable_edges)
        if self._exploration_edge is not None:
            edges.append(self._exploration_edge)
        return edges

    def should_observe_teacher(self) -> bool:
        not_reached_hard_freeze = (
            self.freeze_pairs_epoch <= 0
            or self.current_epoch < self.freeze_pairs_epoch
        )
        return not self._auto_frozen and not_reached_hard_freeze

    @torch.no_grad()
    def observe_teacher(
        self,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not self.should_observe_teacher():
            return
        probabilities = torch.softmax(teacher_logits.detach().float(), dim=1)
        detached_targets = targets.detach().long()
        self.epoch_probability_sums.index_add_(
            0, detached_targets, probabilities
        )
        self.epoch_class_counts.add_(
            torch.bincount(
                detached_targets, minlength=self.num_classes
            ).to(self.epoch_class_counts.dtype)
        )

    def effective_pair_weight(self) -> float:
        if self.current_epoch < self.ramp_start_epoch:
            return 0.0
        if self.ramp_end_epoch <= self.ramp_start_epoch:
            return self.max_pair_weight
        progress = (
            (self.current_epoch - self.ramp_start_epoch)
            / (self.ramp_end_epoch - self.ramp_start_epoch)
        )
        return self.max_pair_weight * min(max(progress, 0.0), 1.0)

    def _edge_record(
        self,
        edge: tuple[int, int],
        score_matrix: torch.Tensor,
        role: str,
    ) -> dict[str, Any]:
        true_class, competitor = edge
        return {
            "true_class": self.class_names[true_class],
            "competitor_class": self.class_names[competitor],
            "true_class_index": int(true_class),
            "competitor_class_index": int(competitor),
            "score": float(score_matrix[true_class, competitor].item()),
            "role": str(role),
        }

    def active_pair_records(
        self,
        score_matrix: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.smoothed_confusion if score_matrix is None else score_matrix
        stable = set(self._stable_edges)
        return [
            {
                **self._edge_record(
                    edge,
                    scores,
                    "stable" if edge in stable else "exploration",
                ),
                # Backward-compatible keys used by train.py's text logger.
                "class_i": self.class_names[edge[0]],
                "class_j": self.class_names[edge[1]],
            }
            for edge in self._active_edges
        ]

    @torch.no_grad()
    def _select_active_edges(
        self,
        score_matrix: torch.Tensor,
        class_counts: torch.Tensor,
        source_epoch: int,
    ) -> None:
        # Keep the strongest directed competitor for every true class. This
        # prevents several slots from collapsing onto one class and makes the
        # light-coverage term reach distinct difficult classes.
        scored_edges: list[tuple[float, int, int]] = []
        for true_class in range(self.num_classes):
            if int(class_counts[true_class].item()) < self.min_class_observations:
                continue
            row = score_matrix[true_class].clone()
            row[true_class] = -1.0
            competitor = int(row.argmax().item())
            score = float(row[competitor].item())
            if score >= self.min_edge_score:
                scored_edges.append((score, true_class, competitor))
        scored_edges.sort(key=lambda item: item[0], reverse=True)

        pool = scored_edges[: self.candidate_pool_size]
        new_streaks = torch.zeros_like(self.candidate_streaks)
        for _, true_class, competitor in pool:
            streak = int(
                self.candidate_streaks[true_class, competitor].item()
            ) + 1
            new_streaks[true_class, competitor] = streak
        self.candidate_streaks.copy_(new_streaks)

        eligible = [
            (score, true_class, competitor)
            for score, true_class, competitor in pool
            if int(self.candidate_streaks[true_class, competitor].item())
            >= self.entry_persistence
        ]
        score_lookup = {
            (true_class, competitor): score
            for score, true_class, competitor in eligible
        }

        retained = [
            edge for edge in self._stable_edges if edge in score_lookup
        ]
        for score, true_class, competitor in eligible:
            edge = (true_class, competitor)
            if edge in retained:
                continue
            if len(retained) < self.stable_pair_count:
                retained.append(edge)
                continue
            weakest = min(retained, key=lambda item: score_lookup[item])
            if score > score_lookup[weakest] * self.stable_replacement_ratio:
                retained[retained.index(weakest)] = edge
        retained = sorted(
            set(retained), key=lambda edge: score_lookup[edge], reverse=True
        )[: self.stable_pair_count]

        exploration_candidates = [
            (true_class, competitor)
            for _, true_class, competitor in eligible
            if (true_class, competitor) not in retained
        ]
        if exploration_candidates:
            keep_current = (
                self._exploration_edge in exploration_candidates
                and self._exploration_updates
                < self.exploration_tenure_updates
            )
            if keep_current:
                exploration = self._exploration_edge
                self._exploration_updates += 1
            else:
                if self._exploration_edge in exploration_candidates:
                    start = (
                        exploration_candidates.index(self._exploration_edge) + 1
                    ) % len(exploration_candidates)
                else:
                    start = self._exploration_cursor % len(
                        exploration_candidates
                    )
                exploration = exploration_candidates[start]
                self._exploration_cursor = (start + 1) % len(
                    exploration_candidates
                )
                self._exploration_updates = 1
        else:
            exploration = None
            self._exploration_updates = 0

        previous_stable = list(self._stable_edges)
        previous_scores = dict(self._last_active_scores)
        self._stable_edges = retained
        self._exploration_edge = exploration
        current_edges = self._active_edges
        current_scores = {
            edge: float(score_matrix[edge[0], edge[1]].item())
            for edge in current_edges
        }
        stable_score_drift = max(
            (
                abs(current_scores[edge] - previous_scores.get(edge, 0.0))
                / max(abs(previous_scores.get(edge, 0.0)), 1e-8)
                for edge in self._stable_edges
            ),
            default=float("inf"),
        )
        if (
            source_epoch >= self.min_auto_freeze_epoch
            and self._stable_edges
            and self._stable_edges == previous_stable
            and stable_score_drift <= self.freeze_score_tolerance
        ):
            self._unchanged_update_count += 1
        else:
            self._unchanged_update_count = 0
        self._last_active_scores = current_scores

        if (
            source_epoch >= self.min_auto_freeze_epoch
            and self._unchanged_update_count >= self.auto_freeze_patience
            and len(self._stable_edges) == self.stable_pair_count
            and self._exploration_edge is not None
        ):
            self._auto_frozen = True
            self._auto_frozen_epoch = int(source_epoch)

        self._selection_history.append(
            {
                "source_epoch": int(source_epoch),
                "active_edges": self.active_pair_records(score_matrix),
                "unchanged_update_count": int(self._unchanged_update_count),
                "stable_score_drift": float(stable_score_drift),
                "auto_frozen": bool(self._auto_frozen),
                "top_candidates": [
                    {
                        **self._edge_record(
                            (true_class, competitor),
                            score_matrix,
                            "candidate",
                        ),
                        "entry_streak": int(
                            self.candidate_streaks[
                                true_class, competitor
                            ].item()
                        ),
                    }
                    for _, true_class, competitor in pool
                ],
            }
        )

    @torch.no_grad()
    def _finalize_previous_epoch(self, source_epoch: int) -> None:
        if float(self.epoch_class_counts.sum().item()) <= 0.0:
            return
        mean_probabilities = self.epoch_probability_sums / (
            self.epoch_class_counts.clamp_min(1.0).unsqueeze(1)
        )
        directed_scores = mean_probabilities.clone()
        directed_scores.fill_diagonal_(0.0)
        if self._has_smoothed_confusion:
            self.smoothed_confusion.mul_(self.confusion_decay).add_(
                directed_scores, alpha=1.0 - self.confusion_decay
            )
        else:
            self.smoothed_confusion.copy_(directed_scores)
            self._has_smoothed_confusion = True

        should_update = (
            source_epoch >= self.warmup_epochs
            and self.should_observe_teacher()
            and (source_epoch - self.warmup_epochs) % self.update_interval == 0
        )
        if should_update:
            self._select_active_edges(
                self.smoothed_confusion,
                self.epoch_class_counts,
                source_epoch,
            )
        self.epoch_probability_sums.zero_()
        self.epoch_class_counts.zero_()

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if self.current_epoch > 0 and epoch != self.current_epoch:
            self._finalize_previous_epoch(self.current_epoch)
        self.current_epoch = epoch

    def extra_state(self) -> dict[str, Any]:
        """Serializable dynamic state stored alongside model checkpoints."""
        return {
            "current_epoch": int(self.current_epoch),
            "stable_edges": [list(edge) for edge in self._stable_edges],
            "exploration_edge": (
                list(self._exploration_edge)
                if self._exploration_edge is not None
                else None
            ),
            "exploration_updates": int(self._exploration_updates),
            "exploration_cursor": int(self._exploration_cursor),
            "selection_history": list(self._selection_history),
            "unchanged_update_count": int(self._unchanged_update_count),
            "last_active_scores": [
                [edge[0], edge[1], score]
                for edge, score in self._last_active_scores.items()
            ],
            "auto_frozen": bool(self._auto_frozen),
            "auto_frozen_epoch": self._auto_frozen_epoch,
            "has_smoothed_confusion": bool(self._has_smoothed_confusion),
        }

    def load_extra_state(self, state: dict[str, Any]) -> None:
        self.current_epoch = int(state.get("current_epoch", 0))
        self._stable_edges = [
            (int(edge[0]), int(edge[1]))
            for edge in state.get("stable_edges", [])
        ]
        exploration = state.get("exploration_edge")
        self._exploration_edge = (
            (int(exploration[0]), int(exploration[1]))
            if exploration is not None
            else None
        )
        self._exploration_updates = int(
            state.get("exploration_updates", 0)
        )
        self._exploration_cursor = int(state.get("exploration_cursor", 0))
        self._selection_history = list(state.get("selection_history", []))
        self._unchanged_update_count = int(
            state.get("unchanged_update_count", 0)
        )
        self._last_active_scores = {
            (int(first), int(second)): float(score)
            for first, second, score in state.get("last_active_scores", [])
        }
        self._auto_frozen = bool(state.get("auto_frozen", False))
        frozen_epoch = state.get("auto_frozen_epoch")
        self._auto_frozen_epoch = (
            int(frozen_epoch) if frozen_epoch is not None else None
        )
        self._has_smoothed_confusion = bool(
            state.get("has_smoothed_confusion", False)
        )

    def graph_state(self) -> dict[str, Any]:
        return {
            "variant": "dcmg_cgr_v4_directional_coverage",
            "current_epoch": int(self.current_epoch),
            "effective_pair_weight": float(self.effective_pair_weight()),
            "active_edges": self.active_pair_records(),
            "stable_edges": [
                self._edge_record(edge, self.smoothed_confusion, "stable")
                for edge in self._stable_edges
            ],
            "exploration_edge": (
                self._edge_record(
                    self._exploration_edge,
                    self.smoothed_confusion,
                    "exploration",
                )
                if self._exploration_edge is not None
                else None
            ),
            "auto_frozen": bool(self._auto_frozen),
            "auto_frozen_epoch": self._auto_frozen_epoch,
            "selection_history": list(self._selection_history),
        }

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        zero = classification.float().sum() * 0.0
        pair_loss = zero
        coverage_rate = 0.0
        hard_route_rate = 0.0
        mean_sample_weight = 0.0
        effective_supervision_mass = 0.0
        effective_weight = self.effective_pair_weight()

        guidance_enabled = (
            self.current_epoch > self.warmup_epochs
            and effective_weight > 0.0
            and bool(self._active_edges)
        )
        if guidance_enabled:
            with torch.no_grad():
                probabilities = torch.softmax(
                    main_logits.detach().float(), dim=1
                )
                top_probabilities, top_indices = probabilities.topk(
                    k=2, dim=1
                )
                probability_gaps = (
                    top_probabilities[:, 0] - top_probabilities[:, 1]
                )

            per_sample_losses = torch.zeros(
                targets.shape[0], device=main_logits.device, dtype=torch.float32
            )
            per_sample_weights = torch.zeros_like(per_sample_losses)
            covered_mask = torch.zeros_like(targets, dtype=torch.bool)
            hard_mask = torch.zeros_like(targets, dtype=torch.bool)

            # Candidate construction keeps at most one directed active edge
            # per true class. The max-weight replacement remains defensive.
            for true_class, competitor in self._active_edges:
                target_mask = targets == true_class
                if not bool(target_mask.any()):
                    continue
                binary_targets = torch.zeros(
                    int(target_mask.sum().item()),
                    dtype=torch.long,
                    device=targets.device,
                )
                pair_logits = main_logits[target_mask][
                    :, [true_class, competitor]
                ].float()
                losses = F.cross_entropy(
                    pair_logits, binary_targets, reduction="none"
                ).clamp(max=self.loss_cap)
                top2_matches = (
                    ((top_indices[:, 0] == true_class)
                     & (top_indices[:, 1] == competitor))
                    | ((top_indices[:, 1] == true_class)
                       & (top_indices[:, 0] == competitor))
                )
                difficult = (
                    target_mask
                    & top2_matches
                    & (
                        probability_gaps
                        <= self.probability_margin_threshold
                    )
                )
                weights = torch.full_like(losses, self.base_sample_weight)
                weights += self.hard_sample_boost * difficult[target_mask].float()
                indices = target_mask.nonzero(as_tuple=False).squeeze(1)
                replace = weights > per_sample_weights[indices]
                selected_indices = indices[replace]
                per_sample_losses[selected_indices] = losses[replace]
                per_sample_weights[selected_indices] = weights[replace]
                covered_mask |= target_mask
                hard_mask |= difficult

            # Full-batch normalization is intentional: coverage and strength
            # cannot silently explode when only one sample is selected.
            pair_loss = (
                per_sample_losses * per_sample_weights
            ).sum() / max(int(targets.shape[0]), 1)
            coverage_rate = float(
                covered_mask.float().mean().detach().item()
            )
            hard_route_rate = float(
                hard_mask.float().mean().detach().item()
            )
            if bool(covered_mask.any()):
                mean_sample_weight = float(
                    per_sample_weights[covered_mask].mean().detach().item()
                )
            effective_supervision_mass = float(
                (
                    per_sample_weights.sum()
                    / max(int(targets.shape[0]), 1)
                ).detach().item()
            )

        total = classification + effective_weight * pair_loss
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "dcmg_cgr_v4_pair_loss": float(pair_loss.detach().item()),
            "dcmg_cgr_v4_active_edge_count": float(len(self._active_edges)),
            "dcmg_cgr_v4_coverage_rate": coverage_rate,
            "dcmg_cgr_v4_hard_route_rate": hard_route_rate,
            "dcmg_cgr_v4_mean_sample_weight": mean_sample_weight,
            "dcmg_cgr_v4_effective_supervision_mass": (
                effective_supervision_mass
            ),
            "dcmg_cgr_v4_effective_pair_weight": float(effective_weight),
            "dcmg_cgr_v4_auto_frozen": float(self._auto_frozen),
        }
        return total


class DPHGClassificationLoss(nn.Module):
    """Main CE plus dynamically routed, training-only pair guidance.

    Unlike CGRH, the deployable main head always receives its own full
    multi-class supervision.  Pair/refinement losses are applied only when the
    current main-head Top-2 prediction has a low confidence margin and contains
    the ground-truth class.  Consequently the auxiliary head can be removed at
    inference without changing the classifier being selected on validation.
    """

    requires_dphg = True

    def __init__(
        self,
        label_smoothing: float,
        pair_indices: list[tuple[int, int]],
        pair_weight: float = 0.20,
        refine_weight: float = 0.20,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        if not pair_indices:
            raise ValueError("DPHG loss requires class pairs")
        if pair_weight < 0.0 or refine_weight < 0.0:
            raise ValueError("DPHG loss weights must be >= 0")
        if warmup_epochs < 0:
            raise ValueError("DPHG warmup_epochs must be >= 0")
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )
        self.pair_indices = [(int(a), int(b)) for a, b in pair_indices]
        self.pair_weight = float(pair_weight)
        self.refine_weight = float(refine_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0
        self.last_components: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        dphg_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        classification = self.cross_entropy(main_logits, targets)
        zero = classification.float().sum() * 0.0
        pair_loss = zero
        refine_loss = zero
        route_rate = 0.0
        supervised_route_rate = 0.0

        guidance_enabled = (
            dphg_state is not None
            and self.current_epoch > self.warmup_epochs
            and (self.pair_weight > 0.0 or self.refine_weight > 0.0)
        )
        if guidance_enabled:
            pair_logits = dphg_state["pair_logits"]
            route_mask = dphg_state["route_mask"]
            state_pairs = dphg_state["pair_indices"]
            refined_logits = dphg_state["refined_logits"]
            expected_pairs = torch.tensor(
                self.pair_indices,
                dtype=state_pairs.dtype,
                device=state_pairs.device,
            )
            if state_pairs.shape != expected_pairs.shape or not torch.equal(
                state_pairs, expected_pairs
            ):
                raise RuntimeError("DPHG model/loss pair definitions do not match")

            first_indices = state_pairs[:, 0].unsqueeze(0)
            second_indices = state_pairs[:, 1].unsqueeze(0)
            expanded_targets = targets.unsqueeze(1)
            target_in_pair = (
                (expanded_targets == first_indices)
                | (expanded_targets == second_indices)
            )
            supervised_pair_mask = route_mask & target_in_pair
            supervised_sample_mask = supervised_pair_mask.any(dim=1)
            route_rate = float(route_mask.any(dim=1).float().mean().item())
            supervised_route_rate = float(
                supervised_sample_mask.float().mean().item()
            )

            if bool(supervised_pair_mask.any()):
                binary_targets = (
                    expanded_targets == second_indices
                ).long()[supervised_pair_mask]
                pair_loss = F.cross_entropy(
                    pair_logits.float()[supervised_pair_mask],
                    binary_targets,
                )
                refine_loss = self.cross_entropy(
                    refined_logits[supervised_sample_mask],
                    targets[supervised_sample_mask],
                )

        total = (
            classification
            + self.pair_weight * pair_loss
            + self.refine_weight * refine_loss
        )
        self.last_components = {
            "classification_loss": float(classification.detach().item()),
            "dphg_pair_loss": float(pair_loss.detach().item()),
            "dphg_refine_loss": float(refine_loss.detach().item()),
            "dphg_route_rate": route_rate,
            "dphg_supervised_route_rate": supervised_route_rate,
        }
        return total


def build_training_criterion(
    experiment_config: dict[str, Any],
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> nn.Module:
    label_smoothing = float(experiment_config["label_smoothing"])
    difficulty_config = model_config.get("difficulty_weighted_loss", {})
    confusion_config = model_config.get("confusion_loss", {})
    binary_aux_config = model_config.get("binary_aux_loss", {})
    global_aux_config = model_config.get("global_aux_loss", {})
    lgrc_config = model_config.get("lgrc", {})
    rcaux_config = model_config.get("rcaux", {})
    opt3_aux_config = model_config.get("opt3_aux", {})
    cgrh_config = model_config.get("cgrh", {})
    cgrh_aux_config = model_config.get("cgrh_aux", {})
    apd_aux_config = model_config.get("apd_aux", {})
    aocg_aux_config = model_config.get("aocg_aux", {})
    odpm_cgrh_config = model_config.get("odpm_cgrh_trainonly", {})
    odpm_hard_v2_config = model_config.get("odpm_cgrh_hard_v2", {})
    coverage_v3_config = model_config.get("odpm_cgrh_coverage_v3", {})
    auto_cgrh_config = model_config.get("auto_cgrh", {})
    lmc_config = model_config.get("lmc", {})
    soft_cgrh_config = model_config.get("soft_dynamic_cgrh", {})
    continuous_top3_config = model_config.get("continuous_top3_cgrh", {})
    competence_top3_config = model_config.get(
        "competence_adaptive_top3_cgrh", {}
    )
    stable_ca_cgrh_config = model_config.get("stable_ca_cgrh", {})
    competence_top2_config = model_config.get(
        "competence_adaptive_top2_cgrh", {}
    )
    competence_top3_cdw_config = model_config.get(
        "competence_adaptive_top3_cgrh_cdw", {}
    )
    competence_top3_v2a_config = model_config.get(
        "competence_adaptive_top3_cgrh_v2a", {}
    )
    ca_lite_dual_ce_config = model_config.get("ca_lite_dual_ce", {})
    dcmg_config = model_config.get("dcmg", {})
    dcmg_v2_config = model_config.get("dcmg_v2", {})
    dcmg_cgr_v3_config = model_config.get("dcmg_cgr_v3", {})
    dcmg_cgr_v4_config = model_config.get("dcmg_cgr_v4", {})
    dphg_config = model_config.get("dphg", {})
    difficulty_enabled = bool(difficulty_config.get("enabled", False))
    confusion_enabled = bool(confusion_config.get("enabled", False))
    binary_aux_enabled = bool(binary_aux_config.get("enabled", False))
    global_aux_enabled = bool(global_aux_config.get("enabled", False))
    lgrc_enabled = bool(lgrc_config.get("enabled", False))
    rcaux_enabled = bool(rcaux_config.get("enabled", False))
    opt3_aux_enabled = bool(opt3_aux_config.get("enabled", False))
    cgrh_enabled = bool(cgrh_config.get("enabled", False))
    cgrh_aux_enabled = bool(cgrh_aux_config.get("enabled", False))
    apd_aux_enabled = bool(apd_aux_config.get("enabled", False))
    aocg_aux_enabled = bool(aocg_aux_config.get("enabled", False))
    odpm_cgrh_enabled = bool(odpm_cgrh_config.get("enabled", False))
    odpm_hard_v2_enabled = bool(
        odpm_hard_v2_config.get("enabled", False)
    )
    coverage_v3_enabled = bool(coverage_v3_config.get("enabled", False))
    auto_cgrh_enabled = bool(auto_cgrh_config.get("enabled", False))
    lmc_enabled = bool(lmc_config.get("enabled", False))
    soft_cgrh_enabled = bool(soft_cgrh_config.get("enabled", False))
    continuous_top3_enabled = bool(
        continuous_top3_config.get("enabled", False)
    )
    competence_top3_enabled = bool(
        competence_top3_config.get("enabled", False)
    )
    stable_ca_cgrh_enabled = bool(
        stable_ca_cgrh_config.get("enabled", False)
    )
    competence_top2_enabled = bool(
        competence_top2_config.get("enabled", False)
    )
    competence_top3_cdw_enabled = bool(
        competence_top3_cdw_config.get("enabled", False)
    )
    competence_top3_v2a_enabled = bool(
        competence_top3_v2a_config.get("enabled", False)
    )
    ca_lite_dual_ce_enabled = bool(
        ca_lite_dual_ce_config.get("enabled", False)
    )
    dcmg_enabled = bool(dcmg_config.get("enabled", False))
    dcmg_v2_enabled = bool(dcmg_v2_config.get("enabled", False))
    dcmg_cgr_v3_enabled = bool(
        dcmg_cgr_v3_config.get("enabled", False)
    )
    dcmg_cgr_v4_enabled = bool(
        dcmg_cgr_v4_config.get("enabled", False)
    )
    dphg_enabled = bool(dphg_config.get("enabled", False))
    enabled_count = (
        int(difficulty_enabled)
        + int(confusion_enabled)
        + int(binary_aux_enabled)
        + int(global_aux_enabled)
        + int(lgrc_enabled)
        + int(rcaux_enabled)
        + int(opt3_aux_enabled)
        + int(cgrh_enabled)
        + int(cgrh_aux_enabled)
        + int(apd_aux_enabled)
        + int(aocg_aux_enabled)
        + int(odpm_cgrh_enabled)
        + int(odpm_hard_v2_enabled)
        + int(coverage_v3_enabled)
        + int(auto_cgrh_enabled)
        + int(lmc_enabled)
        + int(soft_cgrh_enabled)
        + int(continuous_top3_enabled)
        + int(competence_top3_enabled)
        + int(stable_ca_cgrh_enabled)
        + int(competence_top2_enabled)
        + int(competence_top3_cdw_enabled)
        + int(competence_top3_v2a_enabled)
        + int(ca_lite_dual_ce_enabled)
        + int(dcmg_enabled)
        + int(dcmg_v2_enabled)
        + int(dcmg_cgr_v3_enabled)
        + int(dcmg_cgr_v4_enabled)
        + int(dphg_enabled)
    )
    if enabled_count > 1:
        raise ValueError(
            "Enable only one of difficulty_weighted_loss, confusion_loss, "
            "binary_aux_loss, global_aux_loss, lgrc, rcaux, opt3_aux, "
            "cgrh, cgrh_aux, "
            "apd_aux, "
            "aocg_aux, odpm_cgrh_trainonly, odpm_cgrh_hard_v2, "
            "odpm_cgrh_coverage_v3, auto_cgrh, lmc, soft_dynamic_cgrh, "
            "continuous_top3_cgrh, competence_adaptive_top3_cgrh, "
            "stable_ca_cgrh, "
            "competence_adaptive_top2_cgrh, "
            "competence_adaptive_top3_cgrh_cdw, "
            "competence_adaptive_top3_cgrh_v2a, "
            "ca_lite_dual_ce, "
            "dcmg, dcmg_v2, "
            "dcmg_cgr_v3, dcmg_cgr_v4, "
            "and dphg"
        )
    if difficulty_enabled:
        return DifficultyWeightedCrossEntropyLoss(
            num_classes=int(dataset_config["num_classes"]),
            label_smoothing=label_smoothing,
            ema_beta=float(difficulty_config.get("ema_beta", 0.90)),
            warmup_epochs=int(
                difficulty_config.get("warmup_epochs", 5)
            ),
            min_weight=float(difficulty_config.get("min_weight", 0.80)),
            max_weight=float(difficulty_config.get("max_weight", 1.20)),
        )
    if binary_aux_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        healthy_name = str(binary_aux_config.get("healthy_class", "haozao"))
        if healthy_name not in class_names:
            raise ValueError(
                f"binary_aux_loss.healthy_class='{healthy_name}' "
                f"not in class_names={class_names}"
            )
        return BinaryAuxClassificationLoss(
            label_smoothing=label_smoothing,
            healthy_class_index=class_names.index(healthy_name),
            auxiliary_weight=float(binary_aux_config.get("weight", 0.30)),
        )
    if global_aux_enabled:
        return GlobalAuxClassificationLoss(
            label_smoothing=label_smoothing,
            auxiliary_weight=float(global_aux_config.get("weight", 0.30)),
        )
    if lgrc_enabled:
        return LGRCClassificationLoss(
            label_smoothing=label_smoothing,
            max_weight=float(lgrc_config.get("max_weight", 0.10)),
            relation_fraction=float(
                lgrc_config.get("relation_fraction", 0.50)
            ),
            start_epoch=int(lgrc_config.get("start_epoch", 10)),
            warmup_end_epoch=int(
                lgrc_config.get("warmup_end_epoch", 30)
            ),
            decay_start_epoch=int(
                lgrc_config.get("decay_start_epoch", 180)
            ),
            end_epoch=int(lgrc_config.get("end_epoch", 200)),
            smooth_l1_beta=float(
                lgrc_config.get("smooth_l1_beta", 0.10)
            ),
        )
    if rcaux_enabled:
        if bool(rcaux_config.get("usage_floor_enabled", False)):
            return UFBRCAuxClassificationLoss(
                label_smoothing=label_smoothing,
                max_weight=float(rcaux_config.get("max_weight", 0.15)),
                warmup_end_epoch=int(
                    rcaux_config.get("warmup_end_epoch", 20)
                ),
                decay_start_epoch=int(
                    rcaux_config.get("decay_start_epoch", 160)
                ),
                end_epoch=int(rcaux_config.get("end_epoch", 200)),
                floor_weight=float(
                    rcaux_config.get("floor_weight", 0.02)
                ),
                floor_start_epoch=int(
                    rcaux_config.get("floor_start_epoch", 10)
                ),
            )
        rcaux_variant = str(
            rcaux_config.get("variant", "base")
        ).strip().lower()
        if rcaux_variant in RCAuxV4ClassificationLoss.SUPPORTED_VARIANTS:
            return RCAuxV4ClassificationLoss(
                label_smoothing=label_smoothing,
                max_weight=float(rcaux_config.get("max_weight", 0.15)),
                warmup_end_epoch=int(
                    rcaux_config.get("warmup_end_epoch", 20)
                ),
                decay_start_epoch=int(
                    rcaux_config.get("decay_start_epoch", 160)
                ),
                end_epoch=int(rcaux_config.get("end_epoch", 200)),
                variant=rcaux_variant,
                mha_temperature=float(
                    rcaux_config.get("mha_temperature", 2.0)
                ),
                mha_beta=float(rcaux_config.get("mha_beta", 0.30)),
                mha_confidence_threshold=float(
                    rcaux_config.get("mha_confidence_threshold", 0.65)
                ),
                regularizer_start_epoch=int(
                    rcaux_config.get("regularizer_start_epoch", 20)
                ),
                hnm_margin=float(rcaux_config.get("hnm_margin", 0.20)),
                hnm_weight=float(rcaux_config.get("hnm_weight", 0.05)),
                hnm_target_probability=float(
                    rcaux_config.get("hnm_target_probability", 0.80)
                ),
                ardrop_kl_weight=float(
                    rcaux_config.get("ardrop_kl_weight", 0.02)
                ),
            )
        return RCAuxClassificationLoss(
            label_smoothing=label_smoothing,
            max_weight=float(rcaux_config.get("max_weight", 0.15)),
            warmup_end_epoch=int(
                rcaux_config.get("warmup_end_epoch", 20)
            ),
            decay_start_epoch=int(
                rcaux_config.get("decay_start_epoch", 160)
            ),
            end_epoch=int(rcaux_config.get("end_epoch", 200)),
        )
    if opt3_aux_enabled:
        return Optimization3AuxClassificationLoss(
            label_smoothing=label_smoothing,
            max_weight=float(opt3_aux_config.get("max_weight", 0.15)),
            warmup_end_epoch=int(
                opt3_aux_config.get("warmup_end_epoch", 20)
            ),
            decay_start_epoch=int(
                opt3_aux_config.get("decay_start_epoch", 160)
            ),
            end_epoch=int(opt3_aux_config.get("end_epoch", 200)),
        )
    if cgrh_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        pair_mode = str(cgrh_config.get("pair_mode", "explicit")).lower()
        if pair_mode == "all":
            pair_names = [
                [class_names[first], class_names[second]]
                for first in range(len(class_names))
                for second in range(first + 1, len(class_names))
            ]
        elif pair_mode == "explicit":
            pair_names = list(cgrh_config.get("pairs", []))
        else:
            raise ValueError(
                "cgrh.pair_mode must be either 'explicit' or 'all'; "
                f"got {pair_mode!r}"
            )
        if not pair_names:
            raise ValueError("cgrh.enabled=true requires non-empty pairs")
        pairs = _resolve_confusion_pairs(pair_names, class_names)
        return CGRHClassificationLoss(
            label_smoothing=label_smoothing,
            pair_indices=pairs,
            auxiliary_weight=float(cgrh_config.get("auxiliary_weight", 0.20)),
        )
    if cgrh_aux_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        pair_mode = str(
            cgrh_aux_config.get("pair_mode", "explicit")
        ).lower()
        if pair_mode == "all":
            pair_names = [
                [class_names[first], class_names[second]]
                for first in range(len(class_names))
                for second in range(first + 1, len(class_names))
            ]
        elif pair_mode == "explicit":
            pair_names = list(cgrh_aux_config.get("pairs", []))
        else:
            raise ValueError(
                "cgrh_aux.pair_mode must be either 'explicit' or 'all'; "
                f"got {pair_mode!r}"
            )
        if not pair_names:
            raise ValueError("cgrh_aux.enabled=true requires non-empty pairs")
        pairs = _resolve_confusion_pairs(pair_names, class_names)
        return CGRHAuxiliaryTrainingLoss(
            label_smoothing=label_smoothing,
            pair_indices=pairs,
            auxiliary_weight=float(
                cgrh_aux_config.get("auxiliary_weight", 0.20)
            ),
        )
    if apd_aux_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        pair_mode = str(apd_aux_config.get("pair_mode", "all")).lower()
        if pair_mode == "all":
            pair_names = [
                [class_names[first], class_names[second]]
                for first in range(len(class_names))
                for second in range(first + 1, len(class_names))
            ]
        elif pair_mode == "explicit":
            pair_names = list(apd_aux_config.get("pairs", []))
        else:
            raise ValueError(
                "apd_aux.pair_mode must be either 'explicit' or 'all'; "
                f"got {pair_mode!r}"
            )
        pairs = _resolve_confusion_pairs(pair_names, class_names)
        return APDAuxiliaryClassificationLoss(
            label_smoothing=label_smoothing,
            pair_indices=pairs,
            auxiliary_weight=float(
                apd_aux_config.get("auxiliary_weight", 0.20)
            ),
        )
    if aocg_aux_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return AdaptiveOnlineConfusionGraphLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=float(aocg_aux_config.get("pair_weight", 0.05)),
            warmup_epochs=int(
                aocg_aux_config.get("warmup_epochs", 20)
            ),
            update_interval=int(
                aocg_aux_config.get("update_interval", 5)
            ),
            active_pair_count=int(
                aocg_aux_config.get("active_pair_count", 3)
            ),
            confusion_decay=float(
                aocg_aux_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                aocg_aux_config.get("freeze_pairs_epoch", 150)
            ),
            min_pair_score=float(
                aocg_aux_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                aocg_aux_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                aocg_aux_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                aocg_aux_config.get("exit_rank_multiplier", 2)
            ),
        )
    if odpm_cgrh_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return OnlineDynamicPairCGRHTrainOnlyLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=float(odpm_cgrh_config.get("pair_weight", 0.15)),
            refine_weight=float(
                odpm_cgrh_config.get("refine_weight", 0.25)
            ),
            warmup_epochs=int(
                odpm_cgrh_config.get("warmup_epochs", 30)
            ),
            update_interval=int(
                odpm_cgrh_config.get("update_interval", 5)
            ),
            active_pair_count=int(
                odpm_cgrh_config.get("active_pair_count", 3)
            ),
            confusion_decay=float(
                odpm_cgrh_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                odpm_cgrh_config.get("freeze_pairs_epoch", 150)
            ),
            min_pair_score=float(
                odpm_cgrh_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                odpm_cgrh_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                odpm_cgrh_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                odpm_cgrh_config.get("exit_rank_multiplier", 2)
            ),
        )
    if odpm_hard_v2_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return OnlineDynamicPairCGRHHardV2Loss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=float(
                odpm_hard_v2_config.get("pair_weight", 0.05)
            ),
            refine_weight=float(
                odpm_hard_v2_config.get("refine_weight", 0.10)
            ),
            probability_margin_threshold=float(
                odpm_hard_v2_config.get(
                    "probability_margin_threshold", 0.20
                )
            ),
            hard_fraction=float(
                odpm_hard_v2_config.get("hard_fraction", 0.50)
            ),
            min_samples_per_group=int(
                odpm_hard_v2_config.get("min_samples_per_group", 1)
            ),
            max_samples_per_group=int(
                odpm_hard_v2_config.get("max_samples_per_group", 0)
            ),
            warmup_epochs=int(
                odpm_hard_v2_config.get("warmup_epochs", 30)
            ),
            update_interval=int(
                odpm_hard_v2_config.get("update_interval", 10)
            ),
            active_pair_count=int(
                odpm_hard_v2_config.get("active_pair_count", 3)
            ),
            confusion_decay=float(
                odpm_hard_v2_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                odpm_hard_v2_config.get("freeze_pairs_epoch", 90)
            ),
            min_pair_score=float(
                odpm_hard_v2_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                odpm_hard_v2_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                odpm_hard_v2_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                odpm_hard_v2_config.get("exit_rank_multiplier", 2)
            ),
        )
    if coverage_v3_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return OnlineDynamicPairCGRHCoverageV3Loss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            dynamic_pair_weight=float(
                coverage_v3_config.get("dynamic_pair_weight", 0.05)
            ),
            max_pair_weight=float(
                coverage_v3_config.get("max_pair_weight", 0.10)
            ),
            final_pair_weight=float(
                coverage_v3_config.get("final_pair_weight", 0.02)
            ),
            warmup_epochs=int(
                coverage_v3_config.get("warmup_epochs", 30)
            ),
            dynamic_ramp_end_epoch=int(
                coverage_v3_config.get("dynamic_ramp_end_epoch", 60)
            ),
            update_interval=int(
                coverage_v3_config.get("update_interval", 10)
            ),
            active_pair_count=int(
                coverage_v3_config.get("active_pair_count", 3)
            ),
            confusion_decay=float(
                coverage_v3_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                coverage_v3_config.get("freeze_pairs_epoch", 90)
            ),
            frozen_ramp_end_epoch=int(
                coverage_v3_config.get("frozen_ramp_end_epoch", 140)
            ),
            decay_start_epoch=int(
                coverage_v3_config.get("decay_start_epoch", 141)
            ),
            decay_end_epoch=int(
                coverage_v3_config.get("decay_end_epoch", 200)
            ),
            min_pair_score=float(
                coverage_v3_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                coverage_v3_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                coverage_v3_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                coverage_v3_config.get("exit_rank_multiplier", 2)
            ),
        )
    if auto_cgrh_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        configured_max_pairs = auto_cgrh_config.get(
            "max_active_pairs", "auto"
        )
        if (
            isinstance(configured_max_pairs, str)
            and configured_max_pairs.lower() == "auto"
        ):
            max_pair_cap = int(auto_cgrh_config.get("max_pair_cap", 8))
            max_active_pairs = min(
                max_pair_cap,
                (int(dataset_config["num_classes"]) + 1) // 2,
            )
        else:
            max_active_pairs = int(configured_max_pairs)
        return AutoCGRHClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            discovery_epochs=int(
                auto_cgrh_config.get("discovery_epochs", 60)
            ),
            discovery_warmup_epochs=int(
                auto_cgrh_config.get("discovery_warmup_epochs", 20)
            ),
            update_interval=int(
                auto_cgrh_config.get("update_interval", 5)
            ),
            max_active_pairs=max_active_pairs,
            min_active_pairs=int(
                auto_cgrh_config.get("min_active_pairs", 2)
            ),
            confusion_decay=float(
                auto_cgrh_config.get("confusion_decay", 0.90)
            ),
            min_pair_score=float(
                auto_cgrh_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                auto_cgrh_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                auto_cgrh_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                auto_cgrh_config.get("exit_rank_multiplier", 2)
            ),
            min_consensus_rate=float(
                auto_cgrh_config.get("min_consensus_rate", 0.60)
            ),
            auxiliary_weight=float(
                auto_cgrh_config.get("auxiliary_weight", 0.20)
            ),
            pair_ramp_epochs=int(
                auto_cgrh_config.get("pair_ramp_epochs", 0)
            ),
            stable_pair_gating=bool(
                auto_cgrh_config.get("stable_pair_gating", False)
            ),
        )
    if lmc_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return LMCClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(lmc_config.get("warmup_epochs", 20)),
            update_interval=int(lmc_config.get("update_interval", 5)),
            min_discovery_epochs=int(
                lmc_config.get("min_discovery_epochs", 40)
            ),
            max_discovery_epochs=int(
                lmc_config.get("max_discovery_epochs", 100)
            ),
            active_pair_count=int(
                lmc_config.get("active_pair_count", 3)
            ),
            refine_pair_count=(
                int(lmc_config["refine_pair_count"])
                if "refine_pair_count" in lmc_config
                else None
            ),
            min_active_pairs=int(
                lmc_config.get("min_active_pairs", 2)
            ),
            confusion_decay=float(
                lmc_config.get("confusion_decay", 0.90)
            ),
            min_pair_score=float(
                lmc_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                lmc_config.get("min_class_observations", 8)
            ),
            auxiliary_weight=float(
                lmc_config.get("auxiliary_weight", 0.20)
            ),
            pair_ramp_epochs=int(
                lmc_config.get("pair_ramp_epochs", 20)
            ),
            observation_margin_threshold=float(
                lmc_config.get(
                    "observation_margin_threshold",
                    lmc_config.get("margin_threshold", 0.20),
                )
            ),
            low_margin_only=bool(
                lmc_config.get("low_margin_only", True)
            ),
            use_temperature_weights=bool(
                lmc_config.get("use_temperature_weights", False)
            ),
            temperature=float(lmc_config.get("temperature", 0.05)),
            never_freeze=bool(lmc_config.get("never_freeze", False)),
        )
    if soft_cgrh_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return SoftDynamicCGRHClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(
                soft_cgrh_config.get("warmup_epochs", 20)
            ),
            ramp_end_epoch=int(
                soft_cgrh_config.get("ramp_end_epoch", 60)
            ),
            update_interval=int(
                soft_cgrh_config.get("update_interval", 1)
            ),
            confusion_decay=float(
                soft_cgrh_config.get("confusion_decay", 0.90)
            ),
            temperature=float(
                soft_cgrh_config.get("temperature", 0.02)
            ),
            max_pair_share=float(
                soft_cgrh_config.get("max_pair_share", 0.45)
            ),
            max_pair_weight=float(
                soft_cgrh_config.get("max_pair_weight", 0.10)
            ),
            max_refine_weight=float(
                soft_cgrh_config.get("max_refine_weight", 0.05)
            ),
            min_difficulty_scale=float(
                soft_cgrh_config.get("min_difficulty_scale", 0.20)
            ),
            min_class_observations=int(
                soft_cgrh_config.get("min_class_observations", 8)
            ),
        )
    if continuous_top3_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return ContinuousTop3CGRHClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(
                continuous_top3_config.get("warmup_epochs", 20)
            ),
            auxiliary_weight=float(
                continuous_top3_config.get("auxiliary_weight", 0.20)
            ),
            auxiliary_ramp_epochs=int(
                continuous_top3_config.get("auxiliary_ramp_epochs", 40)
            ),
            update_interval=int(
                continuous_top3_config.get("update_interval", 5)
            ),
            active_pair_count=int(
                continuous_top3_config.get("active_pair_count", 3)
            ),
            min_active_pairs=int(
                continuous_top3_config.get("min_active_pairs", 2)
            ),
            confusion_decay=float(
                continuous_top3_config.get("confusion_decay", 0.90)
            ),
            min_pair_score=float(
                continuous_top3_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                continuous_top3_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                continuous_top3_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                continuous_top3_config.get("exit_rank_multiplier", 1)
            ),
            exit_persistence=int(
                continuous_top3_config.get("exit_persistence", 2)
            ),
            replace_margin=float(
                continuous_top3_config.get("replace_margin", 0.0)
            ),
            pair_handoff_epochs=int(
                continuous_top3_config.get("pair_handoff_epochs", 5)
            ),
        )
    if ca_lite_dual_ce_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return CALiteDualCEClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(
                ca_lite_dual_ce_config.get("warmup_epochs", 20)
            ),
            auxiliary_weight=float(
                ca_lite_dual_ce_config.get("auxiliary_weight", 0.20)
            ),
            auxiliary_ramp_epochs=int(
                ca_lite_dual_ce_config.get("auxiliary_ramp_epochs", 40)
            ),
            update_interval=int(
                ca_lite_dual_ce_config.get("update_interval", 5)
            ),
            active_pair_count=int(
                ca_lite_dual_ce_config.get("active_pair_count", 3)
            ),
            min_active_pairs=int(
                ca_lite_dual_ce_config.get("min_active_pairs", 2)
            ),
            confusion_decay=float(
                ca_lite_dual_ce_config.get("confusion_decay", 0.90)
            ),
            min_pair_score=float(
                ca_lite_dual_ce_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                ca_lite_dual_ce_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                ca_lite_dual_ce_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                ca_lite_dual_ce_config.get("exit_rank_multiplier", 1)
            ),
            exit_persistence=int(
                ca_lite_dual_ce_config.get("exit_persistence", 2)
            ),
            replace_margin=float(
                ca_lite_dual_ce_config.get("replace_margin", 0.0)
            ),
            pair_handoff_epochs=int(
                ca_lite_dual_ce_config.get("pair_handoff_epochs", 5)
            ),
        )
    if stable_ca_cgrh_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        config = stable_ca_cgrh_config
        return StableCompetenceAdaptiveTop3CGRHClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(config.get("warmup_epochs", 20)),
            auxiliary_weight=float(config.get("auxiliary_weight", 0.20)),
            auxiliary_ramp_epochs=int(
                config.get("auxiliary_ramp_epochs", 40)
            ),
            update_interval=int(config.get("update_interval", 5)),
            active_pair_count=int(config.get("active_pair_count", 3)),
            min_active_pairs=int(config.get("min_active_pairs", 2)),
            confusion_decay=float(config.get("confusion_decay", 0.90)),
            min_pair_score=float(config.get("min_pair_score", 0.005)),
            min_class_observations=int(
                config.get("min_class_observations", 8)
            ),
            # Base-class streak controls are deliberately disabled here;
            # the sliding vote window is the sole discovery stabilizer.
            entry_persistence=1,
            exit_rank_multiplier=1,
            exit_persistence=1,
            replace_margin=0.0,
            pair_handoff_epochs=int(
                config.get("pair_handoff_epochs", 5)
            ),
            max_residual_scale=float(
                config.get("max_residual_scale", 0.50)
            ),
            main_ce_weight=float(config.get("main_ce_weight", 0.25)),
            refined_ce_weight=float(
                config.get("refined_ce_weight", 0.75)
            ),
            advantage_decay=float(config.get("advantage_decay", 0.95)),
            evidence_decay=float(config.get("evidence_decay", 0.995)),
            inactive_evidence_decay=float(
                config.get("inactive_evidence_decay", 0.90)
            ),
            minimum_effective_samples=float(
                config.get("minimum_effective_samples", 64.0)
            ),
            advantage_margin=float(
                config.get("advantage_margin", 0.02)
            ),
            advantage_temperature=float(
                config.get("advantage_temperature", 0.02)
            ),
            scale_update_rate=float(
                config.get("scale_update_rate", 0.01)
            ),
            vote_window=int(config.get("vote_window", 4)),
            minimum_votes=int(config.get("minimum_votes", 3)),
            replacement_margin_ratio=float(
                config.get("replacement_margin_ratio", 0.10)
            ),
        )
    if competence_top3_enabled or competence_top2_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        competence_config = (
            competence_top2_config
            if competence_top2_enabled
            else competence_top3_config
        )
        return CompetenceAdaptiveTop3CGRHClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(
                competence_config.get("warmup_epochs", 20)
            ),
            auxiliary_weight=float(
                competence_config.get("auxiliary_weight", 0.20)
            ),
            auxiliary_ramp_epochs=int(
                competence_config.get("auxiliary_ramp_epochs", 40)
            ),
            update_interval=int(
                competence_config.get("update_interval", 5)
            ),
            active_pair_count=int(
                competence_config.get("active_pair_count", 3)
            ),
            min_active_pairs=int(
                competence_config.get("min_active_pairs", 2)
            ),
            confusion_decay=float(
                competence_config.get("confusion_decay", 0.90)
            ),
            min_pair_score=float(
                competence_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                competence_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                competence_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                competence_config.get("exit_rank_multiplier", 1)
            ),
            exit_persistence=int(
                competence_config.get("exit_persistence", 2)
            ),
            replace_margin=float(
                competence_config.get("replace_margin", 0.0)
            ),
            pair_handoff_epochs=int(
                competence_config.get("pair_handoff_epochs", 5)
            ),
            max_residual_scale=float(
                competence_config.get("max_residual_scale", 0.50)
            ),
            main_ce_weight=float(
                competence_config.get("main_ce_weight", 0.25)
            ),
            refined_ce_weight=float(
                competence_config.get("refined_ce_weight", 0.75)
            ),
            advantage_decay=float(
                competence_config.get("advantage_decay", 0.95)
            ),
            evidence_decay=float(
                competence_config.get("evidence_decay", 0.995)
            ),
            inactive_evidence_decay=float(
                competence_config.get(
                    "inactive_evidence_decay", 0.90
                )
            ),
            minimum_effective_samples=float(
                competence_config.get(
                    "minimum_effective_samples", 64.0
                )
            ),
            advantage_margin=float(
                competence_config.get("advantage_margin", 0.02)
            ),
            advantage_temperature=float(
                competence_config.get(
                    "advantage_temperature", 0.02
                )
            ),
            scale_update_rate=float(
                competence_config.get("scale_update_rate", 0.01)
            ),
        )
    if competence_top3_cdw_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        config = competence_top3_cdw_config
        return CompetenceAdaptiveTop3CGRHCDWClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(config.get("warmup_epochs", 20)),
            auxiliary_weight=float(config.get("auxiliary_weight", 0.20)),
            auxiliary_ramp_epochs=int(
                config.get("auxiliary_ramp_epochs", 40)
            ),
            update_interval=int(config.get("update_interval", 5)),
            active_pair_count=int(config.get("active_pair_count", 3)),
            min_active_pairs=int(config.get("min_active_pairs", 2)),
            confusion_decay=float(config.get("confusion_decay", 0.90)),
            min_pair_score=float(config.get("min_pair_score", 0.005)),
            min_class_observations=int(
                config.get("min_class_observations", 8)
            ),
            entry_persistence=int(config.get("entry_persistence", 2)),
            exit_rank_multiplier=int(
                config.get("exit_rank_multiplier", 1)
            ),
            exit_persistence=int(config.get("exit_persistence", 2)),
            replace_margin=float(config.get("replace_margin", 0.0)),
            pair_handoff_epochs=int(config.get("pair_handoff_epochs", 5)),
            max_residual_scale=float(
                config.get("max_residual_scale", 0.50)
            ),
            main_ce_weight=float(config.get("main_ce_weight", 0.25)),
            refined_ce_weight=float(
                config.get("refined_ce_weight", 0.75)
            ),
            advantage_decay=float(config.get("advantage_decay", 0.95)),
            evidence_decay=float(config.get("evidence_decay", 0.995)),
            inactive_evidence_decay=float(
                config.get("inactive_evidence_decay", 0.90)
            ),
            minimum_effective_samples=float(
                config.get("minimum_effective_samples", 64.0)
            ),
            advantage_margin=float(config.get("advantage_margin", 0.02)),
            advantage_temperature=float(
                config.get("advantage_temperature", 0.02)
            ),
            scale_update_rate=float(
                config.get("scale_update_rate", 0.01)
            ),
            difficulty_ema_beta=float(
                config.get("difficulty_ema_beta", 0.90)
            ),
            difficulty_warmup_epochs=int(
                config.get("difficulty_warmup_epochs", 5)
            ),
            difficulty_min_weight=float(
                config.get("difficulty_min_weight", 0.80)
            ),
            difficulty_max_weight=float(
                config.get("difficulty_max_weight", 1.20)
            ),
        )
    if competence_top3_v2a_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        config = competence_top3_v2a_config
        return CompetenceAdaptiveTop3CGRHV2AClassificationLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            warmup_epochs=int(config.get("warmup_epochs", 20)),
            auxiliary_weight=float(config.get("auxiliary_weight", 0.20)),
            auxiliary_ramp_epochs=int(
                config.get("auxiliary_ramp_epochs", 40)
            ),
            update_interval=int(config.get("update_interval", 5)),
            active_pair_count=int(config.get("active_pair_count", 3)),
            min_active_pairs=int(config.get("min_active_pairs", 2)),
            confusion_decay=float(config.get("confusion_decay", 0.90)),
            min_pair_score=float(config.get("min_pair_score", 0.005)),
            min_class_observations=int(
                config.get("min_class_observations", 8)
            ),
            entry_persistence=int(config.get("entry_persistence", 2)),
            exit_rank_multiplier=int(
                config.get("exit_rank_multiplier", 1)
            ),
            exit_persistence=int(config.get("exit_persistence", 2)),
            replace_margin=float(config.get("replace_margin", 0.0)),
            pair_handoff_epochs=int(config.get("pair_handoff_epochs", 5)),
            max_residual_scale=float(
                config.get("max_residual_scale", 0.50)
            ),
            main_ce_weight=float(config.get("main_ce_weight", 0.25)),
            refined_ce_weight=float(
                config.get("refined_ce_weight", 0.75)
            ),
            advantage_decay=float(config.get("advantage_decay", 0.95)),
            evidence_decay=float(config.get("evidence_decay", 0.995)),
            inactive_evidence_decay=float(
                config.get("inactive_evidence_decay", 0.90)
            ),
            minimum_effective_samples=float(
                config.get("minimum_effective_samples", 64.0)
            ),
            advantage_margin=float(config.get("advantage_margin", 0.02)),
            advantage_temperature=float(
                config.get("advantage_temperature", 0.02)
            ),
            scale_update_rate=float(
                config.get("scale_update_rate", 0.01)
            ),
            hard_sample_quantile=float(
                config.get("hard_sample_quantile", 0.30)
            ),
            easy_sample_weight_floor=float(
                config.get("easy_sample_weight_floor", 0.10)
            ),
            competence_weight_floor=float(
                config.get("competence_weight_floor", 0.10)
            ),
        )
    if dcmg_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return DynamicConfusionMarginGuidanceLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            pair_weight=float(dcmg_config.get("pair_weight", 0.05)),
            margin=float(dcmg_config.get("margin", 0.10)),
            warmup_epochs=int(dcmg_config.get("warmup_epochs", 20)),
            update_interval=int(dcmg_config.get("update_interval", 5)),
            active_pair_count=int(
                dcmg_config.get("active_pair_count", 3)
            ),
            confusion_decay=float(
                dcmg_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                dcmg_config.get("freeze_pairs_epoch", 150)
            ),
            min_pair_score=float(
                dcmg_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                dcmg_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                dcmg_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                dcmg_config.get("exit_rank_multiplier", 2)
            ),
            score_weighted=bool(
                dcmg_config.get("score_weighted", False)
            ),
        )
    if dcmg_v2_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return ReliableDynamicConfusionMarginGuidanceLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            max_pair_weight=float(
                dcmg_v2_config.get("max_pair_weight", 0.02)
            ),
            margin=float(dcmg_v2_config.get("margin", 0.10)),
            warmup_epochs=int(
                dcmg_v2_config.get("warmup_epochs", 20)
            ),
            update_interval=int(
                dcmg_v2_config.get("update_interval", 10)
            ),
            active_pair_count=int(
                dcmg_v2_config.get("active_pair_count", 3)
            ),
            confusion_decay=float(
                dcmg_v2_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                dcmg_v2_config.get("freeze_pairs_epoch", 150)
            ),
            min_pair_score=float(
                dcmg_v2_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                dcmg_v2_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                dcmg_v2_config.get("entry_persistence", 3)
            ),
            exit_rank_multiplier=int(
                dcmg_v2_config.get("exit_rank_multiplier", 2)
            ),
            ramp_start_epoch=int(
                dcmg_v2_config.get("ramp_start_epoch", 30)
            ),
            ramp_end_epoch=int(
                dcmg_v2_config.get("ramp_end_epoch", 80)
            ),
            min_teacher_true_prob=float(
                dcmg_v2_config.get("min_teacher_true_prob", 0.45)
            ),
            max_teacher_true_prob=float(
                dcmg_v2_config.get("max_teacher_true_prob", 0.95)
            ),
            min_teacher_competitor_prob=float(
                dcmg_v2_config.get(
                    "min_teacher_competitor_prob", 0.05
                )
            ),
            max_teacher_margin=float(
                dcmg_v2_config.get("max_teacher_margin", 0.70)
            ),
            max_main_logit_gap=float(
                dcmg_v2_config.get("max_main_logit_gap", 2.0)
            ),
            loss_cap=float(dcmg_v2_config.get("loss_cap", 1.5)),
        )
    if dcmg_cgr_v3_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return DynamicConfusionGraphPairRoutingLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            max_pair_weight=float(
                dcmg_cgr_v3_config.get("max_pair_weight", 0.10)
            ),
            warmup_epochs=int(
                dcmg_cgr_v3_config.get("warmup_epochs", 20)
            ),
            update_interval=int(
                dcmg_cgr_v3_config.get("update_interval", 5)
            ),
            active_pair_count=int(
                dcmg_cgr_v3_config.get("active_pair_count", 3)
            ),
            stable_pair_count=int(
                dcmg_cgr_v3_config.get("stable_pair_count", 2)
            ),
            candidate_pool_size=int(
                dcmg_cgr_v3_config.get("candidate_pool_size", 6)
            ),
            confusion_decay=float(
                dcmg_cgr_v3_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                dcmg_cgr_v3_config.get("freeze_pairs_epoch", 150)
            ),
            min_auto_freeze_epoch=int(
                dcmg_cgr_v3_config.get("min_auto_freeze_epoch", 120)
            ),
            auto_freeze_patience=int(
                dcmg_cgr_v3_config.get("auto_freeze_patience", 3)
            ),
            min_pair_score=float(
                dcmg_cgr_v3_config.get("min_pair_score", 0.005)
            ),
            min_class_observations=int(
                dcmg_cgr_v3_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                dcmg_cgr_v3_config.get("entry_persistence", 2)
            ),
            exit_rank_multiplier=int(
                dcmg_cgr_v3_config.get("exit_rank_multiplier", 2)
            ),
            probability_margin_threshold=float(
                dcmg_cgr_v3_config.get(
                    "probability_margin_threshold", 0.20
                )
            ),
            minimum_difficulty_weight=float(
                dcmg_cgr_v3_config.get(
                    "minimum_difficulty_weight", 0.25
                )
            ),
            ramp_start_epoch=int(
                dcmg_cgr_v3_config.get("ramp_start_epoch", 25)
            ),
            ramp_end_epoch=int(
                dcmg_cgr_v3_config.get("ramp_end_epoch", 60)
            ),
            loss_cap=float(
                dcmg_cgr_v3_config.get("loss_cap", 2.0)
            ),
        )
    if dcmg_cgr_v4_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        return DirectionalCoverageDynamicConfusionLoss(
            num_classes=int(dataset_config["num_classes"]),
            class_names=class_names,
            label_smoothing=label_smoothing,
            max_pair_weight=float(
                dcmg_cgr_v4_config.get("max_pair_weight", 0.06)
            ),
            warmup_epochs=int(
                dcmg_cgr_v4_config.get("warmup_epochs", 20)
            ),
            update_interval=int(
                dcmg_cgr_v4_config.get("update_interval", 5)
            ),
            stable_pair_count=int(
                dcmg_cgr_v4_config.get("stable_pair_count", 2)
            ),
            candidate_pool_size=int(
                dcmg_cgr_v4_config.get("candidate_pool_size", 6)
            ),
            confusion_decay=float(
                dcmg_cgr_v4_config.get("confusion_decay", 0.90)
            ),
            freeze_pairs_epoch=int(
                dcmg_cgr_v4_config.get("freeze_pairs_epoch", 165)
            ),
            min_auto_freeze_epoch=int(
                dcmg_cgr_v4_config.get("min_auto_freeze_epoch", 145)
            ),
            auto_freeze_patience=int(
                dcmg_cgr_v4_config.get("auto_freeze_patience", 4)
            ),
            freeze_score_tolerance=float(
                dcmg_cgr_v4_config.get("freeze_score_tolerance", 0.05)
            ),
            min_edge_score=float(
                dcmg_cgr_v4_config.get("min_edge_score", 0.005)
            ),
            min_class_observations=int(
                dcmg_cgr_v4_config.get("min_class_observations", 8)
            ),
            entry_persistence=int(
                dcmg_cgr_v4_config.get("entry_persistence", 2)
            ),
            stable_replacement_ratio=float(
                dcmg_cgr_v4_config.get("stable_replacement_ratio", 1.12)
            ),
            exploration_tenure_updates=int(
                dcmg_cgr_v4_config.get("exploration_tenure_updates", 2)
            ),
            base_sample_weight=float(
                dcmg_cgr_v4_config.get("base_sample_weight", 0.15)
            ),
            hard_sample_boost=float(
                dcmg_cgr_v4_config.get("hard_sample_boost", 0.85)
            ),
            probability_margin_threshold=float(
                dcmg_cgr_v4_config.get(
                    "probability_margin_threshold", 0.25
                )
            ),
            ramp_start_epoch=int(
                dcmg_cgr_v4_config.get("ramp_start_epoch", 25)
            ),
            ramp_end_epoch=int(
                dcmg_cgr_v4_config.get("ramp_end_epoch", 65)
            ),
            loss_cap=float(
                dcmg_cgr_v4_config.get("loss_cap", 2.0)
            ),
        )
    if dphg_enabled:
        class_names = [str(name) for name in dataset_config["class_names"]]
        pair_mode = str(dphg_config.get("pair_mode", "all")).lower()
        if pair_mode == "all":
            pair_names = [
                [class_names[first], class_names[second]]
                for first in range(len(class_names))
                for second in range(first + 1, len(class_names))
            ]
        elif pair_mode == "explicit":
            pair_names = list(dphg_config.get("pairs", []))
        else:
            raise ValueError(
                "dphg.pair_mode must be either 'explicit' or 'all'; "
                f"got {pair_mode!r}"
            )
        pairs = _resolve_confusion_pairs(pair_names, class_names)
        return DPHGClassificationLoss(
            label_smoothing=label_smoothing,
            pair_indices=pairs,
            pair_weight=float(dphg_config.get("pair_weight", 0.20)),
            refine_weight=float(dphg_config.get("refine_weight", 0.20)),
            warmup_epochs=int(dphg_config.get("warmup_epochs", 5)),
        )
    if not confusion_enabled:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    config = confusion_config
    pair_names = list(config.get("pairs", []))
    if not pair_names:
        raise ValueError("confusion_loss.enabled=true requires non-empty pairs")
    class_names = [str(name) for name in dataset_config["class_names"]]
    pairs = _resolve_confusion_pairs(pair_names, class_names)
    return ConfusionGuidedClassificationLoss(
        label_smoothing=label_smoothing,
        confusion_pairs=pairs,
        auxiliary_weight=float(config.get("weight", 0.15)),
        max_pair_similarity=float(config.get("max_pair_similarity", 0.20)),
        compactness_weight=float(config.get("compactness_weight", 0.25)),
    )
