from __future__ import annotations

from typing import Any, Iterable

import torch

from src.models.mobilevit_xxs_auto_cgrh import MobileViTXXSAutoCGRH


class MobileViTXXSLMC(MobileViTXXSAutoCGRH):
    """LMC-Net: LocalECA-MFF + deployable CGRH experts with hard-pair stats.

    Architecture matches Auto-CGRH (all-pair experts, low-margin Top-2 routing,
    fixed refine_scale). Pair discovery lives in :class:`LMCClassificationLoss`;
    this module keeps an explicit, queryable count of active / eligible hard
    pairs for logging and analysis.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Number of currently activated refine pairs (mask sum).
        self.register_buffer(
            "lmc_hard_pair_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        # Qualifying undirected pairs from last discovery (score >= threshold),
        # before Top-K truncation. Updated via update_hard_pair_counts().
        self.register_buffer(
            "lmc_eligible_hard_pair_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        # Core Top-K size from last discovery publish.
        self.register_buffer(
            "lmc_core_hard_pair_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "lmc_hard_pair_update_steps",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.last_hard_pair_route_stats: dict[str, Any] = {}

    @property
    def max_undirected_pair_count(self) -> int:
        return int(self.cgrh.pair_indices.shape[0])

    def hard_pair_count(self) -> int:
        """Number of currently active hard pairs on the model."""
        return int(self.auto_cgrh_active_mask.sum().item())

    def eligible_hard_pair_count(self) -> int:
        """Last discovery's qualifying pair count (pre Top-K)."""
        return int(self.lmc_eligible_hard_pair_count.item())

    def core_hard_pair_count(self) -> int:
        return int(self.lmc_core_hard_pair_count.item())

    def set_auto_cgrh_pairs(
        self,
        pairs: Iterable[tuple[int, int]],
    ) -> None:
        super().set_auto_cgrh_pairs(pairs)
        count = int(self.auto_cgrh_active_mask.sum().item())
        self.lmc_hard_pair_count.fill_(count)
        self.lmc_hard_pair_update_steps.add_(1)
        # If discovery metadata was not provided, keep eligible/core >= active.
        if int(self.lmc_eligible_hard_pair_count.item()) < count:
            self.lmc_eligible_hard_pair_count.fill_(count)
        if int(self.lmc_core_hard_pair_count.item()) <= 0:
            self.lmc_core_hard_pair_count.fill_(count)

    def update_hard_pair_counts(
        self,
        *,
        active_count: int | None = None,
        core_count: int | None = None,
        eligible_count: int | None = None,
    ) -> None:
        """Sync discovery counters from the LMC criterion / trainer."""
        if active_count is not None:
            self.lmc_hard_pair_count.fill_(max(0, int(active_count)))
        if core_count is not None:
            self.lmc_core_hard_pair_count.fill_(max(0, int(core_count)))
        if eligible_count is not None:
            self.lmc_eligible_hard_pair_count.fill_(max(0, int(eligible_count)))

    @staticmethod
    def count_qualifying_hard_pairs(
        score_matrix: torch.Tensor,
        *,
        min_pair_score: float = 0.005,
        min_class_observations: int = 8,
        class_counts: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Count undirected pairs that qualify as hard under LMC score rules.

        ``score_matrix`` should be a symmetric ``[C, C]`` confusion / pair-score
        matrix (diagonal ignored). Returns the qualifying count and the sorted
        pair list ``(score, i, j)``.
        """
        if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
            raise ValueError("score_matrix must be square [C, C]")
        num_classes = int(score_matrix.shape[0])
        if class_counts is None:
            class_counts = torch.full(
                (num_classes,),
                fill_value=max(int(min_class_observations), 0),
                dtype=torch.long,
                device=score_matrix.device,
            )
        scored: list[tuple[float, int, int]] = []
        for first in range(num_classes):
            if int(class_counts[first].item()) < min_class_observations:
                continue
            for second in range(first + 1, num_classes):
                if int(class_counts[second].item()) < min_class_observations:
                    continue
                score = float(score_matrix[first, second].item())
                if score >= float(min_pair_score):
                    scored.append((score, first, second))
        scored.sort(key=lambda item: item[0], reverse=True)
        return {
            "eligible_hard_pair_count": len(scored),
            "max_undirected_pairs": num_classes * (num_classes - 1) // 2,
            "scored_pairs": scored,
        }

    def hard_pair_stats(
        self,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Snapshot of hard-pair counts for logging / Excel / debugging."""
        pairs = self.auto_cgrh_pairs()
        named_pairs: list[str] = []
        if class_names is not None:
            if len(class_names) != int(self.num_classes):
                raise ValueError("class_names must match num_classes")
            named_pairs = [
                f"{class_names[first]}~{class_names[second]}"
                for first, second in pairs
            ]
        return {
            "active_hard_pair_count": self.hard_pair_count(),
            "core_hard_pair_count": self.core_hard_pair_count(),
            "eligible_hard_pair_count": self.eligible_hard_pair_count(),
            "max_undirected_pair_count": self.max_undirected_pair_count,
            "hard_pair_update_steps": int(self.lmc_hard_pair_update_steps.item()),
            "active_hard_pairs": pairs,
            "active_hard_pair_names": named_pairs,
            "last_route_stats": dict(self.last_hard_pair_route_stats),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_cgrh: bool = False,
        return_main_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        result = super().forward(
            inputs,
            return_features=return_features,
            return_aux=return_aux,
            return_cgrh=return_cgrh,
            return_main_logits=return_main_logits,
        )
        if not return_cgrh:
            return result

        refined_logits, state = result
        active_pairs = self.auto_cgrh_pairs()
        route_mask = state["route_mask"]
        pair_indices = state["pair_indices"]
        pair_to_slot = {
            tuple(int(value) for value in pair): slot
            for slot, pair in enumerate(pair_indices.tolist())
        }
        per_pair_hits: list[dict[str, Any]] = []
        total_routed = 0
        for first, second in active_pairs:
            key = tuple(sorted((first, second)))
            slot = pair_to_slot[key]
            hits = int(route_mask[:, slot].sum().item())
            total_routed += hits
            per_pair_hits.append(
                {
                    "pair": key,
                    "slot": slot,
                    "route_hits": hits,
                }
            )
        batch_size = int(route_mask.shape[0])
        stats = {
            "hard_pair_count": self.hard_pair_count(),
            "eligible_hard_pair_count": self.eligible_hard_pair_count(),
            "core_hard_pair_count": self.core_hard_pair_count(),
            "routed_sample_hits": total_routed,
            "routed_sample_rate": (
                float(total_routed) / float(batch_size) if batch_size > 0 else 0.0
            ),
            "per_pair_route_hits": per_pair_hits,
        }
        self.last_hard_pair_route_stats = stats
        # Keep tensor-friendly scalars in the CGRH state for trainers.
        state["hard_pair_count"] = refined_logits.new_tensor(
            float(stats["hard_pair_count"])
        )
        state["eligible_hard_pair_count"] = refined_logits.new_tensor(
            float(stats["eligible_hard_pair_count"])
        )
        state["core_hard_pair_count"] = refined_logits.new_tensor(
            float(stats["core_hard_pair_count"])
        )
        return refined_logits, state


def mobilevit_xxs_lmc(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSLMC:
    # Auto-CGRH kwargs use the auto_cgrh_* prefix; accept either naming.
    remapped = dict(kwargs)
    alias_map = {
        "lmc_projection_dim": "auto_cgrh_projection_dim",
        "lmc_projection_dropout": "auto_cgrh_projection_dropout",
        "lmc_margin_threshold": "auto_cgrh_margin_threshold",
        "lmc_refine_scale": "auto_cgrh_refine_scale",
    }
    for source, target in alias_map.items():
        if source in remapped and target not in remapped:
            remapped[target] = remapped.pop(source)
        elif source in remapped:
            remapped.pop(source)
    return MobileViTXXSLMC(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **remapped,
    )
