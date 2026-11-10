from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class OnlineDynamicPairCGRHTrainingHead(nn.Module):
    """All-pair CGRH experts used only while training.

    Every unordered class pair owns one small binary expert.  The loss chooses
    a stable online-mined Top-K subset at epoch boundaries and passes that
    subset to this head.  Active experts can refine low-margin main logits
    during training, while the default forward path always returns the main
    classifier logits.  Consequently validation selects the deployable main
    classifier and test/export can remove this complete module.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        projection_dim: int = 64,
        projection_dropout: float = 0.0,
        margin_threshold: float = 0.20,
        refine_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if in_features < 1 or projection_dim < 1:
            raise ValueError("in_features and projection_dim must be >= 1")
        if num_classes < 2:
            raise ValueError("ODPM-CGRH requires at least two classes")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if refine_scale < 0.0:
            raise ValueError("refine_scale must be >= 0")

        all_pairs = list(combinations(range(int(num_classes)), 2))
        self.num_classes = int(num_classes)
        self.margin_threshold = float(margin_threshold)
        self.refine_scale = float(refine_scale)
        self.register_buffer(
            "pair_indices",
            torch.tensor(all_pairs, dtype=torch.long),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(int(in_features)),
            nn.Linear(int(in_features), int(projection_dim), bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        self.pair_heads = nn.ModuleList(
            nn.Linear(int(projection_dim), 2, bias=True) for _ in all_pairs
        )
        # Adding the training head must initially preserve the baseline logits.
        for head in self.pair_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self._pair_to_slot = {
            tuple(pair): slot for slot, pair in enumerate(all_pairs)
        }

    def _active_slots(
        self,
        active_pairs: Iterable[tuple[int, int]],
    ) -> list[tuple[int, int, int]]:
        slots: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in active_pairs:
            key = tuple(sorted((int(first), int(second))))
            if key in seen:
                continue
            if key not in self._pair_to_slot:
                raise ValueError(f"ODPM active pair is invalid: {key}")
            seen.add(key)
            slots.append((self._pair_to_slot[key], key[0], key[1]))
        return slots

    def forward(
        self,
        features: torch.Tensor,
        main_logits: torch.Tensor,
        active_pairs: Iterable[tuple[int, int]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = self.projection(features)
        pair_logits = torch.stack(
            [head(projected) for head in self.pair_heads], dim=1
        )
        active_slots = self._active_slots(active_pairs)
        active_mask = torch.zeros(
            len(self.pair_heads),
            dtype=torch.bool,
            device=main_logits.device,
        )
        route_mask = torch.zeros(
            (main_logits.shape[0], len(self.pair_heads)),
            dtype=torch.bool,
            device=main_logits.device,
        )
        correction = torch.zeros_like(main_logits)

        if active_slots:
            with torch.no_grad():
                probabilities = torch.softmax(main_logits.float(), dim=1)
                top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
                low_margin = (
                    top_probabilities[:, 0] - top_probabilities[:, 1]
                    <= self.margin_threshold
                )
                first_top = top_indices[:, 0]
                second_top = top_indices[:, 1]

            for slot, first, second in active_slots:
                active_mask[slot] = True
                pair_route = low_margin & (
                    ((first_top == first) & (second_top == second))
                    | ((first_top == second) & (second_top == first))
                )
                route_mask[:, slot] = pair_route

                centered = pair_logits[:, slot]
                centered = centered - centered.mean(dim=1, keepdim=True)
                routed = pair_route.to(main_logits.dtype).unsqueeze(1)
                first_basis = F.one_hot(
                    torch.full(
                        (main_logits.shape[0],),
                        first,
                        device=main_logits.device,
                        dtype=torch.long,
                    ),
                    num_classes=self.num_classes,
                ).to(main_logits.dtype)
                second_basis = F.one_hot(
                    torch.full(
                        (main_logits.shape[0],),
                        second,
                        device=main_logits.device,
                        dtype=torch.long,
                    ),
                    num_classes=self.num_classes,
                ).to(main_logits.dtype)
                correction = correction + self.refine_scale * routed * (
                    centered[:, 0:1] * first_basis
                    + centered[:, 1:2] * second_basis
                )

        refined_logits = main_logits + correction
        state = {
            "main_logits": main_logits,
            "pair_logits": pair_logits,
            "pair_indices": self.pair_indices,
            "active_pair_mask": active_mask,
            "route_mask": route_mask,
            "correction": correction,
        }
        return refined_logits, state


class MobileViTXXSODPMCGRHTrainOnly(MobileViTXXSDFN):
    """MFF-LocalECA backbone with removable online-mined CGRH experts."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        odpm_projection_dim: int = 64,
        odpm_projection_dropout: float = 0.0,
        odpm_margin_threshold: float = 0.20,
        odpm_refine_scale: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.odpm_cgrh_training_head: OnlineDynamicPairCGRHTrainingHead | None = (
            OnlineDynamicPairCGRHTrainingHead(
                in_features=int(self.num_features),
                num_classes=int(num_classes),
                projection_dim=int(odpm_projection_dim),
                projection_dropout=float(odpm_projection_dropout),
                margin_threshold=float(odpm_margin_threshold),
                refine_scale=float(odpm_refine_scale),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSODPMCGRHTrainOnly:
        """Remove the projection and every pair expert before deployment."""
        self.odpm_cgrh_training_head = None
        self.inference_auxiliary_stripped = True
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_odpm_cgrh: bool = False,
        odpm_active_pairs: Iterable[tuple[int, int]] = (),
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("ODPM-CGRH is not combined with a global aux head")
        if return_features and return_odpm_cgrh:
            raise ValueError(
                "return_features and return_odpm_cgrh are mutually exclusive"
            )

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        if return_odpm_cgrh:
            if self.odpm_cgrh_training_head is None:
                raise RuntimeError("ODPM-CGRH training experts were stripped")
            return self.odpm_cgrh_training_head(
                features,
                main_logits,
                odpm_active_pairs,
            )
        if return_features:
            return main_logits, features
        # Validation/test/deployment intentionally use the main head only.
        return main_logits


def mobilevit_xxs_odpm_cgrh_trainonly(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSODPMCGRHTrainOnly:
    return MobileViTXXSODPMCGRHTrainOnly(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
