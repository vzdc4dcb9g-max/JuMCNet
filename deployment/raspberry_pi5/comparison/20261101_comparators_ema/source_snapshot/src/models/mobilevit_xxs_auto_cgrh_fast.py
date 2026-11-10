from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class SparseConditionalCGRHHead(nn.Module):
    """A small fixed expert bank with inference-time conditional execution."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        max_pairs: int = 3,
        projection_dim: int = 64,
        projection_dropout: float = 0.0,
        margin_threshold: float = 0.20,
        refine_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if in_features < 1 or projection_dim < 1:
            raise ValueError("in_features and projection_dim must be >= 1")
        if num_classes < 2 or max_pairs < 1:
            raise ValueError("num_classes >= 2 and max_pairs >= 1 are required")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if refine_scale < 0.0:
            raise ValueError("refine_scale must be >= 0")

        self.num_classes = int(num_classes)
        self.max_pairs = int(max_pairs)
        self.margin_threshold = float(margin_threshold)
        self.refine_scale = float(refine_scale)
        self.register_buffer(
            "pair_indices",
            torch.full((self.max_pairs, 2), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "active_pair_mask",
            torch.zeros(self.max_pairs, dtype=torch.bool),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(int(in_features)),
            nn.Linear(int(in_features), int(projection_dim), bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        self.pair_heads = nn.ModuleList(
            nn.Linear(int(projection_dim), 2, bias=True)
            for _ in range(self.max_pairs)
        )
        for head in self.pair_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def set_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in pairs:
            key = tuple(sorted((int(first), int(second))))
            if key[0] == key[1]:
                raise ValueError(f"Auto-CGRH-Fast pair is invalid: {key}")
            if key[0] < 0 or key[1] >= self.num_classes:
                raise ValueError(f"Auto-CGRH-Fast pair is out of range: {key}")
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        if len(normalized) > self.max_pairs:
            raise ValueError(
                f"Received {len(normalized)} pairs, max_pairs={self.max_pairs}"
            )

        self.pair_indices.fill_(-1)
        self.active_pair_mask.zero_()
        for slot, pair in enumerate(normalized):
            self.pair_indices[slot].copy_(
                torch.tensor(
                    pair,
                    dtype=self.pair_indices.dtype,
                    device=self.pair_indices.device,
                )
            )
            self.active_pair_mask[slot] = True

    def active_pairs(self) -> list[tuple[int, int]]:
        return [
            (int(first), int(second))
            for first, second in self.pair_indices[
                self.active_pair_mask
            ].detach().cpu().tolist()
        ]

    def _routing_mask(self, main_logits: torch.Tensor) -> torch.Tensor:
        route_mask = torch.zeros(
            (main_logits.shape[0], self.max_pairs),
            dtype=torch.bool,
            device=main_logits.device,
        )
        if not bool(self.active_pair_mask.any()):
            return route_mask

        with torch.no_grad():
            probabilities = torch.softmax(main_logits.float(), dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            low_margin = (
                top_probabilities[:, 0] - top_probabilities[:, 1]
                <= self.margin_threshold
            )
            first_top = top_indices[:, 0]
            second_top = top_indices[:, 1]
            for slot in self.active_pair_mask.nonzero(
                as_tuple=False
            ).flatten().tolist():
                first, second = self.pair_indices[slot].tolist()
                route_mask[:, slot] = low_margin & (
                    ((first_top == first) & (second_top == second))
                    | ((first_top == second) & (second_top == first))
                )
        return route_mask

    def _empty_state(
        self,
        main_logits: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "main_logits": main_logits,
            "pair_logits": main_logits.new_zeros(
                (main_logits.shape[0], self.max_pairs, 2)
            ),
            "pair_indices": self.pair_indices,
            "active_pair_mask": self.active_pair_mask,
            "route_mask": route_mask,
            "correction": torch.zeros_like(main_logits),
        }

    def forward(
        self,
        features: torch.Tensor,
        main_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        route_mask = self._routing_mask(main_logits)
        active_slots = self.active_pair_mask.nonzero(
            as_tuple=False
        ).flatten().tolist()
        if not active_slots:
            return main_logits, self._empty_state(main_logits, route_mask)

        # Training uses every selected expert for full-pair supervision.
        # Evaluation skips the complete expert branch when no sample routes.
        if not self.training and not bool(route_mask.any()):
            return main_logits, self._empty_state(main_logits, route_mask)

        projected = self.projection(features)
        pair_logits_by_slot: list[torch.Tensor] = []
        zero_logits = main_logits.new_zeros((main_logits.shape[0], 2))
        for slot, head in enumerate(self.pair_heads):
            pair_logits_by_slot.append(
                head(projected) if slot in active_slots else zero_logits
            )
        pair_logits = torch.stack(pair_logits_by_slot, dim=1)

        correction = torch.zeros_like(main_logits)
        for slot in active_slots:
            first, second = self.pair_indices[slot].tolist()
            centered = pair_logits[:, slot]
            centered = centered - centered.mean(dim=1, keepdim=True)
            routed = route_mask[:, slot].to(main_logits.dtype).unsqueeze(1)
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
            "active_pair_mask": self.active_pair_mask,
            "route_mask": route_mask,
            "correction": correction,
        }
        return refined_logits, state


class MobileViTXXSAutoCGRHFast(MobileViTXXSDFN):
    """Auto-CGRH with a pruned expert bank and conditional inference."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        auto_cgrh_max_pairs: int = 3,
        auto_cgrh_projection_dim: int = 64,
        auto_cgrh_projection_dropout: float = 0.0,
        auto_cgrh_margin_threshold: float = 0.20,
        auto_cgrh_refine_scale: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.cgrh = SparseConditionalCGRHHead(
            in_features=int(self.num_features),
            num_classes=int(num_classes),
            max_pairs=int(auto_cgrh_max_pairs),
            projection_dim=int(auto_cgrh_projection_dim),
            projection_dropout=float(auto_cgrh_projection_dropout),
            margin_threshold=float(auto_cgrh_margin_threshold),
            refine_scale=float(auto_cgrh_refine_scale),
        )

    def set_auto_cgrh_pairs(
        self,
        pairs: Iterable[tuple[int, int]],
    ) -> None:
        self.cgrh.set_pairs(pairs)

    def auto_cgrh_pairs(self) -> list[tuple[int, int]]:
        return self.cgrh.active_pairs()

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
        if return_aux:
            raise ValueError(
                "Auto-CGRH-Fast is not combined with a global aux head"
            )
        exclusive = sum(
            int(flag)
            for flag in (return_features, return_cgrh, return_main_logits)
        )
        if exclusive > 1:
            raise ValueError(
                "return_features, return_cgrh, and return_main_logits "
                "are mutually exclusive"
            )

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        if return_main_logits:
            return main_logits
        refined_logits, state = self.cgrh(features, main_logits)
        if return_cgrh:
            return refined_logits, state
        if return_features:
            return refined_logits, features
        return refined_logits


def mobilevit_xxs_auto_cgrh_fast(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSAutoCGRHFast:
    return MobileViTXXSAutoCGRHFast(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
