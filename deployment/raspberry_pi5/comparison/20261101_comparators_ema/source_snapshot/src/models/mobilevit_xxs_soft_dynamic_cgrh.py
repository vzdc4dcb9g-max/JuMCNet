from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class SoftDynamicCGRHTrainingHead(nn.Module):
    """All-pair experts with continuously weighted training-time refinement."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        projection_dim: int = 64,
        projection_dropout: float = 0.0,
        margin_threshold: float = 0.20,
        refine_scale: float = 0.50,
        min_route_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Soft Dynamic CGRH requires at least two classes")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if refine_scale < 0.0 or min_route_weight < 0.0:
            raise ValueError("refine_scale/min_route_weight must be >= 0")

        pairs = list(combinations(range(int(num_classes)), 2))
        self.num_classes = int(num_classes)
        self.margin_threshold = float(margin_threshold)
        self.refine_scale = float(refine_scale)
        self.min_route_weight = float(min_route_weight)
        self.register_buffer(
            "pair_indices",
            torch.tensor(pairs, dtype=torch.long),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(int(in_features)),
            nn.Linear(int(in_features), int(projection_dim), bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        self.pair_heads = nn.ModuleList(
            nn.Linear(int(projection_dim), 2, bias=True) for _ in pairs
        )
        for head in self.pair_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        features: torch.Tensor,
        main_logits: torch.Tensor,
        pair_weights: torch.Tensor,
        residual_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = self.projection(features)
        pair_logits = torch.stack(
            [head(projected) for head in self.pair_heads],
            dim=1,
        )
        weights = pair_weights.to(
            device=main_logits.device,
            dtype=main_logits.dtype,
        )
        if weights.numel() != len(self.pair_heads):
            raise ValueError("Soft Dynamic CGRH pair-weight size mismatch")
        scales = (
            residual_scales.to(
                device=main_logits.device,
                dtype=main_logits.dtype,
            )
            if residual_scales is not None
            else self.refine_scale * weights
        )
        if scales.numel() != len(self.pair_heads):
            raise ValueError("Soft Dynamic CGRH residual-scale size mismatch")

        with torch.no_grad():
            probabilities = torch.softmax(main_logits.float(), dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            low_margin = (
                top_probabilities[:, 0] - top_probabilities[:, 1]
                <= self.margin_threshold
            )
            sorted_top = top_indices.sort(dim=1).values
            pair_matches = (
                (sorted_top[:, None, 0] == self.pair_indices[None, :, 0])
                & (sorted_top[:, None, 1] == self.pair_indices[None, :, 1])
            )
            route_mask = (
                low_margin[:, None]
                & pair_matches
                & (weights[None, :] >= self.min_route_weight)
            )

        correction = torch.zeros_like(main_logits)
        for slot, (first, second) in enumerate(self.pair_indices.tolist()):
            routed = route_mask[:, slot].to(main_logits.dtype).unsqueeze(1)
            if not bool(route_mask[:, slot].any()):
                continue
            centered = pair_logits[:, slot]
            centered = centered - centered.mean(dim=1, keepdim=True)
            first_basis = F.one_hot(
                torch.full(
                    (main_logits.shape[0],),
                    int(first),
                    device=main_logits.device,
                    dtype=torch.long,
                ),
                num_classes=self.num_classes,
            ).to(main_logits.dtype)
            second_basis = F.one_hot(
                torch.full(
                    (main_logits.shape[0],),
                    int(second),
                    device=main_logits.device,
                    dtype=torch.long,
                ),
                num_classes=self.num_classes,
            ).to(main_logits.dtype)
            correction = correction + (
                scales[slot]
                * routed
                * (
                    centered[:, 0:1] * first_basis
                    + centered[:, 1:2] * second_basis
                )
            )

        refined_logits = main_logits + correction
        return refined_logits, {
            "main_logits": main_logits,
            "pair_logits": pair_logits,
            "pair_indices": self.pair_indices,
            "pair_weights": weights,
            "residual_scales": scales,
            "route_mask": route_mask,
            "correction": correction,
        }


class MobileViTXXSSoftDynamicCGRH(MobileViTXXSDFN):
    """MFF-LocalECA backbone with removable soft-dynamic pair experts."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        soft_cgrh_projection_dim: int = 64,
        soft_cgrh_projection_dropout: float = 0.0,
        soft_cgrh_margin_threshold: float = 0.20,
        soft_cgrh_refine_scale: float = 0.50,
        soft_cgrh_min_route_weight: float = 0.01,
        soft_cgrh_separate_residual_scales: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.soft_cgrh_training_head: SoftDynamicCGRHTrainingHead | None = (
            SoftDynamicCGRHTrainingHead(
                in_features=int(self.num_features),
                num_classes=int(num_classes),
                projection_dim=int(soft_cgrh_projection_dim),
                projection_dropout=float(soft_cgrh_projection_dropout),
                margin_threshold=float(soft_cgrh_margin_threshold),
                refine_scale=float(soft_cgrh_refine_scale),
                min_route_weight=float(soft_cgrh_min_route_weight),
            )
        )
        pair_count = int(num_classes) * (int(num_classes) - 1) // 2
        self.register_buffer(
            "soft_cgrh_pair_weights",
            torch.zeros(pair_count),
            persistent=True,
        )
        self.register_buffer(
            "soft_cgrh_residual_scales",
            torch.zeros(pair_count),
            # Runtime controller state lives in the criterion checkpoint.
            # Non-persistent keeps older Swap checkpoints load-compatible.
            persistent=False,
        )
        self.soft_cgrh_separate_residual_scales = bool(
            soft_cgrh_separate_residual_scales
        )
        self.inference_auxiliary_stripped = False

    def set_soft_cgrh_pair_weights(self, weights: torch.Tensor) -> None:
        flattened = weights.detach().to(
            device=self.soft_cgrh_pair_weights.device,
            dtype=self.soft_cgrh_pair_weights.dtype,
        ).flatten()
        if flattened.shape != self.soft_cgrh_pair_weights.shape:
            raise ValueError("Soft Dynamic CGRH pair-weight shape mismatch")
        self.soft_cgrh_pair_weights.copy_(flattened)

    def set_soft_cgrh_residual_scales(self, scales: torch.Tensor) -> None:
        flattened = scales.detach().to(
            device=self.soft_cgrh_residual_scales.device,
            dtype=self.soft_cgrh_residual_scales.dtype,
        ).flatten()
        if flattened.shape != self.soft_cgrh_residual_scales.shape:
            raise ValueError("Soft Dynamic CGRH residual-scale shape mismatch")
        self.soft_cgrh_residual_scales.copy_(flattened)

    def prepare_for_inference(self) -> MobileViTXXSSoftDynamicCGRH:
        self.soft_cgrh_training_head = None
        self.inference_auxiliary_stripped = True
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_soft_cgrh: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("Soft Dynamic CGRH has no global auxiliary head")
        if return_features and return_soft_cgrh:
            raise ValueError(
                "return_features and return_soft_cgrh are mutually exclusive"
            )

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        if return_soft_cgrh:
            if self.soft_cgrh_training_head is None:
                raise RuntimeError("Soft Dynamic CGRH experts were stripped")
            return self.soft_cgrh_training_head(
                features,
                main_logits,
                self.soft_cgrh_pair_weights,
                (
                    self.soft_cgrh_residual_scales
                    if self.soft_cgrh_separate_residual_scales
                    else None
                ),
            )
        if return_features:
            return main_logits, features
        return main_logits


def mobilevit_xxs_soft_dynamic_cgrh(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSSoftDynamicCGRH:
    return MobileViTXXSSoftDynamicCGRH(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
