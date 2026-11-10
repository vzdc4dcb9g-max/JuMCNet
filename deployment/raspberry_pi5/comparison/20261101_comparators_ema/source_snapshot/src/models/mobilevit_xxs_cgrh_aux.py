from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class FixedHardPairAuxiliaryHead(nn.Module):
    """Training-only binary experts for a small set of fixed hard pairs."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pair_indices: list[tuple[int, int]],
        projection_dim: int = 64,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or projection_dim < 1:
            raise ValueError("in_features and projection_dim must be >= 1")
        if num_classes < 2 or not pair_indices:
            raise ValueError("CGRH-Aux requires classes and fixed hard pairs")

        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in pair_indices:
            first, second = int(first), int(second)
            key = tuple(sorted((first, second)))
            if first == second or not (
                0 <= first < num_classes and 0 <= second < num_classes
            ):
                raise ValueError(f"Invalid CGRH-Aux pair: {(first, second)}")
            if key in seen:
                raise ValueError(f"Duplicate CGRH-Aux pair: {key}")
            seen.add(key)
            normalized.append((first, second))

        self.register_buffer(
            "pair_indices",
            torch.tensor(normalized, dtype=torch.long),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, projection_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        self.pair_heads = nn.ModuleList(
            nn.Linear(projection_dim, 2) for _ in normalized
        )
        for head in self.pair_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.projection(features)
        pair_logits = torch.stack(
            [head(projected) for head in self.pair_heads], dim=1
        )
        return {
            "pair_logits": pair_logits,
            "pair_indices": self.pair_indices,
        }


class MobileViTXXSCGRHAux(MobileViTXXSDFN):
    """MFF-LocalECA with fixed hard-pair training and main-head inference."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        cgrh_aux_pair_indices: list[tuple[int, int]] | None = None,
        cgrh_aux_projection_dim: int = 64,
        cgrh_aux_projection_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.cgrh_auxiliary: FixedHardPairAuxiliaryHead | None = (
            FixedHardPairAuxiliaryHead(
                in_features=int(self.num_features),
                num_classes=int(num_classes),
                pair_indices=cgrh_aux_pair_indices or [],
                projection_dim=int(cgrh_aux_projection_dim),
                projection_dropout=float(cgrh_aux_projection_dropout),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSCGRHAux:
        """Remove all three experts and their shared projection."""
        self.cgrh_auxiliary = None
        self.inference_auxiliary_stripped = True
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_cgrh_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("CGRH-Aux is not combined with global auxiliary head")
        if return_features and return_cgrh_aux:
            raise ValueError(
                "return_features and return_cgrh_aux are mutually exclusive"
            )

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        if return_cgrh_aux:
            if self.cgrh_auxiliary is None:
                raise RuntimeError("CGRH auxiliary experts were stripped")
            return main_logits, self.cgrh_auxiliary(features)
        if return_features:
            return main_logits, features
        return main_logits


def mobilevit_xxs_cgrh_aux(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSCGRHAux:
    return MobileViTXXSCGRHAux(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
