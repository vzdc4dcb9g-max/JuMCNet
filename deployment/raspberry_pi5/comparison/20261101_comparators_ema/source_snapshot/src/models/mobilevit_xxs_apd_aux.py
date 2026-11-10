from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class AllPairDiscriminativeAuxiliaryHead(nn.Module):
    """Vectorized, training-only binary supervision for class pairs."""

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
            raise ValueError("APD auxiliary head requires classes and pairs")
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in pair_indices:
            first, second = int(first), int(second)
            key = tuple(sorted((first, second)))
            if first == second or not (
                0 <= first < num_classes and 0 <= second < num_classes
            ):
                raise ValueError(f"Invalid APD pair: {(first, second)}")
            if key in seen:
                raise ValueError(f"Duplicate unordered APD pair: {key}")
            seen.add(key)
            normalized.append((first, second))

        self.num_pairs = len(normalized)
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
        # One vectorized operation replaces V1's 15 separate Linear calls.
        self.classifier = nn.Linear(projection_dim, self.num_pairs * 2)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pair_logits = self.classifier(self.projection(features)).reshape(
            features.shape[0], self.num_pairs, 2
        )
        return {
            "pair_logits": pair_logits,
            "pair_indices": self.pair_indices,
        }


class MobileViTXXSAPDAux(MobileViTXXSDFN):
    """MFF-LocalECA backbone with a removable training-only APD head."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        apd_pair_indices: list[tuple[int, int]] | None = None,
        apd_projection_dim: int = 64,
        apd_projection_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.apd_auxiliary: AllPairDiscriminativeAuxiliaryHead | None = (
            AllPairDiscriminativeAuxiliaryHead(
                in_features=int(self.num_features),
                num_classes=num_classes,
                pair_indices=apd_pair_indices or [],
                projection_dim=int(apd_projection_dim),
                projection_dropout=float(apd_projection_dropout),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSAPDAux:
        """Remove parameters and computation used only during training."""
        self.apd_auxiliary = None
        self.inference_auxiliary_stripped = True
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_apd_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("APD-Aux is not combined with global auxiliary head")
        if return_features and return_apd_aux:
            raise ValueError(
                "return_features and return_apd_aux are mutually exclusive"
            )
        features = self.forward_features(inputs)
        logits = self.classifier(self.dropout(features))
        if return_apd_aux:
            if self.apd_auxiliary is None:
                raise RuntimeError("APD auxiliary head was stripped for inference")
            return logits, self.apd_auxiliary(features)
        if return_features:
            return logits, features
        # Validation, test, export and benchmark never run the auxiliary head.
        return logits


def mobilevit_xxs_apd_aux(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSAPDAux:
    return MobileViTXXSAPDAux(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
