from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class AdaptiveOnlineConfusionAuxiliaryHead(nn.Module):
    """Shared training-only classifier for dynamically selected hard pairs.

    The head predicts one auxiliary logit per dataset class.  The loss selects
    only the two columns belonging to each currently active hard pair.  This
    keeps parameter growth linear in the number of classes and avoids one
    separately parameterized expert for every possible class pair.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        projection_dim: int = 64,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or projection_dim < 1:
            raise ValueError("in_features and projection_dim must be >= 1")
        if num_classes < 2:
            raise ValueError("AOCG-Aux requires at least two classes")

        self.projection = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, projection_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"aux_logits": self.classifier(self.projection(features))}


class MobileViTXXSAOCGAux(MobileViTXXSDFN):
    """MFF-LocalECA backbone with removable AOCG pair supervision."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        aocg_projection_dim: int = 64,
        aocg_projection_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.aocg_auxiliary: AdaptiveOnlineConfusionAuxiliaryHead | None = (
            AdaptiveOnlineConfusionAuxiliaryHead(
                in_features=int(self.num_features),
                num_classes=int(num_classes),
                projection_dim=int(aocg_projection_dim),
                projection_dropout=float(aocg_projection_dropout),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSAOCGAux:
        """Remove the complete AOCG branch for test/export/deployment."""
        self.aocg_auxiliary = None
        self.inference_auxiliary_stripped = True
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_aocg_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("AOCG-Aux is not combined with global auxiliary head")
        if return_features and return_aocg_aux:
            raise ValueError(
                "return_features and return_aocg_aux are mutually exclusive"
            )

        features = self.forward_features(inputs)
        logits = self.classifier(self.dropout(features))
        if return_aocg_aux:
            if self.aocg_auxiliary is None:
                raise RuntimeError("AOCG auxiliary head was stripped for inference")
            return logits, self.aocg_auxiliary(features)
        if return_features:
            return logits, features
        return logits


def mobilevit_xxs_aocg_aux(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSAOCGAux:
    return MobileViTXXSAOCGAux(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
