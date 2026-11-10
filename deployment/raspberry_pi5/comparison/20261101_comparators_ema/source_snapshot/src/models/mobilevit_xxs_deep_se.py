from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SqueezeExcitation(nn.Module):
    """Classic channel squeeze-and-excitation (Hu et al.)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.fc(self.pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * weights


class ResidualSE(nn.Module):
    """SE residual with zero-init scale so training starts as identity."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction=reduction)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.se(x) - x)


class MobileViTXXSDeepSE(nn.Module):
    """MobileViT-XXS with deep residual SE on late stages and final_conv.

    The timm MobileViT-XXS backbone is left intact. SE is applied only after
    stages 2/3/4 and final_conv (semantic / high-channel maps), not on shallow
    edge stages.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        se_reduction: int = 8,
        use_stage2_se: bool = True,
        use_stage3_se: bool = True,
        use_stage4_se: bool = True,
        use_final_se: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-DeepSE requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        feature_dim = int(self.backbone.num_features)

        self.se_stage2 = (
            ResidualSE(48, reduction=se_reduction)
            if use_stage2_se
            else nn.Identity()
        )
        self.se_stage3 = (
            ResidualSE(64, reduction=se_reduction)
            if use_stage3_se
            else nn.Identity()
        )
        self.se_stage4 = (
            ResidualSE(80, reduction=se_reduction)
            if use_stage4_se
            else nn.Identity()
        )
        self.se_final = (
            ResidualSE(feature_dim, reduction=se_reduction)
            if use_final_se
            else nn.Identity()
        )

        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.num_features = feature_dim

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        features = self.backbone.stages[1](features)
        features = self.backbone.stages[2](features)
        features = self.se_stage2(features)
        features = self.backbone.stages[3](features)
        features = self.se_stage3(features)
        features = self.backbone.stages[4](features)
        features = self.se_stage4(features)
        features = self.backbone.final_conv(features)
        features = self.se_final(features)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


def mobilevit_xxs_deep_se(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSDeepSE:
    return MobileViTXXSDeepSE(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
