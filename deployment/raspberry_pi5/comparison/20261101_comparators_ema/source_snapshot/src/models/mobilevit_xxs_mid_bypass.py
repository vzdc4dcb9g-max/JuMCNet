from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MidFeatureAdapter(nn.Module):
    """Project mid-stage maps and align spatially to the final feature grid."""

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self, inputs: torch.Tensor, spatial_size: tuple[int, int]
    ) -> torch.Tensor:
        features = self.proj(inputs)
        if features.shape[-2:] != spatial_size:
            features = F.adaptive_avg_pool2d(features, spatial_size)
        return features


class SqueezeExcitation(nn.Module):
    """Light SE used as the post-concat decision head."""

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.shape
        weights = self.fc(self.pool(inputs).view(batch, channels)).view(
            batch, channels, 1, 1
        )
        return inputs * weights


class MobileViTXXSMidBypass(nn.Module):
    """MobileViT-XXS with a mid-stage high-res bypass into the classifier.

    Tap after the first MobileViT stage (stages[2], 48ch @ 28x28), adapt with a
    1x1+BN+GELU bottleneck, downsample to the final 7x7 grid, concat with
    final_conv features, apply light SE, then GAP + Linear.
    """

    _STAGE_CHANNELS = {1: 24, 2: 48, 3: 64, 4: 80}

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        bypass_stage: int = 2,
        bypass_channels: int = 64,
        se_reduction: int = 8,
        use_se_head: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-MidBypass requires timm.") from error

        if int(bypass_stage) not in self._STAGE_CHANNELS:
            raise ValueError(
                f"bypass_stage must be one of {sorted(self._STAGE_CHANNELS)}, "
                f"got {bypass_stage}"
            )

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        final_channels = int(self.backbone.num_features)
        mid_in = self._STAGE_CHANNELS[int(bypass_stage)]
        mid_out = int(bypass_channels)
        head_channels = final_channels + mid_out

        self.bypass_stage = int(bypass_stage)
        self.adapter = MidFeatureAdapter(mid_in, mid_out)
        self.se_head = (
            SqueezeExcitation(head_channels, reduction=se_reduction)
            if use_se_head
            else nn.Identity()
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(head_channels, num_classes)
        self.num_features = head_channels

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        mid_feat: torch.Tensor | None = None
        for stage_index, stage in enumerate(self.backbone.stages):
            features = stage(features)
            # timm stages: 0/1 conv, 2/3/4 contain MobileViT blocks.
            if stage_index == self.bypass_stage:
                mid_feat = features

        if mid_feat is None:
            raise RuntimeError(
                f"Failed to tap bypass_stage={self.bypass_stage}"
            )

        features = self.backbone.final_conv(features)
        adapted = self.adapter(mid_feat, features.shape[-2:])
        fused = torch.cat((features, adapted), dim=1)
        fused = self.se_head(fused)
        return F.adaptive_avg_pool2d(fused, 1).flatten(1)

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


def mobilevit_xxs_mid_bypass(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSMidBypass:
    return MobileViTXXSMidBypass(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
