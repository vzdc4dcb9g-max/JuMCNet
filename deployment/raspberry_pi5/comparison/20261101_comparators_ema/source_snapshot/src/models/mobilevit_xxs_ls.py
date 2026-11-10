from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LargeKernelPerception(nn.Module):
    """Large-kernel static perception that predicts small-kernel aggregation weights.

    Inspired by LSNet (CVPR 2025): PW bottleneck -> large DW -> PW weight head.
    """

    def __init__(
        self,
        channels: int,
        groups: int,
        large_kernel: int = 7,
        small_kernel: int = 3,
    ) -> None:
        super().__init__()
        mid = max(8, channels // 2)
        weight_dim = int(groups) * int(small_kernel) * int(small_kernel)
        padding = int(large_kernel) // 2
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.large_dw = nn.Sequential(
            nn.Conv2d(
                mid,
                mid,
                kernel_size=int(large_kernel),
                padding=padding,
                groups=mid,
                bias=False,
            ),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.weight_gen = nn.Conv2d(mid, weight_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.reduce(x)
        features = self.large_dw(features)
        return self.weight_gen(features)


class SmallKernelAggregation(nn.Module):
    """Grouped dynamic small-kernel aggregation guided by LKP weights."""

    def __init__(self, channels: int, groups: int, small_kernel: int = 3) -> None:
        super().__init__()
        if channels % groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by groups ({groups})."
            )
        self.channels = int(channels)
        self.groups = int(groups)
        self.small_kernel = int(small_kernel)
        self.channels_per_group = self.channels // self.groups
        self.padding = self.small_kernel // 2

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}."
            )

        patches = F.unfold(
            x,
            kernel_size=self.small_kernel,
            padding=self.padding,
        )
        # (B, G, C/G, K*K, H*W)
        patches = patches.view(
            batch,
            self.groups,
            self.channels_per_group,
            self.small_kernel * self.small_kernel,
            height * width,
        )
        # (B, G, 1, K*K, H*W)
        kernel_weights = weights.view(
            batch,
            self.groups,
            1,
            self.small_kernel * self.small_kernel,
            height * width,
        )
        aggregated = (patches * kernel_weights).sum(dim=3)
        return aggregated.view(batch, channels, height, width)


class LSConv(nn.Module):
    """LS convolution: large-kernel perception + small-kernel aggregation."""

    def __init__(
        self,
        channels: int,
        large_kernel: int = 7,
        small_kernel: int = 3,
        channels_per_group: int = 8,
    ) -> None:
        super().__init__()
        groups = max(1, channels // int(channels_per_group))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.groups = groups
        self.lkp = LargeKernelPerception(
            channels=channels,
            groups=groups,
            large_kernel=large_kernel,
            small_kernel=small_kernel,
        )
        self.ska = SmallKernelAggregation(
            channels=channels,
            groups=groups,
            small_kernel=small_kernel,
        )
        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.lkp(x)
        aggregated = self.ska(x, weights)
        return self.out_norm(aggregated)


class ResidualLS(nn.Module):
    """Zero-init residual LS so training starts as identity."""

    def __init__(
        self,
        channels: int,
        large_kernel: int = 7,
        small_kernel: int = 3,
        channels_per_group: int = 8,
    ) -> None:
        super().__init__()
        self.ls = LSConv(
            channels=channels,
            large_kernel=large_kernel,
            small_kernel=small_kernel,
            channels_per_group=channels_per_group,
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.ls(x)


class MobileViTXXSLS(nn.Module):
    """MobileViT-XXS with deep residual LS convolution.

    timm MobileViT-XXS is kept intact. LS modules are inserted only after
    stages 2/3/4 and final_conv (semantic maps), matching the deep-SE mount
    points used in this project.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        large_kernel: int = 7,
        small_kernel: int = 3,
        channels_per_group: int = 8,
        use_stage2_ls: bool = True,
        use_stage3_ls: bool = True,
        use_stage4_ls: bool = True,
        use_final_ls: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-LS requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        feature_dim = int(self.backbone.num_features)

        def build_ls(channels: int) -> nn.Module:
            return ResidualLS(
                channels=channels,
                large_kernel=large_kernel,
                small_kernel=small_kernel,
                channels_per_group=channels_per_group,
            )

        self.ls_stage2 = build_ls(48) if use_stage2_ls else nn.Identity()
        self.ls_stage3 = build_ls(64) if use_stage3_ls else nn.Identity()
        self.ls_stage4 = build_ls(80) if use_stage4_ls else nn.Identity()
        self.ls_final = build_ls(feature_dim) if use_final_ls else nn.Identity()

        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.num_features = feature_dim

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        features = self.backbone.stages[1](features)
        features = self.backbone.stages[2](features)
        features = self.ls_stage2(features)
        features = self.backbone.stages[3](features)
        features = self.ls_stage3(features)
        features = self.backbone.stages[4](features)
        features = self.ls_stage4(features)
        features = self.backbone.final_conv(features)
        features = self.ls_final(features)
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


def mobilevit_xxs_ls(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSLS:
    return MobileViTXXSLS(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
