from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ChannelAttention(nn.Module):
    """CBAM-style channel attention (avg + max branches)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // int(reduction))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * self.sigmoid(average + maximum)


class EfficientChannelAttention(nn.Module):
    """ECA: 1D local cross-channel interaction after GAP."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.sigmoid(self.conv(weights))
        weights = weights.transpose(-1, -2).unsqueeze(-1)
        return x * weights


class PoolingTokenMixer(nn.Module):
    """PoolFormer token mixer: AvgPool(x) - x."""

    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        padding = int(pool_size) // 2
        self.pool = nn.AvgPool2d(
            kernel_size=int(pool_size),
            stride=1,
            padding=padding,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class PoolFormerBlock2d(nn.Module):
    """A lightweight 2D PoolFormer block (MetaFormer with pooling mixer)."""

    def __init__(
        self,
        channels: int,
        pool_size: int = 3,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        hidden = max(8, int(channels * float(mlp_ratio)))
        self.norm1 = nn.GroupNorm(1, channels)
        self.token_mixer = PoolingTokenMixer(pool_size=pool_size)
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale_1.view(1, -1, 1, 1) * self.token_mixer(
            self.norm1(x)
        )
        x = x + self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x))
        return x


class CEPModule(nn.Module):
    """CA + ECA + PoolFormer, with zero-init residual to the input.

    Inspired by MobileViT-DAP (Agriculture 2025), but packaged as one residual
    module for clean ablation on MobileViT-XXS.
    """

    def __init__(
        self,
        channels: int,
        ca_reduction: int = 16,
        eca_kernel: int = 3,
        pool_size: int = 3,
        mlp_ratio: float = 2.0,
        use_ca: bool = True,
        use_eca: bool = True,
        use_poolformer: bool = True,
    ) -> None:
        super().__init__()
        self.use_ca = bool(use_ca)
        self.use_eca = bool(use_eca)
        self.use_poolformer = bool(use_poolformer)

        self.ca = (
            ChannelAttention(channels, reduction=ca_reduction)
            if self.use_ca
            else nn.Identity()
        )
        self.eca = (
            EfficientChannelAttention(kernel_size=eca_kernel)
            if self.use_eca
            else nn.Identity()
        )
        self.poolformer = (
            PoolFormerBlock2d(
                channels=channels,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
            )
            if self.use_poolformer
            else nn.Identity()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.ca(x)
        features = self.eca(features)
        features = self.poolformer(features)
        return x + self.gamma * (features - x)


class MobileViTXXSCEP(nn.Module):
    """MobileViT-XXS + deep CEP modules (CA / ECA / PoolFormer)."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        ca_reduction: int = 16,
        eca_kernel: int = 3,
        pool_size: int = 3,
        mlp_ratio: float = 2.0,
        use_ca: bool = True,
        use_eca: bool = True,
        use_poolformer: bool = True,
        use_stage2_cep: bool = True,
        use_stage3_cep: bool = True,
        use_stage4_cep: bool = True,
        use_final_cep: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-CEP requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        feature_dim = int(self.backbone.num_features)

        def build_cep(channels: int) -> nn.Module:
            return CEPModule(
                channels=channels,
                ca_reduction=ca_reduction,
                eca_kernel=eca_kernel,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                use_ca=use_ca,
                use_eca=use_eca,
                use_poolformer=use_poolformer,
            )

        self.cep_stage2 = build_cep(48) if use_stage2_cep else nn.Identity()
        self.cep_stage3 = build_cep(64) if use_stage3_cep else nn.Identity()
        self.cep_stage4 = build_cep(80) if use_stage4_cep else nn.Identity()
        self.cep_final = build_cep(feature_dim) if use_final_cep else nn.Identity()

        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.num_features = feature_dim

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        features = self.backbone.stages[1](features)
        features = self.backbone.stages[2](features)
        features = self.cep_stage2(features)
        features = self.backbone.stages[3](features)
        features = self.cep_stage3(features)
        features = self.backbone.stages[4](features)
        features = self.cep_stage4(features)
        features = self.backbone.final_conv(features)
        features = self.cep_final(features)
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


def mobilevit_xxs_cep(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSCEP:
    return MobileViTXXSCEP(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
