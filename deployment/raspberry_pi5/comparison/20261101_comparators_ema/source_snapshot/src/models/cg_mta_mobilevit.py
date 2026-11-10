from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class HighFrequencyResidual(nn.Module):
    """Preserve shallow edge and spot responses before further downsampling."""

    def __init__(self, channels: int, initial_scale: float = 0.10) -> None:
        super().__init__()
        self.high_pass = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )
        with torch.no_grad():
            self.high_pass.weight.copy_(
                laplacian.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            )
        self.high_pass.weight.requires_grad_(False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        detail = self.act(self.norm(self.high_pass(inputs)))
        return inputs + self.scale * detail


class MorphologyTextureBranch(nn.Module):
    """Lightweight multi-scale branch for spots, cracks, and irregular edges."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local_3x3 = self._depthwise_block(
            channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.horizontal_1x5 = self._depthwise_block(
            channels, kernel_size=(1, 5), padding=(0, 2)
        )
        self.vertical_5x1 = self._depthwise_block(
            channels, kernel_size=(5, 1), padding=(2, 0)
        )
        self.dilated_3x3 = self._depthwise_block(
            channels,
            kernel_size=(3, 3),
            padding=(2, 2),
            dilation=(2, 2),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _depthwise_block(
        channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.local_3x3(inputs)
        directional = 0.5 * (
            self.horizontal_1x5(inputs) + self.vertical_5x1(inputs)
        )
        contextual = self.dilated_3x3(inputs)
        return self.fuse(torch.cat((local, directional, contextual), dim=1))


class AdaptiveLocalGlobalGate(nn.Module):
    """Generate a per-channel local contribution from local/global context."""

    def __init__(self, feature_dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_dim = max(feature_dim // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gate(torch.cat((global_features, local_features), dim=1))
        return global_features + weights * local_features, weights


class CGMTAMobileViT(nn.Module):
    """Confusion-guided morphology-texture adaptive MobileViT-XXS.

    The original timm MobileViT implementation remains untouched. This model
    reuses its public stages and adds independently registered modules.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        use_high_frequency: bool = True,
        use_morphology: bool = True,
        use_adaptive_gate: bool = True,
        morphology_scale: float = 0.10,
        gate_reduction: int = 8,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("CG-MTA-MobileViT requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        self.use_high_frequency = bool(use_high_frequency)
        self.use_morphology = bool(use_morphology)
        self.use_adaptive_gate = bool(use_adaptive_gate and use_morphology)

        shallow_channels = 24
        morphology_channels = 48
        feature_dim = int(self.backbone.num_features)

        self.high_frequency = (
            HighFrequencyResidual(shallow_channels)
            if self.use_high_frequency
            else nn.Identity()
        )
        if self.use_morphology:
            self.morphology = MorphologyTextureBranch(morphology_channels)
            self.morphology_scale = nn.Parameter(
                torch.tensor(float(morphology_scale))
            )
        else:
            self.morphology = nn.Identity()
            self.register_parameter("morphology_scale", None)

        if self.use_adaptive_gate:
            self.local_projection = nn.Sequential(
                nn.Linear(morphology_channels, feature_dim, bias=False),
                nn.LayerNorm(feature_dim),
                nn.SiLU(inplace=True),
            )
            self.local_global_gate = AdaptiveLocalGlobalGate(
                feature_dim=feature_dim,
                reduction=int(gate_reduction),
            )
        else:
            self.local_projection = nn.Identity()
            self.local_global_gate = nn.Identity()
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.num_features = feature_dim
        self.last_gate_weights: torch.Tensor | None = None

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        features = self.backbone.stages[1](features)

        if self.use_high_frequency:
            features = self.high_frequency(features)

        features = self.backbone.stages[2](features)
        local_features: torch.Tensor | None = None
        if self.use_morphology:
            local_map = self.morphology(features)
            features = features + self.morphology_scale * local_map
            local_features = F.adaptive_avg_pool2d(local_map, 1).flatten(1)
            local_features = self.local_projection(local_features)

        features = self.backbone.stages[3](features)
        features = self.backbone.stages[4](features)
        features = self.backbone.final_conv(features)
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        if self.use_adaptive_gate and local_features is not None:
            fused, gate_weights = self.local_global_gate(
                global_features, local_features
            )
            self.last_gate_weights = gate_weights.detach()
            return fused

        self.last_gate_weights = None
        return global_features

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


def cg_mta_mobilevit_xxs(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> CGMTAMobileViT:
    return CGMTAMobileViT(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
