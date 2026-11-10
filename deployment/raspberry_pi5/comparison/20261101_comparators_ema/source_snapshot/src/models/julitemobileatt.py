"""Paper-guided reimplementation of Ju-LiteMobileAtt.

Reference: Zhu et al., PLOS ONE 2025, DOI 10.1371/journal.pone.0337898.
The paper does not provide source code and leaves several tensor/channel details
underspecified.  See docs/reproductions/julitemobileatt_2025.md for the explicit
assumptions used here.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v2


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class EfficientChannelAttention(nn.Module):
    """ECA-style local channel interaction after global average pooling."""

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


class CoordAtt(nn.Module):
    """The paper's named CoordAtt: cascaded SE and ECA, with zero-init residual."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction=reduction)
        self.eca = EfficientChannelAttention(kernel_size=3)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.eca(self.se(x))
        return x + self.residual_scale * attended


class ConvBNHSwish(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class EfficientRCAM(nn.Module):
    def __init__(self, channels: int, compression: int = 2) -> None:
        super().__init__()
        hidden = channels // compression
        self.depthwise = ConvBNHSwish(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
        )
        self.reduce = ConvBNHSwish(channels, hidden, kernel_size=1)
        self.attention = CoordAtt(hidden)
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.reduce(y)
        y = self.attention(y)
        y = self.expand(y)
        return self.activation(x + y)


class CascadedRCAM(nn.Module):
    def __init__(self, channels: int, compression: int = 2) -> None:
        super().__init__()
        out_channels = channels // compression
        self.efficient_rcam = EfficientRCAM(channels, compression=compression)
        self.channel_adapt = ConvBNHSwish(channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.efficient_rcam(x)
        return self.channel_adapt(x + enhanced)


class JuLiteMobileAtt(nn.Module):
    stage_end_indices = (1, 3, 6, 10, 13, 16, 17)
    stage_channels = (16, 24, 32, 64, 96, 160, 320)

    def __init__(self, num_classes: int = 6, dropout: float = 0.0) -> None:
        super().__init__()
        baseline = mobilenet_v2(weights=None, num_classes=num_classes)
        # Keep the initial convolution and all seven MobileNetV2 inverted-residual
        # stages; replace the original 320->1280 tail as described by Fig. 4.
        self.features = nn.ModuleList(list(baseline.features.children())[:18])
        self.stage_attention = nn.ModuleDict(
            {
                str(index): CoordAtt(channels)
                for index, channels in zip(
                    self.stage_end_indices,
                    self.stage_channels,
                )
            }
        )
        self.cascaded_rcam = CascadedRCAM(320, compression=2)
        self.final_conv = ConvBNHSwish(
            self.cascaded_rcam.out_channels,
            1280,
            kernel_size=1,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        for module in [
            self.stage_attention,
            self.cascaded_rcam,
            self.final_conv,
            self.classifier,
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.features):
            x = block(x)
            key = str(index)
            if key in self.stage_attention:
                x = self.stage_attention[key](x)
        x = self.cascaded_rcam(x)
        x = self.final_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def julitemobileatt(num_classes: int = 6, dropout: float = 0.0) -> JuLiteMobileAtt:
    return JuLiteMobileAtt(num_classes=num_classes, dropout=dropout)
