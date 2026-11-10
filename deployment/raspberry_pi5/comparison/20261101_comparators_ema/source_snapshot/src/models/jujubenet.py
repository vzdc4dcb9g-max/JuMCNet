"""Paper-guided reimplementation of JujubeNet.

Reference: Jiang et al., Frontiers in Plant Science 2023,
DOI 10.3389/fpls.2022.1108437.
The authors did not release source code.  Explicit reconstruction assumptions are
documented in docs/reproductions/jujubenet_2023.md.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            (x.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(
            keep_probability
        )
        return x * mask / keep_probability


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class MDCBlock(nn.Module):
    """Multi-branch depthwise-convolution block from Figs. 4 and 7."""

    def __init__(
        self,
        channels: int,
        drop_path: float = 0.0,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
    ) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("MDCBlock requires an even channel count.")
        branch_channels = channels // 2
        expanded_channels = branch_channels * expansion_ratio
        padding = kernel_size // 2

        self.pre_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = LayerNorm2d(channels)

        self.depthwise_branch = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size,
                padding=padding,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size,
                padding=padding,
                groups=branch_channels,
                bias=False,
            ),
        )
        # Figure 4 labels these as Conv-GELU-Conv without kernel/channel sizes.
        # Pointwise 4x expansion is the ConvNeXt-compatible interpretation that
        # reproduces the paper's reported 8.5M-parameter scale.
        self.pointwise_branch = nn.Sequential(
            nn.Conv2d(branch_channels, expanded_channels, 1),
            nn.GELU(),
            nn.Conv2d(expanded_channels, branch_channels, 1),
        )
        self.post_norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(self.pre_depthwise(x))
        left, right = torch.chunk(x, 2, dim=1)
        left = self.depthwise_branch(left)
        right = self.pointwise_branch(right)
        x = torch.cat((left, right), dim=1)
        x = self.activation(self.post_norm(x))
        return shortcut + self.drop_path(x)


class JujubeNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("JujubeNet expects four ConvNeXt-Tiny stages.")

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(dims[index]),
                    nn.Conv2d(
                        dims[index],
                        dims[index + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
                for index in range(3)
            ]
        )
        self.attention = nn.ModuleList([CBAM(dim) for dim in dims])

        total_blocks = sum(depths)
        rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        offset = 0
        stages: list[nn.Sequential] = []
        for depth, dim in zip(depths, dims):
            stages.append(
                nn.Sequential(
                    *[
                        MDCBlock(dim, drop_path=rates[offset + block_index])
                        for block_index in range(depth)
                    ]
                )
            )
            offset += depth
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(dims[-1], num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention[0](self.stem(x))
        x = self.stages[0](x)
        for index in range(1, 4):
            x = self.downsamples[index - 1](x)
            x = self.attention[index](x)
            x = self.stages[index](x)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        return self.classifier(x)


def jujubenet(
    num_classes: int = 6,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
) -> JujubeNet:
    return JujubeNet(
        num_classes=num_classes,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )
