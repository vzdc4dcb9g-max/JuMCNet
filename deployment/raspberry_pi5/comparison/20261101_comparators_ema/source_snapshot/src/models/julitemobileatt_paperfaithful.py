"""Paper-faithful reimplementation of Ju-LiteMobileAtt (PLOS ONE, 2025).

The paper does not release source code and leaves the grouped convolution in
its named CoordAtt underspecified.  This implementation follows Table 1,
Figures 2-4, and Equations (1), (3), and (4) as literally as tensor dimensions
allow.  See ``docs/reproductions/julitemobileatt_paperfaithful_2025.md`` for
the fixed assumptions.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v2


class PaperSE(nn.Module):
    """SE path denoted by ``s`` in Equations (1) and (4)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return x * self.sigmoid(scale)


class PaperCoordAtt(nn.Module):
    """The paper's SE-then-grouped-spatial-attention CoordAtt.

    Equation (1) applies a grouped convolution to ``s * F``, averages the
    result over channels (``dim=1``), and uses the resulting spatial mask to
    modulate ``s * F`` before adding the input residual.  The grouped
    convolution kernel and group count are not reported; a 3x3 depthwise
    convolution is the smallest standard grouped interpretation.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd")
        self.se = PaperSE(channels, reduction=reduction)
        self.group_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            groups=channels,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.reset_attention_logits()

    def reset_attention_logits(self) -> None:
        # Section 3.4 explicitly states that the convolution generating the
        # attention weights is zero-initialized.
        nn.init.zeros_(self.group_conv.weight)
        nn.init.zeros_(self.group_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_selected = self.se(x)
        grouped = self.group_conv(channel_selected)
        spatial_mask = self.sigmoid(grouped.mean(dim=1, keepdim=True))
        return x + spatial_mask * channel_selected


class ConvBNHSwish(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class PaperEfficientRCAM(nn.Module):
    """EfficientRCAM following Figure 3's compression/expansion pathway."""

    def __init__(
        self,
        channels: int,
        compression: int = 2,
        se_reduction: int = 16,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        if channels % compression != 0:
            raise ValueError("channels must be divisible by compression")
        hidden = channels // compression
        self.depthwise = ConvBNHSwish(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
        )
        self.reduce = ConvBNHSwish(channels, hidden, kernel_size=1)
        self.attention = PaperCoordAtt(
            hidden,
            reduction=se_reduction,
            spatial_kernel=spatial_kernel,
        )
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.reduce(x)
        x = self.attention(x)
        x = self.expand(x)
        return self.activation(residual + x)


class PaperCascadedRCAM(nn.Module):
    """CascadedRCAM using Equation (3)'s add-then-adapt interpretation."""

    def __init__(
        self,
        channels: int,
        compression: int = 2,
        se_reduction: int = 16,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.efficient_rcam = PaperEfficientRCAM(
            channels,
            compression=compression,
            se_reduction=se_reduction,
            spatial_kernel=spatial_kernel,
        )
        self.channel_adapt = ConvBNHSwish(
            channels,
            channels // compression,
            kernel_size=1,
        )
        self.out_channels = channels // compression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equation (3): ConvBNHSwish(F_in + F_rcam, K_adapt, C/2).
        enhanced = self.efficient_rcam(x)
        return self.channel_adapt(x + enhanced)


class JuLiteMobileAttPaperFaithful(nn.Module):
    """Ju-LiteMobileAtt reconstructed from the paper's published specification."""

    # torchvision indices 3, 6, 10, 13, 16, and 17 are the final blocks of
    # paper Stages 1-6.  Stage 0 (index 1) explicitly has no CoordAtt in Table 1.
    stage_end_indices = (3, 6, 10, 13, 16, 17)
    stage_channels = (24, 32, 64, 96, 160, 320)

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.0,
        se_reduction: int = 16,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        baseline = mobilenet_v2(weights=None, num_classes=num_classes)
        features = list(baseline.features.children())[:18]

        # Figure 4 specifies a Conv-BN-HSwish stem, whereas torchvision's
        # MobileNetV2 stem uses ReLU6.
        features[0][2] = nn.Hardswish(inplace=True)
        self.features = nn.ModuleList(features)

        self.stage_attention = nn.ModuleDict(
            {
                str(index): PaperCoordAtt(
                    channels,
                    reduction=se_reduction,
                    spatial_kernel=spatial_kernel,
                )
                for index, channels in zip(
                    self.stage_end_indices,
                    self.stage_channels,
                )
            }
        )
        self.cascaded_rcam = PaperCascadedRCAM(
            320,
            compression=2,
            se_reduction=se_reduction,
            spatial_kernel=spatial_kernel,
        )
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
        self._restore_paper_zero_initialization()

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

    def _restore_paper_zero_initialization(self) -> None:
        for module in self.modules():
            if isinstance(module, PaperCoordAtt):
                module.reset_attention_logits()

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


def julitemobileatt_paperfaithful(
    num_classes: int = 6,
    dropout: float = 0.0,
    se_reduction: int = 16,
    spatial_kernel: int = 3,
) -> JuLiteMobileAttPaperFaithful:
    return JuLiteMobileAttPaperFaithful(
        num_classes=num_classes,
        dropout=dropout,
        se_reduction=se_reduction,
        spatial_kernel=spatial_kernel,
    )
