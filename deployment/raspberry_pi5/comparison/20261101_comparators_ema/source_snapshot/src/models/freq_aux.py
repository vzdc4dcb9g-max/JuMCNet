from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.julitemobileatt import JuLiteMobileAtt


class FFTMagnitudeBranch(nn.Module):
    """Log-magnitude FFT encoder (FAF-Net style frequency auxiliary branch)."""

    def __init__(self, out_dim: int, spectrum_size: int = 112) -> None:
        super().__init__()
        self.spectrum_size = int(spectrum_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, int(out_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Channel-mean spectrum from the (already normalized) RGB tensor.
        gray = inputs.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft2(gray.float(), norm="ortho")
        magnitude = torch.log1p(torch.abs(spectrum))
        magnitude = F.interpolate(
            magnitude,
            size=(self.spectrum_size, self.spectrum_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        magnitude = (magnitude - mean) / std
        return self.encoder(magnitude.to(dtype=inputs.dtype))


class FreqAuxClassifier(nn.Module):
    """Spatial backbone + FFT branch (+ optional healthy/diseased aux head)."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        use_freq_branch: bool = True,
        use_aux_head: bool = True,
        freq_dim: int = 128,
        spectrum_size: int = 112,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.use_freq_branch = bool(use_freq_branch)
        self.use_aux_head = bool(use_aux_head)

        self.backbone, spatial_dim = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.spatial_dim = int(spatial_dim)
        self.freq_dim = int(freq_dim) if self.use_freq_branch else 0

        if self.use_freq_branch:
            self.freq_branch = FFTMagnitudeBranch(
                out_dim=self.freq_dim,
                spectrum_size=spectrum_size,
            )
            fused_in = self.spatial_dim + self.freq_dim
            self.fuse = nn.Sequential(
                nn.Linear(fused_in, self.spatial_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.freq_branch = None
            self.fuse = None

        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(self.spatial_dim, num_classes)
        self.aux_head = (
            nn.Linear(self.spatial_dim, 2) if self.use_aux_head else None
        )
        self.num_features = self.spatial_dim

    @staticmethod
    def _build_backbone(
        backbone_name: str,
        pretrained: bool,
        dropout: float,
    ) -> tuple[nn.Module, int]:
        name = backbone_name.lower()
        if name in {"mobilevit_xxs", "repvit_m0_9"}:
            try:
                import timm
            except ImportError as error:
                raise ImportError(
                    f"{backbone_name} FAF wrapper requires timm."
                ) from error
            backbone = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,
                drop_rate=dropout,
            )
            spatial_dim = int(backbone.num_features)
            return backbone, spatial_dim

        if name == "julitemobileatt":
            if pretrained:
                raise ValueError(
                    "julitemobileatt_faf follows the from-scratch protocol; "
                    "set pretrained=false."
                )
            backbone = JuLiteMobileAtt(num_classes=6, dropout=dropout)
            # Keep trunk; discard its 6-way head (we rebuild fused classifier).
            backbone.classifier = nn.Identity()
            return backbone, 1280

        raise ValueError(
            f"Unsupported FAF backbone '{backbone_name}'. "
            "Use mobilevit_xxs, julitemobileatt, or repvit_m0_9."
        )

    def forward_spatial_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backbone_name.lower() == "julitemobileatt":
            features = inputs
            for index, block in enumerate(self.backbone.features):
                features = block(features)
                key = str(index)
                if key in self.backbone.stage_attention:
                    features = self.backbone.stage_attention[key](features)
            features = self.backbone.cascaded_rcam(features)
            features = self.backbone.final_conv(features)
            return self.backbone.pool(features).flatten(1)

        features = self.backbone(inputs)
        if features.ndim > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features

    def forward_fused_features(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = self.forward_spatial_features(inputs)
        if not self.use_freq_branch or self.freq_branch is None or self.fuse is None:
            return spatial
        frequency = self.freq_branch(inputs)
        return self.fuse(torch.cat([spatial, frequency], dim=1))

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        features = self.forward_fused_features(inputs)
        logits = self.classifier(self.dropout(features))

        aux_logits: torch.Tensor | None = None
        if return_aux:
            if self.aux_head is None:
                raise RuntimeError(
                    "return_aux=True but use_aux_head=False on this model."
                )
            aux_logits = self.aux_head(features)

        if return_features and return_aux:
            assert aux_logits is not None
            return logits, features, aux_logits
        if return_aux:
            assert aux_logits is not None
            return logits, aux_logits
        if return_features:
            return logits, features
        return logits


def freq_aux_classifier(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> FreqAuxClassifier:
    return FreqAuxClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )


def mobilevit_xxs_faf(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> FreqAuxClassifier:
    return freq_aux_classifier(
        backbone_name="mobilevit_xxs",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )


def julitemobileatt_faf(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> FreqAuxClassifier:
    return freq_aux_classifier(
        backbone_name="julitemobileatt",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )


def repvit_m0_9_faf(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> FreqAuxClassifier:
    return freq_aux_classifier(
        backbone_name="repvit_m0_9",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
