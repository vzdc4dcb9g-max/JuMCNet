from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class _SpatialGuideHead(nn.Module):
    """Downsample a shallow cue into one centered spatial guidance map."""

    def __init__(self, channels: int, downsample_steps: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(int(downsample_steps)):
            blocks.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ]
            )
        self.downsample = nn.Sequential(*blocks)
        self.to_mask = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(
        self, inputs: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        mask = torch.tanh(self.to_mask(self.downsample(inputs)))
        if mask.shape[-2:] != target_size:
            mask = F.interpolate(
                mask,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return mask


class ShallowToDeepDefectGuidance(nn.Module):
    """Generate Stage3/4 spatial cues from Stage2 or fused Stage1+2 features."""

    def __init__(
        self,
        mode: Literal["s2", "s12"],
        guidance_channels: int = 16,
    ) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in {"s2", "s12"}:
            raise ValueError("SDDG mode must be 's2' or 's12'.")
        if int(guidance_channels) < 1:
            raise ValueError("guidance_channels must be >= 1.")

        channels = int(guidance_channels)
        self.mode = mode
        self.stage2_proj = nn.Sequential(
            nn.Conv2d(48, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.stage1_proj = (
            nn.Sequential(
                nn.Conv2d(24, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            if mode == "s12"
            else None
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.stage3_guide = _SpatialGuideHead(channels, downsample_steps=1)
        self.stage4_guide = _SpatialGuideHead(channels, downsample_steps=2)

        # Zero initialization keeps the initial network identical to S34.
        # The scale learns first, after which gradients reach the guide branch.
        self.stage3_scale = nn.Parameter(torch.zeros(1))
        self.stage4_scale = nn.Parameter(torch.zeros(1))
        self.stage3_scale._exclude_from_weight_decay = True
        self.stage4_scale._exclude_from_weight_decay = True

    def _source(
        self, stage1: torch.Tensor, stage2: torch.Tensor
    ) -> torch.Tensor:
        source = self.stage2_proj(stage2)
        if self.stage1_proj is not None:
            shallow = self.stage1_proj(stage1)
            shallow = F.interpolate(
                shallow,
                size=source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            source = source + shallow
        return self.fuse(source)

    def forward(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3_size: tuple[int, int],
        stage4_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self._source(stage1, stage2)
        guide3 = self.stage3_guide(source, stage3_size)
        guide4 = self.stage4_guide(source, stage4_size)
        return guide3, guide4

    def apply_stage3(
        self, features: torch.Tensor, guide: torch.Tensor
    ) -> torch.Tensor:
        return features * (1.0 + self.stage3_scale * guide)

    def apply_stage4(
        self, features: torch.Tensor, guide: torch.Tensor
    ) -> torch.Tensor:
        return features * (1.0 + self.stage4_scale * guide)


class MobileViTXXSSDDG(MobileViTXXSDFN):
    """S34 MobileViT with shallow-to-deep spatial defect guidance."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        sddg_mode: Literal["s2", "s12"] = "s2",
        sddg_channels: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        unsupported = {
            "dual_scale_head": self.use_dual_scale_head,
            "class_evidence_pool": self.use_class_evidence_pool,
            "gem_pool": self.use_gem_pool,
            "mean_std_pool": self.use_mean_std_pool,
            "mean_std_gate_pool": self.use_mean_std_gate_pool,
            "gap_gem_gate_pool": self.use_gap_gem_gate_pool,
            "global_aux": self.use_global_aux,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(
                "SDDG variants require the standard classification head; "
                f"disable: {', '.join(enabled)}."
            )
        self.sddg_mode = str(sddg_mode).strip().lower()
        self.sddg = ShallowToDeepDefectGuidance(
            mode=self.sddg_mode,
            guidance_channels=int(sddg_channels),
        )

    @staticmethod
    def _half_size(features: torch.Tensor) -> tuple[int, int]:
        height, width = features.shape[-2:]
        return max((height + 1) // 2, 1), max((width + 1) // 2, 1)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        stage1 = self.backbone.stages[1](features)
        stage1_enhanced = self.high_frequency(stage1)
        stage2 = self.backbone.stages[2](stage1_enhanced)

        stage3_size = self._half_size(stage2)
        stage4_size = (
            max((stage3_size[0] + 1) // 2, 1),
            max((stage3_size[1] + 1) // 2, 1),
        )
        guide3, guide4 = self.sddg(
            stage1=stage1,
            stage2=stage2,
            stage3_size=stage3_size,
            stage4_size=stage4_size,
        )

        stage3 = self.backbone.stages[3](stage2)
        if guide3.shape[-2:] != stage3.shape[-2:]:
            guide3 = F.interpolate(
                guide3,
                size=stage3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        stage3 = self.sddg.apply_stage3(stage3, guide3)

        stage4 = self.backbone.stages[4](stage3)
        if guide4.shape[-2:] != stage4.shape[-2:]:
            guide4 = F.interpolate(
                guide4,
                size=stage4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        stage4 = self.sddg.apply_stage4(stage4, guide4)

        features = self.backbone.final_conv(stage4)
        features = self.tail_enhance(features)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


def mobilevit_xxs_sddg(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSSDDG:
    return MobileViTXXSSDDG(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
