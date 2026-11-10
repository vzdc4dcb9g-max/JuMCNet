from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveLocalGlobalGate(nn.Module):
    """Predict per-channel mix weights between local and global maps."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // int(reduction))
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        last_conv = self.gate[3]
        assert isinstance(last_conv, nn.Conv2d)
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.gate(torch.cat((local_feat, global_feat), dim=1))
        return weights * global_feat + (1.0 - weights) * local_feat


class AdaptiveFusionMobileVitBlock(nn.Module):
    """Wrap a timm MobileVitBlock and replace static concat-fusion.

    Keeps the original conv_fusion path and adds a zero-init residual toward
    an adaptive local/global mixture, so training starts identical to XXS.
    """

    def __init__(self, block: nn.Module, gate_reduction: int = 4) -> None:
        super().__init__()
        if block.conv_fusion is None:
            raise ValueError("Adaptive fusion requires conv_fusion to exist.")
        self.block = block
        out_channels = int(block.conv_proj.conv.out_channels)
        self.adaptive_gate = AdaptiveLocalGlobalGate(
            out_channels,
            reduction=gate_reduction,
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block = self.block
        shortcut = inputs

        features = block.conv_kxk(inputs)
        features = block.conv_1x1(features)

        patch_h, patch_w = block.patch_size
        batch, channels, height, width = features.shape
        new_h = math.ceil(height / patch_h) * patch_h
        new_w = math.ceil(width / patch_w) * patch_w
        num_patch_h = new_h // patch_h
        num_patch_w = new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        interpolate = False
        if new_h != height or new_w != width:
            features = F.interpolate(
                features,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            interpolate = True

        features = (
            features.reshape(batch * channels * num_patch_h, patch_h, num_patch_w, patch_w)
            .transpose(1, 2)
        )
        features = (
            features.reshape(batch, channels, num_patches, block.patch_area)
            .transpose(1, 3)
            .reshape(batch * block.patch_area, num_patches, -1)
        )

        features = block.transformer(features)
        features = block.norm(features)

        features = features.contiguous().view(
            batch,
            block.patch_area,
            num_patches,
            -1,
        )
        features = features.transpose(1, 3).reshape(
            batch * channels * num_patch_h,
            num_patch_w,
            patch_h,
            patch_w,
        )
        features = features.transpose(1, 2).reshape(
            batch,
            channels,
            num_patch_h * patch_h,
            num_patch_w * patch_w,
        )
        if interpolate:
            features = F.interpolate(
                features,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        global_feat = block.conv_proj(features)
        static_fused = block.conv_fusion(torch.cat((shortcut, global_feat), dim=1))
        adaptive_fused = self.adaptive_gate(shortcut, global_feat)
        return static_fused + self.gamma * (adaptive_fused - static_fused)


class MobileViTXXSAFusion(nn.Module):
    """MobileViT-XXS with adaptive local-global fusion in deep MobileViT blocks.

    By default only the deepest MobileViT block (stage 4) is wrapped.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        gate_reduction: int = 4,
        use_stage3_fusion: bool = False,
        use_stage4_fusion: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-AFusion requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        self._wrap_deep_blocks(
            gate_reduction=gate_reduction,
            use_stage3_fusion=use_stage3_fusion,
            use_stage4_fusion=use_stage4_fusion,
        )

    def _wrap_deep_blocks(
        self,
        gate_reduction: int,
        use_stage3_fusion: bool,
        use_stage4_fusion: bool,
    ) -> None:
        from timm.models.mobilevit import MobileVitBlock

        stage_flags = {
            3: bool(use_stage3_fusion),
            4: bool(use_stage4_fusion),
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block_index, block in enumerate(stage):
                if isinstance(block, MobileVitBlock):
                    stage[block_index] = AdaptiveFusionMobileVitBlock(
                        block,
                        gate_reduction=gate_reduction,
                    )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def mobilevit_xxs_afusion(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSAFusion:
    return MobileViTXXSAFusion(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
