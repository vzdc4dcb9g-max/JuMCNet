from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MFFMobileVitBlock
from src.models.mobilevit_xxs_mid_bypass import MidFeatureAdapter, SqueezeExcitation


class SelectiveChannelGate(nn.Module):
    """SE-like channel gate for mid-bypass (scheme-2 selective distillation).

    GAP → Linear(C, C/r) → ReLU → Linear(C/r, C) → Sigmoid → channel scale.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, int(channels) // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(int(channels), hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, int(channels), bias=True),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.shape
        gates = self.fc(self.pool(inputs).view(batch, channels)).view(
            batch, channels, 1, 1
        )
        return inputs * gates


class MobileViTXXSMFFMidBypass(nn.Module):
    """MobileViT-XXS + default MFF blocks + mid-stage bypass head.

    - MFF wraps MobileViT blocks in stages 2/3/4 (same as EMA-MFF).
    - Mid bypass taps stages[bypass_stage] (default 2: 48ch @ 28x28),
      optionally applies SelectiveChannelGate, adapts to final 7x7 grid,
      concat with final_conv, light SE, GAP+Linear.
    - No HF / Tail. EMA is enabled via model yaml (training recipe).
    """

    _STAGE_CHANNELS = {1: 24, 2: 48, 3: 64, 4: 80}

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        use_mff: bool = True,
        use_stage2_mff: bool = True,
        use_stage3_mff: bool = True,
        use_stage4_mff: bool = True,
        bypass_stage: int = 2,
        bypass_channels: int = 64,
        se_reduction: int = 8,
        use_se_head: bool = True,
        use_mid_channel_gate: bool = False,
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "MobileViT-XXS-MFF-MidBypass requires timm."
            ) from error

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
        if bool(use_mff):
            self._wrap_mff_blocks(
                use_stage2_mff=bool(use_stage2_mff),
                use_stage3_mff=bool(use_stage3_mff),
                use_stage4_mff=bool(use_stage4_mff),
            )

        final_channels = int(self.backbone.num_features)
        mid_in = self._STAGE_CHANNELS[int(bypass_stage)]
        mid_out = int(bypass_channels)
        head_channels = final_channels + mid_out

        self.bypass_stage = int(bypass_stage)
        self.use_mid_channel_gate = bool(use_mid_channel_gate)
        self.mid_channel_gate = (
            SelectiveChannelGate(mid_in, reduction=gate_reduction)
            if self.use_mid_channel_gate
            else nn.Identity()
        )
        self.adapter = MidFeatureAdapter(mid_in, mid_out)
        self.se_head = (
            SqueezeExcitation(head_channels, reduction=se_reduction)
            if use_se_head
            else nn.Identity()
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(head_channels, num_classes)
        self.num_features = head_channels

    def _wrap_mff_blocks(
        self,
        use_stage2_mff: bool,
        use_stage3_mff: bool,
        use_stage4_mff: bool,
    ) -> None:
        from timm.models.mobilevit import MobileVitBlock

        stage_flags = {
            2: use_stage2_mff,
            3: use_stage3_mff,
            4: use_stage4_mff,
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block_index, block in enumerate(stage):
                if isinstance(block, MobileVitBlock):
                    stage[block_index] = MFFMobileVitBlock(block)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        mid_feat: torch.Tensor | None = None
        for stage_index, stage in enumerate(self.backbone.stages):
            features = stage(features)
            if stage_index == self.bypass_stage:
                mid_feat = features

        if mid_feat is None:
            raise RuntimeError(
                f"Failed to tap bypass_stage={self.bypass_stage}"
            )

        features = self.backbone.final_conv(features)
        mid_feat = self.mid_channel_gate(mid_feat)
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


def mobilevit_xxs_ema_mff_mid_bypass(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSMFFMidBypass:
    return MobileViTXXSMFFMidBypass(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )


def mobilevit_xxs_ema_mff_mid_sgate(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSMFFMidBypass:
    """EMA+MFF + mid-bypass with selective channel gate (scheme-2)."""
    kwargs = dict(kwargs)
    kwargs.setdefault("use_mid_channel_gate", True)
    return MobileViTXXSMFFMidBypass(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
