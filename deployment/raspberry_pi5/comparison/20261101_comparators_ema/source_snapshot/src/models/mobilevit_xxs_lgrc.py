from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import (
    MFFMobileVitBlock,
    MobileViTXXSDFN,
)


class LocalGlobalRelationHead(nn.Module):
    """Training-only relation and relative-position head for Stage3.

    The local and global maps may have different channel counts, so their
    cosine relations are computed in their own feature spaces.  Only the
    global pair descriptor is fed to the coordinate regressor.
    """

    def __init__(
        self,
        global_channels: int,
        hidden_dim: int = 128,
        num_pairs: int = 64,
    ) -> None:
        super().__init__()
        if global_channels < 1 or hidden_dim < 1 or num_pairs < 1:
            raise ValueError(
                "global_channels, hidden_dim, and num_pairs must be >= 1"
            )
        self.num_pairs = int(num_pairs)
        self.position_head = nn.Sequential(
            nn.LayerNorm(int(global_channels) * 3),
            nn.Linear(int(global_channels) * 3, int(hidden_dim), bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), 2, bias=True),
        )
        # Start coordinate predictions at zero without changing main logits.
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

    @staticmethod
    def _gather_tokens(
        tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        channels = int(tokens.shape[-1])
        return torch.gather(
            tokens,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, channels),
        )

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if local_features.ndim != 4 or global_features.ndim != 4:
            raise ValueError("LGRC expects BCHW local/global feature maps")
        if local_features.shape[0] != global_features.shape[0]:
            raise ValueError("LGRC local/global batch sizes differ")

        height, width = global_features.shape[-2:]
        if local_features.shape[-2:] != (height, width):
            local_features = F.interpolate(
                local_features,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        num_tokens = int(height * width)
        if num_tokens < 2:
            raise ValueError("LGRC requires at least two spatial tokens")

        local_tokens = local_features.flatten(2).transpose(1, 2)
        global_tokens = global_features.flatten(2).transpose(1, 2)
        batch = int(global_tokens.shape[0])
        pair_count = min(self.num_pairs, num_tokens)

        first_indices = torch.randint(
            num_tokens,
            (batch, pair_count),
            device=global_features.device,
        )
        # A non-zero cyclic offset guarantees distinct endpoints without a
        # rejection loop and remains uniformly distributed over other tokens.
        offsets = torch.randint(
            1,
            num_tokens,
            (batch, pair_count),
            device=global_features.device,
        )
        second_indices = (first_indices + offsets) % num_tokens

        local_first = self._gather_tokens(local_tokens, first_indices)
        local_second = self._gather_tokens(local_tokens, second_indices)
        global_first = self._gather_tokens(global_tokens, first_indices)
        global_second = self._gather_tokens(global_tokens, second_indices)

        # The local relation is an internal anchor, not a target that should
        # itself move to satisfy the auxiliary loss.
        local_relation = F.cosine_similarity(
            local_first.float(), local_second.float(), dim=-1
        ).detach()
        global_relation = F.cosine_similarity(
            global_first.float(), global_second.float(), dim=-1
        )

        pair_descriptor = torch.cat(
            (
                global_first,
                global_second,
                global_second - global_first,
            ),
            dim=-1,
        )
        position_prediction = self.position_head(pair_descriptor.float())

        first_y = torch.div(first_indices, width, rounding_mode="floor")
        first_x = first_indices % width
        second_y = torch.div(second_indices, width, rounding_mode="floor")
        second_x = second_indices % width
        height_scale = float(max(height - 1, 1))
        width_scale = float(max(width - 1, 1))
        position_target = torch.stack(
            (
                (second_x - first_x).float() / width_scale,
                (second_y - first_y).float() / height_scale,
            ),
            dim=-1,
        )

        return {
            "local_relation": local_relation,
            "global_relation": global_relation,
            "position_prediction": position_prediction,
            "position_target": position_target,
        }


class MobileViTXXSLGRC(MobileViTXXSDFN):
    """MobileViT-XXS + MFF + LocalECA with training-only Stage3 LGRC."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        lgrc_hidden_dim: int = 128,
        lgrc_num_pairs: int = 64,
        **kwargs: Any,
    ) -> None:
        if str(kwargs.get("mff_variant", "default")).strip().lower() not in {
            "default",
            "mff",
        }:
            raise ValueError("LGRC currently requires mff_variant='default'")
        if not bool(kwargs.get("use_mff", True)) or not bool(
            kwargs.get("use_stage3_mff", True)
        ):
            raise ValueError("LGRC requires Stage3 default MFF")

        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        stage3_blocks = [
            block
            for block in self.backbone.stages[3]
            if isinstance(block, MFFMobileVitBlock)
        ]
        if len(stage3_blocks) != 1:
            raise RuntimeError(
                "LGRC expects exactly one Stage3 MFFMobileVitBlock; "
                f"found {len(stage3_blocks)}"
            )
        self.lgrc_block = stage3_blocks[0]
        self.lgrc_head: LocalGlobalRelationHead | None = (
            LocalGlobalRelationHead(
                global_channels=int(self.lgrc_block.out_channels),
                hidden_dim=int(lgrc_hidden_dim),
                num_pairs=int(lgrc_num_pairs),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSLGRC:
        """Remove every LGRC-only trainable parameter before deployment."""
        self.lgrc_head = None
        self.lgrc_block.capture_relations = False
        self.lgrc_block.last_local_feat = None
        self.lgrc_block.last_global_feat = None
        self.inference_auxiliary_stripped = True
        return self

    def _forward_lgrc(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.lgrc_head is None:
            raise RuntimeError("LGRC training head was stripped for inference")
        self.lgrc_block.capture_relations = True
        try:
            features = self.forward_features(inputs)
            local_features = self.lgrc_block.last_local_feat
            global_features = self.lgrc_block.last_global_feat
            if local_features is None or global_features is None:
                raise RuntimeError("Stage3 LGRC feature capture failed")
            logits = self.classifier(self.dropout(features))
            state = self.lgrc_head(local_features, global_features)
        finally:
            self.lgrc_block.capture_relations = False
            self.lgrc_block.last_local_feat = None
            self.lgrc_block.last_global_feat = None
        return logits, state

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_lgrc: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("LGRC is not combined with the global aux head")
        if return_features and return_lgrc:
            raise ValueError("return_features and return_lgrc are exclusive")
        if return_lgrc:
            return self._forward_lgrc(inputs)
        return super().forward(inputs, return_features=return_features)


def mobilevit_xxs_lgrc(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSLGRC:
    return MobileViTXXSLGRC(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
