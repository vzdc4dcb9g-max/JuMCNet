from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MFFMobileVitBlock, MobileViTXXSDFN


class ResidualCodebookTextureHead(nn.Module):
    """Training-only residual encoding head for unordered local textures."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        reduced_channels: int = 48,
        num_codewords: int = 4,
        projection_dim: int = 64,
        assignment_scale: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(
            in_channels,
            num_classes,
            reduced_channels,
            projection_dim,
        ) < 1:
            raise ValueError("RCAux dimensions must all be >= 1")
        if num_codewords < 2:
            raise ValueError("RCAux num_codewords must be >= 2")
        if assignment_scale <= 0.0 or eps <= 0.0:
            raise ValueError("RCAux assignment_scale and eps must be > 0")

        self.num_codewords = int(num_codewords)
        self.reduced_channels = int(reduced_channels)
        self.eps = float(eps)
        self.reduce = nn.Conv2d(
            int(in_channels), self.reduced_channels, kernel_size=1, bias=True
        )
        self.codewords = nn.Parameter(
            torch.empty(self.num_codewords, self.reduced_channels)
        )
        # Positive, per-codeword smoothing factors follow the Deep TEN form
        # exp(-s_k ||x_i-c_k||^2), while softplus keeps them well-defined.
        raw_scale = math.log(math.expm1(float(assignment_scale)))
        self.raw_assignment_scale = nn.Parameter(
            torch.full((self.num_codewords,), raw_scale)
        )
        self.project = nn.Sequential(
            nn.Linear(
                self.num_codewords * self.reduced_channels,
                int(projection_dim),
                bias=True,
            ),
            nn.LayerNorm(int(projection_dim)),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(
            int(projection_dim), int(num_classes), bias=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.reduce.weight, mode="fan_out")
        nn.init.zeros_(self.reduce.bias)
        nn.init.uniform_(
            self.codewords,
            -1.0 / math.sqrt(self.reduced_channels),
            1.0 / math.sqrt(self.reduced_channels),
        )
        nn.init.trunc_normal_(self.project[0].weight, std=0.02)
        nn.init.zeros_(self.project[0].bias)
        nn.init.ones_(self.project[1].weight)
        nn.init.zeros_(self.project[1].bias)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("RCAux expects a BCHW Stage3 local feature map")
        descriptors = self.reduce(features).flatten(2).transpose(1, 2)
        residuals = descriptors.unsqueeze(2) - self.codewords.view(
            1, 1, self.num_codewords, self.reduced_channels
        )
        squared_distance = residuals.float().square().sum(dim=-1)
        scales = F.softplus(self.raw_assignment_scale.float()).clamp_min(
            self.eps
        )
        assignments = torch.softmax(
            -squared_distance * scales.view(1, 1, -1), dim=-1
        )

        encoded = (
            assignments.unsqueeze(-1) * residuals.float()
        ).sum(dim=1)
        encoded = F.normalize(encoded, p=2, dim=-1, eps=self.eps)
        encoded = F.normalize(
            encoded.flatten(1), p=2, dim=-1, eps=self.eps
        )
        projected = self.project(encoded)
        auxiliary_logits = self.classifier(projected)

        usage = assignments.mean(dim=(0, 1))
        sample_entropy = -(
            assignments.clamp_min(self.eps)
            * assignments.clamp_min(self.eps).log()
        ).sum(dim=-1)
        normalized_entropy = sample_entropy.mean() / math.log(
            self.num_codewords
        )
        return {
            "auxiliary_logits": auxiliary_logits,
            "codebook_usage": usage,
            "assignment_entropy": normalized_entropy,
        }


class JuMCNet(MobileViTXXSDFN):
    """LocalECA mainline with a removable Stage3 residual-codebook head."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        rcaux_reduced_channels: int = 48,
        rcaux_num_codewords: int = 4,
        rcaux_projection_dim: int = 64,
        rcaux_assignment_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if str(kwargs.get("mff_variant", "default")).strip().lower() not in {
            "default",
            "mff",
        }:
            raise ValueError("RCAux currently requires mff_variant='default'")
        if not bool(kwargs.get("use_mff", True)) or not bool(
            kwargs.get("use_stage3_mff", True)
        ):
            raise ValueError("RCAux requires Stage3 default MFF")
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
                "RCAux expects exactly one Stage3 MFFMobileVitBlock; "
                f"found {len(stage3_blocks)}"
            )
        # Keep a convenient reference without registering the same Stage3
        # block twice in state_dict; the block is already owned by backbone.
        object.__setattr__(self, "rcaux_block", stage3_blocks[0])
        transformer_channels = int(
            self.rcaux_block.block.conv_1x1.out_channels
        )
        self.rcaux_head: ResidualCodebookTextureHead | None = (
            ResidualCodebookTextureHead(
                in_channels=transformer_channels,
                num_classes=int(num_classes),
                reduced_channels=int(rcaux_reduced_channels),
                num_codewords=int(rcaux_num_codewords),
                projection_dim=int(rcaux_projection_dim),
                assignment_scale=float(rcaux_assignment_scale),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> JuMCNet:
        """Delete RCAux parameters and disable feature capture for deployment."""
        self.rcaux_head = None
        self.rcaux_block.capture_relations = False
        self.rcaux_block.last_local_feat = None
        self.rcaux_block.last_global_feat = None
        self.inference_auxiliary_stripped = True
        return self

    def _forward_rcaux(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.rcaux_head is None:
            raise RuntimeError("RCAux training head was stripped for inference")
        self.rcaux_block.capture_relations = True
        try:
            features = self.forward_features(inputs)
            local_features = self.rcaux_block.last_local_feat
            if local_features is None:
                raise RuntimeError("Stage3 RCAux local feature capture failed")
            logits = self.classifier(self.dropout(features))
            state = self.rcaux_head(local_features)
        finally:
            self.rcaux_block.capture_relations = False
            self.rcaux_block.last_local_feat = None
            self.rcaux_block.last_global_feat = None
        return logits, state

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_rcaux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("RCAux is not combined with the global aux head")
        if return_features and return_rcaux:
            raise ValueError("return_features and return_rcaux are exclusive")
        if return_rcaux:
            return self._forward_rcaux(inputs)
        return super().forward(inputs, return_features=return_features)


def jumcnet(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> JuMCNet:
    return JuMCNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
