from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from src.models.jumcnet import (
    JuMCNet,
    ResidualCodebookTextureHead,
)


class PowerNormalizedResidualCodebookTextureHead(
    ResidualCodebookTextureHead
):
    """RCAux encoding with signed-square-root power normalization."""

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("PN-RCAux expects a BCHW Stage3 local feature map")
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
        encoded = (assignments.unsqueeze(-1) * residuals.float()).sum(dim=1)
        encoded = F.normalize(encoded, p=2, dim=-1, eps=self.eps)
        encoded = torch.sign(encoded) * torch.sqrt(
            encoded.abs().clamp_min(self.eps)
        )
        encoded = F.normalize(encoded.flatten(1), p=2, dim=-1, eps=self.eps)
        auxiliary_logits = self.classifier(self.project(encoded))
        usage = assignments.mean(dim=(0, 1))
        values = assignments.clamp_min(self.eps)
        entropy = -(values * values.log()).sum(dim=-1).mean()
        return {
            "auxiliary_logits": auxiliary_logits,
            "codebook_usage": usage,
            "assignment_entropy": entropy / math.log(self.num_codewords),
            "power_norm_mean_abs": encoded.detach().abs().mean(),
        }


class MobileViTXXSRCAuxPN(JuMCNet):
    """LocalECA mainline with a removable power-normalized RCAux head."""

    rcaux_variant = "pn"

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        if self.rcaux_head is None:
            raise RuntimeError("PN-RCAux requires an active RCAux head")
        original = self.rcaux_head
        in_channels = int(original.reduce.in_channels)
        projection_dim = int(original.project[0].out_features)
        assignment_scale = float(
            F.softplus(original.raw_assignment_scale.detach()).mean().item()
        )
        self.rcaux_head = PowerNormalizedResidualCodebookTextureHead(
            in_channels=in_channels,
            num_classes=int(num_classes),
            reduced_channels=original.reduced_channels,
            num_codewords=original.num_codewords,
            projection_dim=projection_dim,
            assignment_scale=assignment_scale,
            eps=original.eps,
        )


def mobilevit_xxs_rcaux_pn(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxPN:
    return MobileViTXXSRCAuxPN(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
