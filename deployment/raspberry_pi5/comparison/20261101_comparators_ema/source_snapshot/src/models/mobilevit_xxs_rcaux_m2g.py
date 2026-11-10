from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.jumcnet import (
    JuMCNet,
    ResidualCodebookTextureHead,
)


class GatedSecondMomentResidualCodebookTextureHead(
    ResidualCodebookTextureHead
):
    """RCAux first moment plus a bounded, near-zero second-moment branch."""

    def __init__(
        self,
        *args: Any,
        second_order_dim: int = 16,
        max_second_order_gate: float = 0.30,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if second_order_dim < 1:
            raise ValueError("second_order_dim must be >= 1")
        if not 0.0 < max_second_order_gate <= 1.0:
            raise ValueError("max_second_order_gate must be in (0, 1]")
        projection_dim = int(self.project[0].out_features)
        input_dim = self.num_codewords * self.reduced_channels
        self.second_project = nn.Sequential(
            nn.Linear(input_dim, int(second_order_dim)),
            nn.LayerNorm(int(second_order_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(int(second_order_dim), projection_dim),
        )
        self.raw_second_order_gate = nn.Parameter(torch.tensor(-8.0))
        self.max_second_order_gate = float(max_second_order_gate)
        self._reset_second_order()

    def _reset_second_order(self) -> None:
        for module in self.second_project:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("M2G-RCAux expects BCHW Stage3 features")
        descriptors = self.reduce(features).flatten(2).transpose(1, 2)
        residuals = descriptors.unsqueeze(2) - self.codewords.view(
            1, 1, self.num_codewords, self.reduced_channels
        )
        distance = residuals.float().square().sum(dim=-1)
        scales = F.softplus(self.raw_assignment_scale.float()).clamp_min(
            self.eps
        )
        assignments = torch.softmax(
            -distance * scales.view(1, 1, -1), dim=-1
        )
        weights = assignments.unsqueeze(-1)
        first = (weights * residuals.float()).sum(dim=1)
        first = F.normalize(first, p=2, dim=-1, eps=self.eps)
        first = F.normalize(first.flatten(1), p=2, dim=-1, eps=self.eps)
        denominator = assignments.sum(dim=1).unsqueeze(-1).clamp_min(self.eps)
        second = (weights * residuals.float().square()).sum(dim=1) / denominator
        second = torch.sqrt(second.clamp_min(self.eps))
        second = F.normalize(second, p=2, dim=-1, eps=self.eps)
        second = F.normalize(second.flatten(1), p=2, dim=-1, eps=self.eps)
        gate = (
            torch.sigmoid(self.raw_second_order_gate)
            * self.max_second_order_gate
        )
        projected = self.project(first) + gate * self.second_project(second)
        auxiliary_logits = self.classifier(projected)
        usage = assignments.mean(dim=(0, 1))
        values = assignments.clamp_min(self.eps)
        entropy = -(values * values.log()).sum(dim=-1).mean()
        return {
            "auxiliary_logits": auxiliary_logits,
            "codebook_usage": usage,
            "assignment_entropy": entropy / math.log(self.num_codewords),
            "second_order_gate": gate,
        }


class MobileViTXXSRCAuxM2G(JuMCNet):
    """LocalECA mainline with a gated second-moment RCAux head."""

    rcaux_variant = "m2g"

    def __init__(
        self,
        num_classes: int,
        rcaux_second_order_dim: int = 16,
        rcaux_max_second_order_gate: float = 0.30,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        if self.rcaux_head is None:
            raise RuntimeError("M2G-RCAux requires an active RCAux head")
        original = self.rcaux_head
        self.rcaux_head = GatedSecondMomentResidualCodebookTextureHead(
            in_channels=int(original.reduce.in_channels),
            num_classes=int(num_classes),
            reduced_channels=original.reduced_channels,
            num_codewords=original.num_codewords,
            projection_dim=int(original.project[0].out_features),
            assignment_scale=float(
                F.softplus(original.raw_assignment_scale.detach()).mean().item()
            ),
            eps=original.eps,
            second_order_dim=rcaux_second_order_dim,
            max_second_order_gate=rcaux_max_second_order_gate,
        )


def mobilevit_xxs_rcaux_m2g(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxM2G:
    return MobileViTXXSRCAuxM2G(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
