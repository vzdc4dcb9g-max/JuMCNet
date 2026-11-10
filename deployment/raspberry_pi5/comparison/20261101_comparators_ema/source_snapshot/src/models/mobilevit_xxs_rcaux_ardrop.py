from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from src.models.jumcnet import (
    JuMCNet,
    ResidualCodebookTextureHead,
)


class AuxiliaryRDropResidualCodebookTextureHead(
    ResidualCodebookTextureHead
):
    """Repeat only the tiny auxiliary projection/classifier under dropout."""

    def __init__(
        self,
        *args: Any,
        auxiliary_dropout: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= auxiliary_dropout < 1.0:
            raise ValueError("auxiliary_dropout must be in [0, 1)")
        self.auxiliary_dropout = float(auxiliary_dropout)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("ARDrop-RCAux expects BCHW Stage3 features")

        # Rebuild the compact descriptor once.  The backbone and residual
        # encoding are not repeated; only projection/classification receive
        # two independent dropout masks.
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
        encoded = (assignments.unsqueeze(-1) * residuals.float()).sum(dim=1)
        encoded = F.normalize(encoded, p=2, dim=-1, eps=self.eps)
        encoded = F.normalize(encoded.flatten(1), p=2, dim=-1, eps=self.eps)
        projected = self.project(encoded)
        first = self.classifier(
            F.dropout(
                projected,
                p=self.auxiliary_dropout,
                training=self.training,
            )
        )
        second = self.classifier(
            F.dropout(
                projected,
                p=self.auxiliary_dropout,
                training=self.training,
            )
        )
        usage = assignments.mean(dim=(0, 1))
        values = assignments.clamp_min(self.eps)
        entropy = -(values * values.log()).sum(dim=-1).mean()
        return {
            "auxiliary_logits": first,
            "auxiliary_logits_2": second,
            "codebook_usage": usage,
            "assignment_entropy": entropy / math.log(self.num_codewords),
        }


class MobileViTXXSRCAuxARDrop(JuMCNet):
    """LocalECA mainline with a removable auxiliary-only R-Drop head."""

    rcaux_variant = "ardrop"

    def __init__(
        self,
        num_classes: int,
        rcaux_auxiliary_dropout: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        if self.rcaux_head is None:
            raise RuntimeError("ARDrop-RCAux requires an active RCAux head")
        original = self.rcaux_head
        self.rcaux_head = AuxiliaryRDropResidualCodebookTextureHead(
            in_channels=int(original.reduce.in_channels),
            num_classes=int(num_classes),
            reduced_channels=original.reduced_channels,
            num_codewords=original.num_codewords,
            projection_dim=int(original.project[0].out_features),
            assignment_scale=float(
                F.softplus(original.raw_assignment_scale.detach()).mean().item()
            ),
            eps=original.eps,
            auxiliary_dropout=rcaux_auxiliary_dropout,
        )


def mobilevit_xxs_rcaux_ardrop(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxARDrop:
    return MobileViTXXSRCAuxARDrop(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
