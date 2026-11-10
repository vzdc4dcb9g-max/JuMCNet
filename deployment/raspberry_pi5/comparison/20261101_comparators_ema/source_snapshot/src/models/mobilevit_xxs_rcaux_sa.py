from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from src.models.jumcnet import (
    JuMCNet,
    ResidualCodebookTextureHead,
)


class SaliencyAwareResidualCodebookTextureHead(
    ResidualCodebookTextureHead
):
    """Blend uniform and feature-energy weights before residual aggregation."""

    def __init__(
        self,
        *args: Any,
        saliency_mix: float = 0.50,
        saliency_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= saliency_mix <= 1.0:
            raise ValueError("saliency_mix must be in [0, 1]")
        if saliency_temperature <= 0.0:
            raise ValueError("saliency_temperature must be > 0")
        self.saliency_mix = float(saliency_mix)
        self.saliency_temperature = float(saliency_temperature)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("SA-RCAux expects a BCHW Stage3 local feature map")
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
        energy = descriptors.float().square().mean(dim=-1).sqrt()
        energy = (energy - energy.mean(dim=1, keepdim=True)) / energy.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(self.eps)
        focused = torch.softmax(
            energy / self.saliency_temperature, dim=1
        )
        count = descriptors.shape[1]
        descriptor_weights = (
            (1.0 - self.saliency_mix)
            + self.saliency_mix * focused * float(count)
        )
        encoded = (
            assignments.unsqueeze(-1)
            * descriptor_weights.unsqueeze(-1).unsqueeze(-1)
            * residuals.float()
        ).sum(dim=1)
        encoded = F.normalize(encoded, p=2, dim=-1, eps=self.eps)
        encoded = F.normalize(encoded.flatten(1), p=2, dim=-1, eps=self.eps)
        auxiliary_logits = self.classifier(self.project(encoded))
        usage = assignments.mean(dim=(0, 1))
        values = assignments.clamp_min(self.eps)
        entropy = -(values * values.log()).sum(dim=-1).mean()
        saliency_values = focused.clamp_min(self.eps)
        saliency_entropy = -(
            saliency_values * saliency_values.log()
        ).sum(dim=1).mean() / math.log(max(2, count))
        return {
            "auxiliary_logits": auxiliary_logits,
            "codebook_usage": usage,
            "assignment_entropy": entropy / math.log(self.num_codewords),
            "saliency_entropy": saliency_entropy,
        }


class MobileViTXXSRCAuxSA(JuMCNet):
    """LocalECA mainline with saliency-aware RCAux aggregation."""

    rcaux_variant = "sa"

    def __init__(
        self,
        num_classes: int,
        rcaux_saliency_mix: float = 0.50,
        rcaux_saliency_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        if self.rcaux_head is None:
            raise RuntimeError("SA-RCAux requires an active RCAux head")
        original = self.rcaux_head
        self.rcaux_head = SaliencyAwareResidualCodebookTextureHead(
            in_channels=int(original.reduce.in_channels),
            num_classes=int(num_classes),
            reduced_channels=original.reduced_channels,
            num_codewords=original.num_codewords,
            projection_dim=int(original.project[0].out_features),
            assignment_scale=float(
                F.softplus(original.raw_assignment_scale.detach()).mean().item()
            ),
            eps=original.eps,
            saliency_mix=rcaux_saliency_mix,
            saliency_temperature=rcaux_saliency_temperature,
        )


def mobilevit_xxs_rcaux_sa(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxSA:
    return MobileViTXXSRCAuxSA(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
