from __future__ import annotations

from typing import Any, Iterable

import torch

from src.models.mobilevit_xxs_odpm_cgrh_trainonly import (
    MobileViTXXSODPMCGRHTrainOnly,
)


class MobileViTXXSODPMCGRHCoverageV3(MobileViTXXSODPMCGRHTrainOnly):
    """Dynamic high-coverage pair supervision with main-head inference."""

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_odpm_cgrh_coverage_v3: bool = False,
        odpm_active_pairs: Iterable[tuple[int, int]] = (),
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if not return_odpm_cgrh_coverage_v3:
            return super().forward(
                inputs,
                return_features=return_features,
                return_aux=return_aux,
                return_odpm_cgrh=False,
            )
        if return_features or return_aux:
            raise ValueError(
                "Coverage-V3 auxiliary state cannot be combined with "
                "return_features/return_aux"
            )
        _, state = super().forward(
            inputs,
            return_odpm_cgrh=True,
            odpm_active_pairs=odpm_active_pairs,
        )
        # Training metrics and all validation/test paths use the deployable
        # main classifier.  Expert corrections are intentionally discarded.
        return state["main_logits"], state


def mobilevit_xxs_odpm_cgrh_coverage_v3(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSODPMCGRHCoverageV3:
    return MobileViTXXSODPMCGRHCoverageV3(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
