from __future__ import annotations

from typing import Any, Iterable

import torch

from src.models.mobilevit_xxs_odpm_cgrh_trainonly import (
    MobileViTXXSODPMCGRHTrainOnly,
)


class MobileViTXXSODPMCGRHHardV2(MobileViTXXSODPMCGRHTrainOnly):
    """Hard-sample ODPM-CGRH with a fully removable training branch.

    This class intentionally has its own public forward flag and model name so
    its checkpoints/output directory cannot be confused with ODPM-CGRH V1.
    The expert architecture is shared with V1; Hard-V2's selective supervision
    is implemented by its dedicated training criterion.
    """

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_odpm_cgrh_hard_v2: bool = False,
        odpm_active_pairs: Iterable[tuple[int, int]] = (),
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        return super().forward(
            inputs,
            return_features=return_features,
            return_aux=return_aux,
            return_odpm_cgrh=return_odpm_cgrh_hard_v2,
            odpm_active_pairs=odpm_active_pairs,
        )


def mobilevit_xxs_odpm_cgrh_hard_v2(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSODPMCGRHHardV2:
    return MobileViTXXSODPMCGRHHardV2(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
