from __future__ import annotations

from typing import Any

from src.models.jumcnet import JuMCNet


class MobileViTXXSRCAuxMHA(JuMCNet):
    """RCAux trained with main-head-aligned soft targets.

    The architecture is intentionally identical to RCAux.  The dedicated
    model type and config select the MHA criterion while keeping this variant
    independently removable from the experiment registry.
    """

    rcaux_variant = "mha"


def mobilevit_xxs_rcaux_mha(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxMHA:
    return MobileViTXXSRCAuxMHA(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
