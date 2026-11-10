from __future__ import annotations

from typing import Any

from src.models.jumcnet import JuMCNet


class MobileViTXXSRCAuxPAW(JuMCNet):
    """RCAux with a peak-aligned auxiliary-weight schedule in its config."""

    rcaux_variant = "paw"


def mobilevit_xxs_rcaux_paw(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxPAW:
    return MobileViTXXSRCAuxPAW(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
