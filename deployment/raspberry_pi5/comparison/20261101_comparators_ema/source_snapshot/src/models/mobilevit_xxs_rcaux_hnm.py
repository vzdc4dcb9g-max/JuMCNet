from __future__ import annotations

from typing import Any

from src.models.jumcnet import JuMCNet


class MobileViTXXSRCAuxHNM(JuMCNet):
    """RCAux trained with an online hard-negative margin objective."""

    rcaux_variant = "hnm"


def mobilevit_xxs_rcaux_hnm(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxHNM:
    return MobileViTXXSRCAuxHNM(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
