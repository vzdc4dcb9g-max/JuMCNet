from __future__ import annotations

from typing import Any

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class MobileViTXXSHeadDrop(MobileViTXXSDFN):
    """LocalECA with dropout only between final GAP and classifier."""

    def __init__(self, *args: Any, head_dropout: float = 0.10, **kwargs: Any):
        super().__init__(
            *args,
            head_dropout=float(head_dropout),
            **kwargs,
        )


def mobilevit_xxs_headdrop(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    head_dropout: float = 0.10,
    **kwargs: Any,
) -> MobileViTXXSHeadDrop:
    return MobileViTXXSHeadDrop(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        head_dropout=head_dropout,
        **kwargs,
    )
