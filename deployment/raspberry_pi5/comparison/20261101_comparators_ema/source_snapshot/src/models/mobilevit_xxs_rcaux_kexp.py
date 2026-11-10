from __future__ import annotations

from typing import Any

from src.models.jumcnet import JuMCNet


class MobileViTXXSRCAuxKExp(JuMCNet):
    """Capacity-matched RCAux codebook exploration (K=6 by default)."""

    rcaux_variant = "kexp"

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        rcaux_reduced_channels: int = 40,
        rcaux_num_codewords: int = 6,
        rcaux_projection_dim: int = 56,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            rcaux_reduced_channels=rcaux_reduced_channels,
            rcaux_num_codewords=rcaux_num_codewords,
            rcaux_projection_dim=rcaux_projection_dim,
            **kwargs,
        )


def mobilevit_xxs_rcaux_kexp(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSRCAuxKExp:
    return MobileViTXXSRCAuxKExp(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
