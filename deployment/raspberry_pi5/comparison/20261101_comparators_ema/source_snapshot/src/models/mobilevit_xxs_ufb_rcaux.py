from __future__ import annotations

from typing import Any

from src.models.jumcnet import JuMCNet


class MobileViTXXSUFBRCAux(JuMCNet):
    """RCAux with usage-floor training and codebook optimizer hygiene.

    The architecture and deployed forward path are intentionally identical to
    RCAux.  The usage-floor term is implemented by the training criterion;
    this class only keeps the codebook centers and assignment scales out of
    AdamW decay so that clustering geometry is not pulled toward zero.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        if self.rcaux_head is None:
            raise RuntimeError("UFB-RCAux requires an active RCAux head")
        self.rcaux_head.codewords._exclude_from_weight_decay = True
        self.rcaux_head.raw_assignment_scale._exclude_from_weight_decay = True


def mobilevit_xxs_ufb_rcaux(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSUFBRCAux:
    return MobileViTXXSUFBRCAux(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
