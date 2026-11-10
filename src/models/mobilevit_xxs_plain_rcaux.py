from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.jumcnet import ResidualCodebookTextureHead


class MobileViTXXSPlainRCAux(nn.Module):
    """Unmodified timm MobileViT-XXS with a removable Stage-3 RCAux head.

    The main path is exactly timm's ``mobilevit_xxs``. During RCAux training,
    a temporary hook observes the Stage-3 MobileViT block's ``conv_1x1``
    output, i.e. the local descriptor map immediately before patch unfolding.
    No MFF, LocalECA, or other experimental module is inserted in the backbone.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        rcaux_reduced_channels: int = 48,
        rcaux_num_codewords: int = 4,
        rcaux_projection_dim: int = 64,
        rcaux_assignment_scale: float = 1.0,
        architecture_name: str = "mobilevit_xxs",
        **_: Any,
    ) -> None:
        super().__init__()
        import timm

        if str(architecture_name) != "mobilevit_xxs":
            raise ValueError(
                "MobileViTXXSPlainRCAux requires architecture_name='mobilevit_xxs'"
            )
        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=bool(pretrained),
            num_classes=int(num_classes),
            drop_rate=float(dropout),
        )
        block = self._stage3_mobilevit_block()
        transformer_channels = int(block.conv_1x1.out_channels)
        self.rcaux_head: ResidualCodebookTextureHead | None = (
            ResidualCodebookTextureHead(
                in_channels=transformer_channels,
                num_classes=int(num_classes),
                reduced_channels=int(rcaux_reduced_channels),
                num_codewords=int(rcaux_num_codewords),
                projection_dim=int(rcaux_projection_dim),
                assignment_scale=float(rcaux_assignment_scale),
            )
        )
        self.inference_auxiliary_stripped = False

    def _stage3_mobilevit_block(self) -> nn.Module:
        candidates = [
            module
            for module in self.backbone.stages[3]
            if all(
                hasattr(module, name)
                for name in ("conv_kxk", "conv_1x1", "transformer")
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Expected exactly one standard MobileViT block in Stage 3; "
                f"found {len(candidates)}"
            )
        return candidates[0]

    def prepare_for_inference(self) -> MobileViTXXSPlainRCAux:
        """Remove every RCAux-only parameter; the backbone remains unchanged."""
        self.rcaux_head = None
        self.inference_auxiliary_stripped = True
        return self

    def _forward_rcaux(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.rcaux_head is None:
            raise RuntimeError("RCAux training head was stripped for inference")

        captured: list[torch.Tensor] = []

        def capture_local(
            _module: nn.Module,
            _module_inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured.append(output)

        handle = self._stage3_mobilevit_block().conv_1x1.register_forward_hook(
            capture_local
        )
        try:
            logits = self.backbone(inputs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                "Stage-3 local feature capture failed: "
                f"expected one tensor, got {len(captured)}"
            )
        return logits, self.rcaux_head(captured[0])

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_rcaux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("RCAux is not combined with a global auxiliary head")
        if return_features and return_rcaux:
            raise ValueError("return_features and return_rcaux are exclusive")
        if return_rcaux:
            return self._forward_rcaux(inputs)
        if return_features:
            features = self.backbone.forward_features(inputs)
            logits = self.backbone.forward_head(features)
            return logits, features
        return self.backbone(inputs)


def mobilevit_xxs_plain_rcaux(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSPlainRCAux:
    return MobileViTXXSPlainRCAux(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
