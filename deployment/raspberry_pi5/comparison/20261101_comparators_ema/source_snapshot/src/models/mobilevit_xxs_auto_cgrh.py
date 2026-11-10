from __future__ import annotations

from typing import Any, Iterable

import torch

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN
from src.models.mobilevit_xxs_odpm_cgrh_trainonly import (
    OnlineDynamicPairCGRHTrainingHead,
)


class MobileViTXXSAutoCGRH(MobileViTXXSDFN):
    """MFF-LocalECA backbone with automatically frozen CGRH experts.

    Every possible binary expert is allocated before the optimizer is built.
    An external training criterion discovers a small pair set from EMA-teacher
    predictions and synchronizes it through :meth:`set_auto_cgrh_pairs`.
    The selected experts remain active for validation, testing, and deployment.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        auto_cgrh_projection_dim: int = 64,
        auto_cgrh_projection_dropout: float = 0.0,
        auto_cgrh_margin_threshold: float = 0.20,
        auto_cgrh_refine_scale: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.cgrh = OnlineDynamicPairCGRHTrainingHead(
            in_features=int(self.num_features),
            num_classes=int(num_classes),
            projection_dim=int(auto_cgrh_projection_dim),
            projection_dropout=float(auto_cgrh_projection_dropout),
            margin_threshold=float(auto_cgrh_margin_threshold),
            refine_scale=float(auto_cgrh_refine_scale),
        )
        self.register_buffer(
            "auto_cgrh_active_mask",
            torch.zeros(len(self.cgrh.pair_heads), dtype=torch.bool),
            persistent=True,
        )
        # Test/deploy toggle: False → score main head only (experts idle).
        self.inference_use_experts = True
        self.inference_auxiliary_stripped = False

    def set_inference_experts_enabled(self, enabled: bool) -> None:
        """Enable/disable expert residual at eval (routing still low-margin)."""
        self.inference_use_experts = bool(enabled)
        self.inference_auxiliary_stripped = not bool(enabled)

    def prepare_for_inference(self) -> None:
        """Keep experts unless :meth:`set_inference_experts_enabled` disabled them."""
        self.inference_auxiliary_stripped = not bool(
            self.inference_use_experts
        )

    def set_auto_cgrh_pairs(
        self,
        pairs: Iterable[tuple[int, int]],
    ) -> None:
        """Persist the active unordered pair set in the model checkpoint."""
        mask = torch.zeros_like(self.auto_cgrh_active_mask)
        for first, second in pairs:
            key = tuple(sorted((int(first), int(second))))
            slot = self.cgrh._pair_to_slot.get(key)
            if slot is None:
                raise ValueError(f"Auto-CGRH active pair is invalid: {key}")
            mask[slot] = True
        self.auto_cgrh_active_mask.copy_(mask)

    def auto_cgrh_pairs(self) -> list[tuple[int, int]]:
        indices = self.cgrh.pair_indices[self.auto_cgrh_active_mask]
        return [
            (int(first), int(second))
            for first, second in indices.detach().cpu().tolist()
        ]

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_cgrh: bool = False,
        return_main_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("Auto-CGRH is not combined with a global aux head")
        exclusive = sum(
            int(flag)
            for flag in (return_features, return_cgrh, return_main_logits)
        )
        if exclusive > 1:
            raise ValueError(
                "return_features, return_cgrh, and return_main_logits "
                "are mutually exclusive"
            )

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        if return_main_logits:
            return main_logits

        active_pairs = self.auto_cgrh_pairs()
        if not bool(getattr(self, "inference_use_experts", True)):
            active_pairs = []
        if active_pairs:
            refined_logits, state = self.cgrh(
                features,
                main_logits,
                active_pairs,
            )
        else:
            batch_size = int(main_logits.shape[0])
            pair_count = int(self.cgrh.pair_indices.shape[0])
            refined_logits = main_logits
            state = {
                "main_logits": main_logits,
                "pair_logits": main_logits.new_zeros(
                    (batch_size, pair_count, 2)
                ),
                "pair_indices": self.cgrh.pair_indices,
                "active_pair_mask": self.auto_cgrh_active_mask,
                "route_mask": torch.zeros(
                    (batch_size, pair_count),
                    dtype=torch.bool,
                    device=main_logits.device,
                ),
                "correction": torch.zeros_like(main_logits),
            }

        if return_cgrh:
            return refined_logits, state
        if return_features:
            return refined_logits, features
        return refined_logits


def mobilevit_xxs_auto_cgrh(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSAutoCGRH:
    return MobileViTXXSAutoCGRH(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
