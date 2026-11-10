from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class DynamicPairwiseHardnessGuidanceHead(nn.Module):
    """Training-only dynamic guidance for uncertain Top-2 class pairs.

    The pair universe is fixed, but the active expert is selected per sample
    from the current main-head Top-2 prediction.  Only low-margin samples whose
    target belongs to that Top-2 pair contribute pair/refinement supervision.
    This avoids teaching an expert from a pair that does not contain the true
    class.  The complete head can be removed for validation/test/deployment.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pair_indices: list[tuple[int, int]],
        projection_dim: int = 64,
        projection_dropout: float = 0.0,
        margin_threshold: float = 0.20,
        refine_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if in_features < 1 or projection_dim < 1:
            raise ValueError("in_features and projection_dim must be >= 1")
        if num_classes < 2 or not pair_indices:
            raise ValueError("DPHG requires at least two classes and one pair")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if refine_scale < 0.0:
            raise ValueError("refine_scale must be >= 0")

        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in pair_indices:
            first, second = int(first), int(second)
            key = tuple(sorted((first, second)))
            if first == second or not (
                0 <= first < num_classes and 0 <= second < num_classes
            ):
                raise ValueError(f"Invalid DPHG pair: {(first, second)}")
            if key in seen:
                raise ValueError(f"Duplicate unordered DPHG pair: {key}")
            seen.add(key)
            normalized.append((first, second))

        self.num_classes = int(num_classes)
        self.num_pairs = len(normalized)
        self.margin_threshold = float(margin_threshold)
        self.refine_scale = float(refine_scale)
        self.register_buffer(
            "pair_indices",
            torch.tensor(normalized, dtype=torch.long),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, projection_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        # A vectorized classifier is materially faster than 15 Python-level
        # Linear calls during training and is removed entirely for inference.
        self.classifier = nn.Linear(projection_dim, self.num_pairs * 2)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _routing_mask(self, main_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probabilities = torch.softmax(main_logits.float(), dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            low_margin = (
                top_probabilities[:, 0] - top_probabilities[:, 1]
                <= self.margin_threshold
            )
            first_top = top_indices[:, 0]
            second_top = top_indices[:, 1]
            masks: list[torch.Tensor] = []
            for first, second in self.pair_indices.tolist():
                matched = (
                    ((first_top == first) & (second_top == second))
                    | ((first_top == second) & (second_top == first))
                )
                masks.append(low_margin & matched)
            return torch.stack(masks, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        main_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_logits = self.classifier(self.projection(features)).reshape(
            features.shape[0], self.num_pairs, 2
        )
        route_mask = self._routing_mask(main_logits)

        correction = torch.zeros_like(main_logits)
        for pair_index, (first, second) in enumerate(
            self.pair_indices.tolist()
        ):
            centered = pair_logits[:, pair_index]
            centered = centered - centered.mean(dim=1, keepdim=True)
            active = route_mask[:, pair_index].to(main_logits.dtype).unsqueeze(1)
            first_basis = F.one_hot(
                torch.full(
                    (main_logits.shape[0],),
                    first,
                    device=main_logits.device,
                    dtype=torch.long,
                ),
                num_classes=self.num_classes,
            ).to(main_logits.dtype)
            second_basis = F.one_hot(
                torch.full(
                    (main_logits.shape[0],),
                    second,
                    device=main_logits.device,
                    dtype=torch.long,
                ),
                num_classes=self.num_classes,
            ).to(main_logits.dtype)
            correction = correction + self.refine_scale * active * (
                centered[:, 0:1] * first_basis
                + centered[:, 1:2] * second_basis
            )

        return {
            "refined_logits": main_logits + correction,
            "pair_logits": pair_logits,
            "route_mask": route_mask,
            "pair_indices": self.pair_indices,
        }


class MobileViTXXSDPHG(MobileViTXXSDFN):
    """MFF-LocalECA backbone with removable training-time DPHG."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        dphg_pair_indices: list[tuple[int, int]] | None = None,
        dphg_projection_dim: int = 64,
        dphg_projection_dropout: float = 0.0,
        dphg_margin_threshold: float = 0.20,
        dphg_refine_scale: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        self.dphg: DynamicPairwiseHardnessGuidanceHead | None = (
            DynamicPairwiseHardnessGuidanceHead(
                in_features=int(self.num_features),
                num_classes=num_classes,
                pair_indices=dphg_pair_indices or [],
                projection_dim=int(dphg_projection_dim),
                projection_dropout=float(dphg_projection_dropout),
                margin_threshold=float(dphg_margin_threshold),
                refine_scale=float(dphg_refine_scale),
            )
        )
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSDPHG:
        self.dphg = None
        self.inference_auxiliary_stripped = True
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_dphg: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("DPHG is not combined with a global auxiliary head")
        if return_features and return_dphg:
            raise ValueError("return_features and return_dphg are mutually exclusive")

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        if return_dphg:
            if self.dphg is None:
                raise RuntimeError("DPHG head was stripped for inference")
            return main_logits, self.dphg(features, main_logits)
        if return_features:
            return main_logits, features
        # Validation, test, export and deployment use only the main head.
        return main_logits


def mobilevit_xxs_dphg(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSDPHG:
    return MobileViTXXSDPHG(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
