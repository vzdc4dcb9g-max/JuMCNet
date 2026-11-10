from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MobileViTXXSDFN


class ConfusionGraphRefinementHead(nn.Module):
    """Conditionally refine logits for validation-derived hard class pairs.

    The main classifier remains responsible for the six-way decision. A shared
    low-rank projection feeds one tiny binary expert per hard pair. During
    inference an expert is used only when the main classifier's Top-2 classes
    match its pair and their probability margin is small. Zero-initialized
    expert classifiers make the initial refined logits identical to the main
    logits, so the module can be added without disturbing the baseline at
    initialization.
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
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not pair_indices:
            raise ValueError("CGRH requires at least one hard class pair")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if refine_scale < 0.0:
            raise ValueError("refine_scale must be >= 0")

        normalized_pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in pair_indices:
            first, second = int(first), int(second)
            if first == second:
                raise ValueError(f"CGRH pair must use two classes: {(first, second)}")
            if not (0 <= first < num_classes and 0 <= second < num_classes):
                raise ValueError(
                    f"CGRH pair {(first, second)} is outside [0, {num_classes})"
                )
            key = tuple(sorted((first, second)))
            if key in seen:
                raise ValueError(f"Duplicate unordered CGRH pair: {key}")
            seen.add(key)
            normalized_pairs.append((first, second))

        self.num_classes = int(num_classes)
        self.margin_threshold = float(margin_threshold)
        self.refine_scale = float(refine_scale)
        self.register_buffer(
            "pair_indices",
            torch.tensor(normalized_pairs, dtype=torch.long),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, projection_dim, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(projection_dropout)),
        )
        self.pair_heads = nn.ModuleList(
            nn.Linear(projection_dim, 2, bias=True)
            for _ in normalized_pairs
        )
        for head in self.pair_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

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
                forward_match = (first_top == first) & (second_top == second)
                reverse_match = (first_top == second) & (second_top == first)
                masks.append(low_margin & (forward_match | reverse_match))
            return torch.stack(masks, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        main_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = self.projection(features)
        pair_logits = torch.stack(
            [head(projected) for head in self.pair_heads], dim=1
        )
        route_mask = self._routing_mask(main_logits)

        correction = torch.zeros_like(main_logits)
        for pair_index, (first, second) in enumerate(
            self.pair_indices.tolist()
        ):
            # Only the relative binary evidence is used; common-mode bias does
            # not alter the main classifier's confidence scale.
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

        refined_logits = main_logits + correction
        state = {
            "main_logits": main_logits,
            "pair_logits": pair_logits,
            "route_mask": route_mask,
            "pair_indices": self.pair_indices,
        }
        return refined_logits, state


class MobileViTXXSCGRH(MobileViTXXSDFN):
    """MobileViT-XXS + MFF-LocalECA + CGRH decision refinement."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        cgrh_pair_indices: list[tuple[int, int]] | None = None,
        cgrh_projection_dim: int = 64,
        cgrh_projection_dropout: float = 0.0,
        cgrh_margin_threshold: float = 0.20,
        cgrh_refine_scale: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        pairs = cgrh_pair_indices or []
        self.cgrh = ConfusionGraphRefinementHead(
            in_features=int(self.num_features),
            num_classes=num_classes,
            pair_indices=pairs,
            projection_dim=int(cgrh_projection_dim),
            projection_dropout=float(cgrh_projection_dropout),
            margin_threshold=float(cgrh_margin_threshold),
            refine_scale=float(cgrh_refine_scale),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_cgrh: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[
        torch.Tensor, dict[str, torch.Tensor]
    ]:
        if return_aux:
            raise ValueError("CGRH is not combined with a global auxiliary head")
        if return_features and return_cgrh:
            raise ValueError("return_features and return_cgrh are mutually exclusive")

        features = self.forward_features(inputs)
        main_logits = self.classifier(self.dropout(features))
        refined_logits, state = self.cgrh(features, main_logits)
        if return_cgrh:
            return refined_logits, state
        if return_features:
            return refined_logits, features
        return refined_logits


def mobilevit_xxs_cgrh(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSCGRH:
    return MobileViTXXSCGRH(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
