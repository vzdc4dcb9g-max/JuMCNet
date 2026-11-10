from __future__ import annotations

from typing import Any

import torch


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    targets = targets.to(torch.int64).cpu()
    predictions = predictions.to(torch.int64).cpu()
    valid = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[valid] + predictions[valid]
    return torch.bincount(
        indices, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(matrix: torch.Tensor) -> dict[str, Any]:
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)

    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1.0)

    return {
        "accuracy": float(accuracy.item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.to(torch.int64).tolist(),
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


class ClassificationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )
        self.loss_sum = 0.0
        self.sample_count = 0

    def update(
        self,
        loss: float,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        batch_size = int(targets.numel())
        self.loss_sum += float(loss) * batch_size
        self.sample_count += batch_size
        self.matrix += confusion_matrix(
            targets, predictions, self.num_classes
        )

    def compute(self) -> dict[str, Any]:
        result = metrics_from_confusion_matrix(self.matrix)
        result["loss"] = self.loss_sum / max(self.sample_count, 1)
        return result

