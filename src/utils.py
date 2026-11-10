from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from src.config import project_path


def resolve_eval_with_experts(
    model_config: dict[str, Any],
    cli_value: bool | None = None,
) -> bool:
    """Whether test/benchmark should apply CGRH/LMC expert residuals.

    CLI overrides model yaml. LMC/Auto-CGRH configs may set
    ``eval_with_experts``; otherwise experts stay on.
    """
    if cli_value is not None:
        return bool(cli_value)
    lmc_config = model_config.get("lmc") or {}
    if "eval_with_experts" in lmc_config:
        return bool(lmc_config["eval_with_experts"])
    auto_config = model_config.get("auto_cgrh") or {}
    if "eval_with_experts" in auto_config:
        return bool(auto_config["eval_with_experts"])
    return True


def experiment_directory(
    dataset_config: dict[str, Any],
    experiment_config: dict[str, Any],
    model_config: dict[str, Any],
    train_seed: int,
) -> Path:
    directory = (
        project_path(experiment_config["output_root"])
        / str(dataset_config["dataset_name"])
    )
    output_namespace = str(
        experiment_config.get("output_namespace", "")
    ).strip()
    if output_namespace:
        directory = directory / output_namespace
    return directory / str(model_config["model_name"]) / f"seed_{train_seed}"


def build_optimizer(
    model: torch.nn.Module,
    experiment_config: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(experiment_config["optimizer"]).lower()
    learning_rate = float(experiment_config["learning_rate"])
    weight_decay = float(experiment_config["weight_decay"])

    # Some learnable scale/shape parameters (for example the GeM exponent and
    # its branch gate) should be optimized by the loss but not shrunk by
    # decoupled weight decay.  Models opt in by setting
    # ``parameter._exclude_from_weight_decay = True``; all unmarked parameters
    # keep the legacy optimizer behavior.
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        target = (
            no_decay_params
            if bool(getattr(parameter, "_exclude_from_weight_decay", False))
            else decay_params
        )
        target.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if decay_params:
        parameter_groups.append(
            {"params": decay_params, "weight_decay": weight_decay}
        )
    if no_decay_params:
        parameter_groups.append(
            {"params": no_decay_params, "weight_decay": 0.0}
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            lr=learning_rate,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    experiment_config: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    epochs = int(experiment_config["epochs"])
    warmup_epochs = int(experiment_config["warmup_epochs"])
    scheduler_name = str(experiment_config["scheduler"]).lower()
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {scheduler_name}")

    # Optional faithful warm-up for new reproduction profiles. Existing
    # profiles without warmup_start_factor keep the legacy schedule.
    if warmup_epochs > 0 and "warmup_start_factor" in experiment_config:
        start_factor = float(experiment_config["warmup_start_factor"])
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=0.0,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    def learning_rate_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        cosine_epochs = max(epochs - warmup_epochs, 1)
        progress = min(max(epoch - warmup_epochs, 0), cosine_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress / cosine_epochs))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_lambda
    )


def clone_state_dict_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Mean floating tensors; keep the last snapshot for integer/bool buffers."""
    if not state_dicts:
        raise ValueError("average_state_dicts() needs at least one state dict")
    averaged: dict[str, torch.Tensor] = {}
    for key, last_tensor in state_dicts[-1].items():
        tensors = [item[key] for item in state_dicts]
        if last_tensor.is_floating_point():
            stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=last_tensor.dtype)
        else:
            averaged[key] = last_tensor.clone()
    return averaged


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    configs: dict[str, Any],
    weight_source: str = "raw",
    criterion: torch.nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_metric": best_metric,
        "weight_source": str(weight_source),
        "configs": configs,
    }
    if criterion is not None:
        checkpoint["criterion_state"] = criterion.state_dict()
        extra_state = getattr(criterion, "extra_state", None)
        if callable(extra_state):
            checkpoint["criterion_extra_state"] = extra_state()
    torch.save(checkpoint, path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
