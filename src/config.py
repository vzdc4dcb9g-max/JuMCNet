from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"The top-level configuration must be a dictionary: {config_path}")
    return data


def get_display_class_names(dataset_config: dict[str, Any]) -> list[str]:
    names = [str(name) for name in dataset_config["class_names"]]
    display_names = dataset_config.get("class_names_en", {})
    return [str(display_names.get(name, name)) for name in names]


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_all_configs(
    dataset_config: str | Path,
    experiment_config: str | Path,
    model_config: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_yaml(dataset_config),
        load_yaml(experiment_config),
        load_yaml(model_config),
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resolved-config",
        default=None,
        help="Load a saved full configuration and resolve its paths against the current project",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Override the training seed; defaults to the saved seed with a full configuration, or 2222 with separate dataset, experiment, and model configurations",
    )


def _local_path(value: str, anchor: str, fallback: Path | None = None) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    for index, part in enumerate(parts):
        if part.casefold() == anchor.casefold():
            tail = parts[index + 1:]
            if ".." in tail:
                raise ValueError(f"Project paths must not contain parent-directory components: {value}")
            return str(PROJECT_ROOT / anchor / Path(*tail))
    if fallback is not None:
        return str(PROJECT_ROOT / fallback)
    raise ValueError(f"Cannot resolve the path under the current project's {anchor} directory: {value}")


def load_runtime_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    paths = (args.dataset_config, args.experiment_config, args.model_config)
    if args.resolved_config is None:
        dataset, experiment, model = load_all_configs(
            paths[0] or "configs/datasets/jujube12000.yaml",
            paths[1] or "configs/experiments/default.yaml",
            paths[2] or "configs/models/mobilenetv2.yaml",
        )
        train_seed = 2222 if args.train_seed is None else args.train_seed
        return dataset, experiment, model, train_seed

    if any(path is not None for path in paths):
        raise ValueError(
            "--resolved-config cannot be combined with --dataset-config, --experiment-config"
            " or --model-config"
        )
    snapshot = load_yaml(args.resolved_config)
    for key in ("dataset", "experiment", "model"):
        if not isinstance(snapshot.get(key), dict):
            raise ValueError(f"The full configuration is missing a dictionary field: {key}")
    saved_seed = snapshot.get("train_seed")
    if not isinstance(saved_seed, int) or isinstance(saved_seed, bool):
        raise ValueError("The full configuration must contain an integer train_seed")
    dataset, experiment, model = (
        snapshot["dataset"], snapshot["experiment"], snapshot["model"]
    )
    split = dataset.get("split")
    if not isinstance(split, dict) or "seed" not in split:
        raise ValueError("The full configuration is missing dataset.split.seed")
    dataset_name = str(dataset["dataset_name"])
    data_defaults = {
        "Jujube12000": Path("dataSets/Jujube12000"),
        "Cassava": Path("dataSets/Cassava/train_images"),
    }
    dataset["data_root"] = _local_path(
        dataset["data_root"], "dataSets", data_defaults.get(dataset_name)
    )
    split["output_dir"] = _local_path(
        split["output_dir"],
        "splits",
        Path("splits") / dataset_name / f"split_seed_{split['seed']}",
    )
    experiment["output_root"] = str(PROJECT_ROOT / "runs")
    experiment["output_namespace"] = ""
    train_seed = saved_seed if args.train_seed is None else args.train_seed
    return dataset, experiment, model, train_seed


def resolve_checkpoint(checkpoint: str | None, run_dir: Path) -> Path:
    path = (
        run_dir / "checkpoints" / "best.pth"
        if checkpoint is None
        else project_path(checkpoint)
    )
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path

