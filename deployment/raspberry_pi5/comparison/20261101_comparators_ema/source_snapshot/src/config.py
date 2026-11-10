from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"The top level of the configuration file must be a dictionary: {config_path}")
    return data


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

