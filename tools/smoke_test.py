from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_all_configs  # noqa: E402
from src.datasets import create_dataloaders  # noqa: E402
from src.models import create_model  # noqa: E402
from src.reproducibility import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check data loading and model forward passes")
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/jujube12000.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/default.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/mobilenetv2.yaml",
    )
    parser.add_argument("--train-seed", type=int, default=2222)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config, experiment_config, model_config = load_all_configs(
        args.dataset_config,
        args.experiment_config,
        args.model_config,
    )
    set_seed(args.train_seed, deterministic=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no compatible GPU is available in the current environment.")
    loaders = create_dataloaders(
        dataset_config,
        experiment_config,
        train_seed=args.train_seed,
        include_test=True,
        model_config=model_config,
    )
    model = create_model(
        model_config,
        num_classes=int(dataset_config["num_classes"]),
        class_names=[str(name) for name in dataset_config["class_names"]],
    ).to(device)
    model.eval()

    images, labels, paths = next(iter(loaders["train"]))
    images = images.to(device)
    with torch.inference_mode():
        logits = model(images)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model={model_config['model_name']}")
    print(f"parameters={parameter_count}")
    print(f"trainable_parameters={trainable_count}")
    print(
        "split_sizes="
        + ", ".join(
            f"{name}:{len(loader.dataset)}" for name, loader in loaders.items()
        )
    )
    print(f"input_shape={tuple(images.shape)}")
    print(f"label_shape={tuple(labels.shape)}")
    print(f"logit_shape={tuple(logits.shape)}")
    print(f"sample_path={paths[0]}")
    expected_shape = (images.shape[0], int(dataset_config["num_classes"]))
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(logits.shape)} != {expected_shape}"
        )
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
