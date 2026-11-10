from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import (
    add_config_arguments, get_display_class_names, load_runtime_configs,
    resolve_checkpoint,
)
from src.datasets import build_transforms
from src.models import create_model
from src.utils import experiment_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single image")
    parser.add_argument("image")
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config, experiment_config, model_config, args.train_seed = (
        load_runtime_configs(args)
    )
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = experiment_directory(
        dataset_config,
        experiment_config,
        model_config,
        args.train_seed,
    )
    checkpoint_path = resolve_checkpoint(args.checkpoint, output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_config,
        num_classes=int(dataset_config["num_classes"]),
        class_names=[str(name) for name in dataset_config["class_names"]],
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, evaluation_transform = build_transforms(
        dataset_config, experiment_config
    )
    with Image.open(image_path) as image:
        tensor = evaluation_transform(image.convert("RGB")).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    topk = min(args.topk, probabilities.numel())
    scores, indices = probabilities.topk(topk)
    class_names = get_display_class_names(dataset_config)
    for rank, (score, index) in enumerate(
        zip(scores.tolist(), indices.tolist()), start=1
    ):
        print(f"{rank}. {class_names[index]}: {score:.4%}")


if __name__ == "__main__":
    main()
