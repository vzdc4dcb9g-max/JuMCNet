from __future__ import annotations

import argparse
import csv

import torch
from torch import nn

from src.config import (
    add_config_arguments, get_display_class_names, load_runtime_configs,
    project_path, resolve_checkpoint,
)
from src.datasets import create_dataloaders
from src.engine import evaluate
from src.models import create_model
from src.reproducibility import set_seed
from src.utils import (
    experiment_directory,
    resolve_eval_with_experts,
    write_json,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the held-out test set")
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None, help="Optional directory for test results; does not change the checkpoint source")
    expert_group = parser.add_mutually_exclusive_group()
    expert_group.add_argument(
        "--eval-with-experts",
        dest="eval_with_experts",
        action="store_true",
        default=None,
        help="Force expert-assisted test metrics (overrides model config)",
    )
    expert_group.add_argument(
        "--eval-without-experts",
        dest="eval_with_experts",
        action="store_false",
        help="Force main-head-only test metrics (overrides model config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config, experiment_config, model_config, args.train_seed = (
        load_runtime_configs(args)
    )
    set_seed(args.train_seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(experiment_config["amp"]) and device.type == "cuda"
    num_classes = int(dataset_config["num_classes"])
    class_names = list(dataset_config["class_names"])
    display_class_names = get_display_class_names(dataset_config)

    output_dir = experiment_directory(
        dataset_config,
        experiment_config,
        model_config,
        args.train_seed,
    )
    checkpoint_path = resolve_checkpoint(args.checkpoint, output_dir)

    loaders = create_dataloaders(
        dataset_config,
        experiment_config,
        train_seed=args.train_seed,
        include_test=True,
    )
    model = create_model(
        model_config,
        num_classes=num_classes,
        class_names=class_names,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu" if device.type == "cpu" else device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    eval_with_experts = resolve_eval_with_experts(
        model_config, args.eval_with_experts
    )
    set_experts = getattr(model, "set_inference_experts_enabled", None)
    if callable(set_experts):
        set_experts(eval_with_experts)
    prepare_for_inference = getattr(model, "prepare_for_inference", None)
    if callable(prepare_for_inference):
        prepare_for_inference()
    criterion = nn.CrossEntropyLoss()

    metrics, predictions = evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        amp=amp_enabled,
        collect_predictions=True,
    )
    metrics["class_names"] = display_class_names
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["train_seed"] = args.train_seed
    metrics["eval_with_experts"] = bool(eval_with_experts)
    metrics["training_auxiliary_stripped"] = bool(
        getattr(model, "inference_auxiliary_stripped", False)
    )

    for row in predictions:
        row["target_name"] = display_class_names[row["target"]]
        row["prediction_name"] = display_class_names[row["prediction"]]
        if "main_prediction" in row:
            row["main_prediction_name"] = display_class_names[
                row["main_prediction"]
            ]

    if args.output_dir is not None:
        output_dir = project_path(args.output_dir)
    write_json(output_dir / "test_metrics.json", metrics)
    write_predictions(output_dir / "test_predictions.csv", predictions)

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *display_class_names])
        for class_name, row in zip(
            display_class_names, metrics["confusion_matrix"]
        ):
            writer.writerow([class_name, *row])

    print(f"checkpoint={checkpoint_path}")
    print(f"test_samples={len(loaders['test'].dataset)}")
    print(f"eval_with_experts={eval_with_experts}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_precision={metrics['macro_precision']:.6f}")
    print(f"macro_recall={metrics['macro_recall']:.6f}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    if "cgrh_route_rate" in metrics:
        print(f"cgrh_main_accuracy={metrics['cgrh_main_accuracy']:.6f}")
        print(f"cgrh_main_macro_f1={metrics['cgrh_main_macro_f1']:.6f}")
        print(f"cgrh_route_rate={metrics['cgrh_route_rate']:.6f}")
        print(
            "cgrh_changed_predictions="
            f"{metrics['cgrh_changed_predictions']}"
        )
        print(f"cgrh_corrected_errors={metrics['cgrh_corrected_errors']}")
        print(f"cgrh_harmed_correct={metrics['cgrh_harmed_correct']}")
        print(
            "cgrh_pair_route_counts="
            f"{metrics['cgrh_pair_route_counts']}"
        )
    print(f"Results directory={output_dir}")


if __name__ == "__main__":
    main()
