from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from statistics import mean, stdev
import time
from typing import Any

import torch

from src.config import (
    add_config_arguments, load_runtime_configs, project_path, resolve_checkpoint,
)
from src.models import create_model
from src.utils import experiment_directory, resolve_eval_with_experts, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure classification model parameters, MACs/FLOPs, and FPS"
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="By default, load checkpoints/best.pth from the corresponding experiment directory",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="CUDA only; disabled by default to keep measurements in FP32",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path; defaults to the model experiment directory",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "warmup", "repeats", "rounds"):
        value = int(getattr(args, name))
        minimum = 0 if name == "warmup" else 1
        if value < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {minimum}")


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


@torch.inference_mode()
def run_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    iterations: int,
    device: torch.device,
    precision: str,
) -> None:
    with autocast_context(device, precision):
        for _ in range(iterations):
            model(inputs)


def measure_fps(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    precision: str,
    warmup: int,
    repeats: int,
    rounds: int,
) -> tuple[list[dict[str, float]], float]:
    if warmup:
        run_forward(model, inputs, warmup, device, precision)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, float]] = []
    for round_index in range(1, rounds + 1):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()
            run_forward(model, inputs, repeats, device, precision)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            start_time = time.perf_counter()
            run_forward(model, inputs, repeats, device, precision)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        images = inputs.shape[0] * repeats
        fps = images / (elapsed_ms / 1000.0)
        rows.append(
            {
                "round": round_index,
                "elapsed_ms": elapsed_ms,
                "latency_ms_per_batch": elapsed_ms / repeats,
                "latency_ms_per_image": elapsed_ms / images,
                "fps": fps,
            }
        )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return rows, peak_memory_mb


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    try:
        from thop import profile
    except ImportError as error:
        raise SystemExit(
            "THOP is not installed. Run: python -m pip install thop"
        ) from error

    dataset_config, experiment_config, model_config, args.train_seed = (
        load_runtime_configs(args)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "fp16" and device.type != "cuda":
        raise RuntimeError("FP16 benchmarking requires a CUDA GPU")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    num_classes = int(dataset_config["num_classes"])
    image_size = int(dataset_config["image"]["input_size"])
    output_dir = experiment_directory(
        dataset_config,
        experiment_config,
        model_config,
        args.train_seed,
    )
    checkpoint_path = resolve_checkpoint(args.checkpoint, output_dir)

    model = create_model(
        model_config,
        num_classes=num_classes,
        class_names=[str(name) for name in dataset_config["class_names"]],
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu" if device.type == "cpu" else device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    eval_with_experts = resolve_eval_with_experts(model_config)
    set_experts = getattr(model, "set_inference_experts_enabled", None)
    if callable(set_experts):
        set_experts(eval_with_experts)
    prepare_for_inference = getattr(model, "prepare_for_inference", None)
    if callable(prepare_for_inference):
        prepare_for_inference()
    benchmark_reparameterize = bool(
        model_config.get("benchmark_reparameterize", False)
    )
    if benchmark_reparameterize:
        try:
            from timm.utils import reparameterize_model
        except ImportError as error:
            raise RuntimeError(
                "benchmark_reparameterize=true requires timm."
            ) from error
        model = reparameterize_model(model, inplace=True).to(device)
    model.eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    flop_input = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )
    # Profile a deepcopy: thop leaves forward hooks that can break later timing
    # (AttributeError: Conv2d has no attribute 'total_ops') on some models.
    flop_model = copy.deepcopy(model)
    with torch.inference_mode():
        macs, thop_params = profile(
            flop_model,
            inputs=(flop_input,),
            verbose=False,
        )
    del flop_model

    timing_input = torch.randn(
        args.batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )
    round_rows, peak_memory_mb = measure_fps(
        model=model,
        inputs=timing_input,
        device=device,
        precision=args.precision,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )

    fps_summary = summarize([row["fps"] for row in round_rows])
    latency_summary = summarize(
        [row["latency_ms_per_image"] for row in round_rows]
    )
    thop_macs_g = float(macs) / 1e9
    if hasattr(model, "paper_style_flops"):
        macs_g = float(
            model.paper_style_flops((3, image_size, image_size))
        ) / 1e9
        complexity_source = "model_fvcore"
    else:
        macs_g = thop_macs_g
        complexity_source = "thop"

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    results: dict[str, Any] = {
        "model": str(model_config["model_name"]),
        "dataset": str(dataset_config["dataset_name"]),
        "train_seed": args.train_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": device_name,
        "torch_version": torch.__version__,
        "input_shape_for_complexity": [1, 3, image_size, image_size],
        "batch_size_for_speed": args.batch_size,
        "precision": args.precision,
        "allow_tf32": bool(args.allow_tf32),
        "reparameterized_for_benchmark": benchmark_reparameterize,
        "eval_with_experts": bool(eval_with_experts),
        "training_auxiliary_stripped": bool(
            getattr(model, "inference_auxiliary_stripped", False)
        ),
        "warmup": args.warmup,
        "repeats_per_round": args.repeats,
        "rounds": args.rounds,
        "params": {
            "total": total_params,
            "trainable": trainable_params,
            "total_m": total_params / 1e6,
            "thop_total_m": float(thop_params) / 1e6,
        },
        "complexity": {
            "macs_g": macs_g,
            "paper_style_flops_g_1mac_equals_1flop": macs_g,
            "strict_flops_g_1mac_equals_2flops": macs_g * 2.0,
            "source": complexity_source,
            "thop_macs_g_raw": thop_macs_g,
        },
        "speed": {
            "fps": fps_summary,
            "latency_ms_per_image": latency_summary,
            "peak_memory_mb": peak_memory_mb,
            "round_details": round_rows,
        },
        "measurement_scope": (
            "model forward only; excludes image loading and preprocessing"
        ),
    }

    output_path = (
        project_path(args.output)
        if args.output is not None
        else output_dir
        / f"benchmark_{args.precision}_bs{args.batch_size}.json"
    )
    write_json(output_path, results)

    print(f"model={results['model']}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device_name}")
    print(f"params_m={total_params / 1e6:.6f}")
    print(f"macs_g={macs_g:.6f}")
    print(f"paper_style_flops_g={macs_g:.6f}")
    print(f"strict_flops_g={macs_g * 2.0:.6f}")
    print(f"complexity_source={complexity_source}")
    print(f"reparameterized_for_benchmark={benchmark_reparameterize}")
    for row in round_rows:
        print(
            f"round={int(row['round'])} "
            f"fps={row['fps']:.3f} "
            f"latency_ms={row['latency_ms_per_image']:.4f}"
        )
    print(
        f"fps_mean={fps_summary['mean']:.3f} "
        f"fps_sd={fps_summary['sd']:.3f}"
    )
    print(
        f"latency_ms_mean={latency_summary['mean']:.4f} "
        f"latency_ms_sd={latency_summary['sd']:.4f}"
    )
    print(f"peak_memory_mb={peak_memory_mb:.3f}")
    print(f"result={output_path}")


if __name__ == "__main__":
    main()
