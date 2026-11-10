"""Export a selected checkpoint and build a self-contained FP32 deployment bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--profile", choices=("comparison", "optimization"), default="comparison")
    parser.add_argument("--sample-count", type=int, default=0,
                        help="Test images to include; 0 includes the complete fixed test split")
    parser.add_argument("--verify-count", type=int, default=0,
                        help="Test images checked against PyTorch; 0 checks every included image")
    parser.add_argument("--bank-count", type=int, default=None,
                        help="Inference-only timing bank size; default: 24 test images for comparison, 48 validation images for optimization")
    args = parser.parse_args()
    if args.sample_count < 0 or args.verify_count < 0 or (args.bank_count is not None and args.bank_count < 1):
        parser.error("sample-count and verify-count must be >= 0; bank-count must be >= 1")
    return args


def relative_reference(path: Path, root: Path) -> dict:
    try:
        return {"path": path.resolve().relative_to(root).as_posix()}
    except ValueError:
        return {"filename": path.name, "outside_project": True}


def read_split(path: Path, data_root: Path, class_names: list[str]) -> list[dict]:
    records, seen = [], set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected path, label and class name")
            name, label_text, class_name = fields
            name = name.replace("\\", "/")
            relative = PurePosixPath(name)
            if not relative.parts or relative.is_absolute() or ".." in relative.parts or ":" in relative.parts[0]:
                raise ValueError(f"Split contains a non-relative image path: {name}")
            name = relative.as_posix()
            label = int(label_text)
            if not 0 <= label < len(class_names) or class_names[label] != class_name:
                raise ValueError(f"Split class and label disagree: {name}")
            if name in seen:
                raise ValueError(f"Duplicate image in split: {name}")
            source = (data_root / Path(*relative.parts)).resolve()
            if not source.is_relative_to(data_root) or not source.is_file():
                raise FileNotFoundError(f"Split image is missing or outside data_root: {source}")
            seen.add(name)
            records.append({"source_path": name, "source": source, "label": label})
    if not records:
        raise ValueError(f"Empty split: {path}")
    return records


def select_records(records: list[dict], count: int) -> list[dict]:
    if count == 0 or count >= len(records):
        return list(records)
    groups: dict[int, list[dict]] = {}
    for record in records:
        groups.setdefault(record["label"], []).append(record)
    for group in groups.values():
        group.sort(key=lambda item: (hashlib.sha256(item["source_path"].encode("utf-8")).hexdigest(),
                                     item["source_path"]))
    selected, offset = [], 0
    while len(selected) < count:
        for label in sorted(groups):
            if offset < len(groups[label]):
                selected.append(groups[label][offset])
                if len(selected) == count:
                    return selected
        offset += 1
    return selected


def copy_records(records: list[dict], folder: str, output: Path) -> list[dict]:
    destination = output / folder
    destination.mkdir()
    copied = []
    for index, record in enumerate(records):
        source = record["source"]
        relative = f"{folder}/{index:06d}{source.suffix.lower()}"
        digest = sha256(source)
        target = output / relative
        shutil.copyfile(source, target)
        if sha256(target) != digest or sha256(source) != digest:
            raise RuntimeError(f"Image changed during preparation: {source}")
        copied.append({"path": relative, "source_path": record["source_path"],
                       "label": record["label"], "sha256": digest, "bytes": target.stat().st_size})
    return copied


def preprocess(path: Path, specification: dict):
    import numpy as np
    from PIL import Image

    size, short = specification["crop_size"], specification["resize_short_side"]
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            new_width, new_height = short, int(short * height / width)
        else:
            new_width, new_height = int(short * width / height), short
        if image.size != (new_width, new_height):
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        left = int(round((new_width - size) / 2.0))
        top = int(round((new_height - size) / 2.0))
        image = image.crop((left, top, left + size, top + size))
        array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.asarray(specification["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(specification["std"], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((array - mean) / std)[None], dtype=np.float32)


def main() -> None:
    args = parse_args()
    project = (Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]).resolve()
    if not (project / "src" / "config.py").is_file():
        raise FileNotFoundError("Cannot locate the project; provide --project-root")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(project))
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        import torchvision
        import timm
        import PIL
        from PIL import Image
        from src.config import load_runtime_configs, project_path, resolve_checkpoint
        from src.datasets import build_transforms
        from src.models import create_model
        from src.utils import experiment_directory, resolve_eval_with_experts
    except ImportError as error:
        raise RuntimeError("Install the project requirements and deployment/raspberry_pi5/requirements-export.txt") from error

    source_files = {path.relative_to(project).as_posix(): sha256(path)
                    for path in sorted((project / "src").rglob("*.py"))}
    script_sha256 = sha256(Path(__file__).resolve())
    loader_args = argparse.Namespace(resolved_config=args.resolved_config, dataset_config=None,
                                     experiment_config=None, model_config=None, train_seed=None)
    dataset, experiment, model_config, seed = load_runtime_configs(loader_args)
    resolved_path = project_path(args.resolved_config).resolve()
    run_dir = experiment_directory(dataset, experiment, model_config, seed)
    checkpoint_path = resolve_checkpoint(args.checkpoint, run_dir).resolve()
    output = project_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Bundle directory already exists; choose a new --output-dir: {output}")
    if model_config.get("pretrained", False):
        raise ValueError("A selected trained checkpoint must use pretrained=false; no weights are downloaded")
    if dataset.get("preprocessing", {}).get("watermark_mask", {}).get("enabled", False):
        raise ValueError("This deployment entry does not implement watermark_mask preprocessing")
    classes = list(dataset["class_names"])
    if int(dataset["num_classes"]) != len(classes) or int(dataset["image"]["channels"]) != 3:
        raise ValueError("Invalid class count or non-RGB dataset")
    size = int(dataset["image"]["input_size"])
    preprocessing = {"resize_short_side": int(round(size / 0.875)), "crop_size": size,
                     "crop": "center", "interpolation": "bicubic", "rgb": True,
                     "scale": 255.0, "mean": list(dataset["normalization"]["mean"]),
                     "std": list(dataset["normalization"]["std"])}
    data_root = project_path(dataset["data_root"]).resolve()
    split_root = project_path(dataset["split"]["output_dir"]).resolve()
    test_split, val_split = split_root / "test.txt", split_root / "val.txt"
    source_hashes = {path: sha256(path) for path in (resolved_path, checkpoint_path, test_split, val_split)}
    test_records = read_split(test_split, data_root, classes)
    val_records = read_split(val_split, data_root, classes)
    if {item["source_path"] for item in test_records} & {item["source_path"] for item in val_records}:
        raise ValueError("Test and validation splits overlap")
    selected_test = select_records(test_records, args.sample_count)
    comparison = args.profile == "comparison"
    bank_count = args.bank_count if args.bank_count is not None else (24 if comparison else 48)
    bank_records = test_records if comparison else val_records
    selected_bank = select_records(bank_records, bank_count)
    selected_checks = select_records(selected_test, args.verify_count)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Expected a project checkpoint containing model_state")
    saved = checkpoint.get("configs", {})
    if saved.get("model") != model_config or saved.get("train_seed") != seed:
        raise ValueError("Checkpoint model configuration or training seed differs from --resolved-config")
    saved_dataset = saved.get("dataset", {})
    if saved_dataset.get("num_classes") != len(classes) or saved_dataset.get("class_names") != classes:
        raise ValueError("Checkpoint classes differ from --resolved-config")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = create_model(model_config, num_classes=len(classes), class_names=classes).cpu().float().eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    set_experts = getattr(model, "set_inference_experts_enabled", None)
    if callable(set_experts):
        set_experts(resolve_eval_with_experts(model_config))
    original_logits = []
    _, evaluation_transform = build_transforms(dataset, experiment, model_config=model_config)
    for record in selected_checks:
        x = preprocess(record["source"], preprocessing)
        with Image.open(record["source"]) as image:
            expected = evaluation_transform(image.convert("RGB")).numpy()[None]
        if not np.array_equal(x, expected):
            raise RuntimeError(f"Portable preprocessing differs from the project: {record['source_path']}")
        with torch.inference_mode():
            original_logits.append(model(torch.from_numpy(x)).numpy()[0])
    original_logits = np.asarray(original_logits, dtype=np.float32)
    prepare_for_inference = getattr(model, "prepare_for_inference", None)
    if callable(prepare_for_inference):
        prepare_for_inference()
    reparameterized = bool(model_config.get("benchmark_reparameterize", False))
    if reparameterized:
        from timm.utils import reparameterize_model
        model = reparameterize_model(model, inplace=True).cpu().float().eval()
    auxiliary_stripped = bool(getattr(model, "inference_auxiliary_stripped", False))
    parameter_count_deployed = sum(parameter.numel() for parameter in model.parameters())
    output.mkdir(parents=True, exist_ok=False)
    samples = copy_records(selected_test, "images", output)
    validation_samples = copy_records(selected_bank, "validation_images", output)
    by_source = {item["source_path"]: item for item in samples}
    onnx_path = output / "model.onnx"
    dummy = torch.from_numpy(preprocess(output / samples[0]["path"], preprocessing))
    with torch.inference_mode():
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "image" or session.get_inputs()[0].shape != [1, 3, size, size]:
        raise RuntimeError("Unexpected ONNX input signature")
    if session.get_inputs()[0].type != "tensor(float)":
        raise RuntimeError("ONNX input is not FP32")
    export_checks, max_abs, deployment_max_abs = [], 0.0, 0.0
    for index, record in enumerate(selected_checks):
        sample = by_source[record["source_path"]]
        x = preprocess(output / sample["path"], preprocessing)
        with torch.inference_mode():
            deployed = model(torch.from_numpy(x)).numpy()[0]
        logits = session.run(["logits"], {"image": x})[0][0]
        original = original_logits[index]
        if not np.isfinite(logits).all() or logits.shape != (len(classes),):
            raise RuntimeError(f"Invalid ONNX output: {sample['path']}")
        if not np.allclose(original, deployed, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"Inference preparation changed logits: {sample['path']}")
        if not np.allclose(original, logits, atol=1e-4, rtol=1e-4) or int(original.argmax()) != int(logits.argmax()):
            raise RuntimeError(f"ONNX/PyTorch verification failed: {sample['path']}")
        max_abs = max(max_abs, float(np.abs(original - logits).max()))
        deployment_max_abs = max(deployment_max_abs, float(np.abs(original - deployed).max()))
        export_checks.append({"path": sample["path"], "label": sample["label"],
                              "torch_logits": original.tolist(), "onnx_logits": logits.tolist()})
        if (index + 1) % 250 == 0:
            print(f"Verified {index + 1}/{len(selected_checks)} images", flush=True)
    for path, digest in source_hashes.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Input changed during export: {path}")
    for name, digest in source_files.items():
        if sha256(project / name) != digest:
            raise RuntimeError(f"Project source changed during export: {name}")
    if sha256(Path(__file__).resolve()) != script_sha256:
        raise RuntimeError("Preparation script changed during export")
    full_test_count, sample_count = len(test_records), len(samples)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_config["model_name"], "dataset_name": dataset["dataset_name"],
        "train_seed": seed, "variant": "plain", "precision": "FP32",
        "checkpoint": {**relative_reference(checkpoint_path, project),
                       "sha256": source_hashes[checkpoint_path], "epoch": checkpoint.get("epoch"),
                       "weight_source": checkpoint.get("weight_source")},
        "resolved_config": {**relative_reference(resolved_path, project), "sha256": source_hashes[resolved_path]},
        "model_file": "model.onnx", "model_sha256": sha256(onnx_path),
        "class_names": classes, "input_size": size, "input_shape": [1, 3, size, size],
        "preprocessing": preprocessing, "samples": samples,
        "full_test_count": full_test_count, "test_sample_count": sample_count,
        "measurement_scope": "full_test" if full_test_count == sample_count else "test_subset",
        "selection": {"full_test_order": "original fixed test manifest order",
                      "subset_and_validation_order": "round-robin class labels; SHA256 of UTF-8 relative path within each class",
                      "requested_test_count": args.sample_count, "requested_verify_count": args.verify_count,
                      "split_seed": dataset["split"]["seed"],
                      "test_split": {**relative_reference(test_split, project), "sha256": source_hashes[test_split]},
                      "validation_split": {**relative_reference(val_split, project), "sha256": source_hashes[val_split]}},
        "protocol": {"profile": args.profile, "threads": 4, "inter_threads": 1, "warmup": 100,
                     "repeats": 500, "rounds": 1 if comparison else 2,
                     "benchmark_rounds": 3 if comparison else 1,
                     "execution_mode": "sequential", "graph_optimization": "all", "thread_spinning": False},
        "validation": {"samples": validation_samples,
                       "sample_source": "fixed test split" if comparison else "fixed validation split",
                       "full_validation_count": len(val_records), "export_checks": export_checks,
                       "export_check_count": len(export_checks), "all_selected_test_images_verified": len(export_checks) == sample_count,
                       "full_test_equivalence_verified": len(export_checks) == full_test_count,
                       "torch_onnx_max_abs": max_abs, "inference_preparation_max_abs": deployment_max_abs,
                       "prediction_mismatches": 0, "atol": 1e-4, "rtol": 1e-4,
                       "preprocessing_matches_project": True, "passed": True},
        "deployment": {"auxiliary_stripped": auxiliary_stripped, "reparameterized": reparameterized,
                       "parameters_before": parameter_count_before, "parameters_deployed": parameter_count_deployed,
                       "model_config": model_config},
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": np.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__, "pillow": PIL.__version__},
        "source_files_sha256": source_files, "prepare_script_sha256": script_sha256,
    }
    temporary = output / "job.json.partial"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "job.json")
    print(f"Prepared {output}: {sample_count}/{full_test_count} test images; "
          f"{len(export_checks)} checked; {len(validation_samples)} {args.profile} timing images", flush=True)


if __name__ == "__main__":
    main()
