"""Evaluate a prepared FP32 ONNX bundle without PyTorch or SSH."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image
import psutil


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Result already exists: {path}")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def bundle_file(bundle, relative):
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError(f"Bundle file is missing or outside the bundle: {relative}")
    return path


def text_file(path):
    try:
        return Path(path).read_text().strip().strip("\0")
    except OSError:
        return None


def temperature():
    value = text_file("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(value) / 1000 if value is not None else None
    except ValueError:
        return None


def throttled():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"],
                                       text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware():
    versions = {}
    for name in ("onnxruntime", "numpy", "Pillow", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    board = text_file("/proc/device-tree/model")
    return {
        "hostname": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(), "versions": versions, "board": board,
        "is_raspberry_pi5": bool(board and "Raspberry Pi 5" in board),
        "governor": text_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "frequency_khz": text_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        "temperature_c": temperature(), "throttled": throttled(),
    }


def hardware_signature(record):
    keys = ("platform", "machine", "processor", "cpu_count", "ram_total_bytes",
            "python", "versions", "board", "governor")
    return {key: record[key] for key in keys}


def cropped_image(path, size, config):
    if config.get("interpolation", "bicubic").lower() != "bicubic":
        raise ValueError("This deployment entry supports bicubic interpolation only")
    if config.get("crop", "center").lower() != "center":
        raise ValueError("This deployment entry supports center cropping only")
    short = int(config.get("resize_short_side", 256))
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        nw, nh = ((short, int(short * height / width)) if width <= height
                  else (int(short * width / height), short))
        if image.size != (nw, nh):
            image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        left, top = int(round((nw - size) / 2.)), int(round((nh - size) / 2.))
        image = image.crop((left, top, left + size, top + size))
        return image


def preprocess_pixels(path, size, config):
    return np.array(cropped_image(path, size, config), dtype=np.uint8)


def preprocess(path, size, config):
    mean = np.asarray(config["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(config["std"], dtype=np.float32)[:, None, None]
    if mean.shape != (3, 1, 1) or std.shape != (3, 1, 1) or np.any(std <= 0):
        raise ValueError("Invalid RGB normalization")
    pixels = np.array(cropped_image(path, size, config), dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    return np.ascontiguousarray(((pixels - mean) / std)[None], dtype=np.float32)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty or non-finite measurements")
    return {"count": int(len(values)), "mean_ms": float(values.mean()),
            "sample_sd_ms": float(values.std(ddof=1)) if len(values) > 1 else None,
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def classification_metrics(targets, predictions, classes):
    targets, predictions = np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)
    cm = np.bincount(classes * targets + predictions, minlength=classes ** 2).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(), "support": cm.sum(1).tolist()}


class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop.wait(0.01)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--native-dir", type=Path, help="Compiled ARM64 custom libraries; required for optimized bundles")
    for name in ("threads", "inter-threads", "warmup", "repeats", "rounds", "benchmark-rounds"):
        parser.add_argument("--" + name, type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses every prepared test sample")
    parser.add_argument("--cooldown-seconds", type=float, default=None)
    parser.add_argument("--max-cooldown-seconds", type=float, default=None)
    parser.add_argument("--temperature-limit", type=float, default=50)
    parser.add_argument("--skip-cooldown", action="store_true", help="Smoke checks only; recorded in the result")
    parser.add_argument("--_worker-round", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def configuration(args):
    bundle = args.bundle.resolve()
    job = read_json(bundle_file(bundle, "job.json"))
    if job.get("schema_version") != 1:
        raise ValueError("Unsupported job schema_version")
    profile = job.get("protocol", {}).get("profile", "optimization")
    if profile not in ("comparison", "optimization"):
        raise ValueError("Unknown deployment profile")
    defaults = {"threads": 4, "inter_threads": 1, "warmup": 100, "repeats": 500,
                "rounds": 1 if profile == "comparison" else 2,
                "benchmark_rounds": 3 if profile == "comparison" else 1}
    protocol = {key: (getattr(args, key) if getattr(args, key) is not None
                      else job.get("protocol", {}).get(key, value)) for key, value in defaults.items()}
    for key, value in protocol.items():
        if not isinstance(value, int) or value < (0 if key == "warmup" else 1):
            raise ValueError(f"Invalid protocol value: {key}={value}")
    if args.cooldown_seconds is None:
        args.cooldown_seconds = 15 if profile == "comparison" else 12
    if args.max_cooldown_seconds is None:
        args.max_cooldown_seconds = 300 if profile == "comparison" else 60
    if args.skip_cooldown:
        args.cooldown_seconds = args.max_cooldown_seconds = 0
    if args.max_samples < 0 or args.cooldown_seconds < 0 or args.max_cooldown_seconds < args.cooldown_seconds:
        raise ValueError("Invalid sample limit or cooldown interval")
    size = job["input_size"]
    if isinstance(size, list):
        if len(size) != 2 or size[0] != size[1]:
            raise ValueError("Only square inputs are supported")
        size = size[0]
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Invalid input_size")
    samples = job["samples"][:args.max_samples or None]
    if not samples or not job["class_names"]:
        raise ValueError("Empty test samples or class_names")
    if len({row["path"] for row in samples}) != len(samples):
        raise ValueError("Duplicate test sample paths")
    for row in samples:
        if not isinstance(row["label"], int) or not 0 <= row["label"] < len(job["class_names"]):
            raise ValueError("Invalid class label")
    full_count = int(job.get("full_test_count", len(job["samples"])))
    if full_count < len(job["samples"]):
        raise ValueError("full_test_count is smaller than the prepared sample count")
    validation = job.get("validation", {})
    bank_rows = validation.get("samples") or samples[:48]
    optimized = job.get("variant", "plain") == "optimized"
    if optimized and args.native_dir is None:
        raise ValueError("An optimized bundle requires --native-dir")
    if optimized and (size != 224 or int(job["preprocessing"].get("resize_short_side", 256)) != 256):
        raise ValueError("The native preprocessing implementation supports Resize256/CenterCrop224 only")
    protocol.update({"profile": profile, "precision": "FP32", "batch_size": 1,
                     "provider": "CPUExecutionProvider", "execution_mode": "ORT_SEQUENTIAL",
                     "graph_optimization": "ORT_ENABLE_ALL", "thread_spinning": False,
                     "pure_input_source": validation.get("sample_source", "validation_subset") if validation.get("samples") else "test_subset",
                     "pure_input_count": len(bank_rows), "input_size": size,
                     "preprocessing_implementation": "native_cropresize_lut" if optimized else "Pillow_numpy",
                     "preprocessing": job["preprocessing"], "test_sample_count": len(samples),
                     "full_test_count": full_count,
                     "measurement_scope": "full_test" if len(samples) == full_count else "test_subset",
                     "cooldown_min_seconds": args.cooldown_seconds,
                     "cooldown_max_seconds": args.max_cooldown_seconds,
                     "cooldown_skipped_for_smoke": args.skip_cooldown,
                     "timing_order": "full_test_then_pure" if profile == "comparison" else "pure_then_full_test",
                     "temperature_limit_c": args.temperature_limit,
                     "timing": "file read/decode/preprocess + inference + softmax/argmax; excludes model loading, startup, camera, network, and result writing"})
    return bundle, job, protocol, size, samples, bank_rows


def evaluate_round(args):
    import onnxruntime as ort
    job_hash = sha256(args.bundle.resolve() / "job.json")
    bundle, job, protocol, size, samples, bank_rows = configuration(args)
    if sha256(bundle / "job.json") != job_hash:
        raise ValueError("job.json changed while loading the round configuration")
    model = bundle_file(bundle, job["model_file"])
    model_hash = sha256(model)
    if model_hash != job["model_sha256"]:
        raise ValueError("ONNX SHA256 does not match job.json")
    rows_to_check = {row["path"]: row for row in samples + bank_rows}
    for row in rows_to_check.values():
        path = bundle_file(bundle, row["path"])
        if row.get("sha256") and sha256(path) != row["sha256"]:
            raise ValueError(f"Image SHA256 mismatch: {row['path']}")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    native = None
    native_hashes = {}
    optimized = job.get("variant", "plain") == "optimized"
    if optimized:
        from portable_native import NativePreprocess, register_custom_ops
        native_dir = args.native_dir.resolve()
        names = ("dw_silu", "pw_silu", "pw24_silu", "stem_silu", "more_ops", "direct", "dw8", "fc", "lut", "cropresize")
        native_hashes = {name: sha256(native_dir / (name + ".so")) for name in names}
        native = NativePreprocess(native_dir, job["preprocessing"]["mean"], job["preprocessing"]["std"])
    with MemoryMonitor() as memory:
        options = ort.SessionOptions()
        options.intra_op_num_threads = protocol["threads"]
        options.inter_op_num_threads = protocol["inter_threads"]
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            options.add_session_config_entry(key, "0")
        if optimized:
            register_custom_ops(options, native_dir)
        session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Expected a single-input, single-output FP32 classification ONNX")
        if inputs[0].shape != [1, 3, size, size] or session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("Unexpected input shape or execution provider")
        name = inputs[0].name
        actual_options = session.get_session_options()
        if (actual_options.intra_op_num_threads != protocol["threads"] or
                actual_options.inter_op_num_threads != protocol["inter_threads"] or
                actual_options.execution_mode != ort.ExecutionMode.ORT_SEQUENTIAL or
                actual_options.graph_optimization_level != ort.GraphOptimizationLevel.ORT_ENABLE_ALL or
                any(actual_options.get_session_config_entry(key) != "0" for key in
                    ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"))):
            raise ValueError("Runtime thread/optimization options differ from the recorded protocol")
        def infer(tensor):
            logits = session.run(None, {name: tensor})[0]
            if logits.shape != (1, len(job["class_names"])) or not np.isfinite(logits).all():
                raise ValueError("Invalid or non-finite classifier output")
            return logits[0]
        def tensor(row):
            path = bundle_file(bundle, row["path"])
            return native(path) if optimized else preprocess(path, size, job["preprocessing"])
        equivalence = {"applicable": optimized, "sample_count": 0,
                       "rgb_pixels_bitwise_equal": None, "fp32_inputs_bitwise_equal": None,
                       "full_test_predictions_equal": None, "max_abs_logit_difference": None}
        if optimized:
            plain_path = bundle_file(bundle, job["reference_plain_model"])
            if sha256(plain_path) != job["reference_plain_sha256"]:
                raise ValueError("Reference plain ONNX SHA256 does not match job.json")
            plain_options = ort.SessionOptions()
            plain_options.intra_op_num_threads = protocol["threads"]
            plain_options.inter_op_num_threads = protocol["inter_threads"]
            plain_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            plain_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
                plain_options.add_session_config_entry(key, "0")
            plain_session = ort.InferenceSession(str(plain_path), sess_options=plain_options, providers=["CPUExecutionProvider"])
            plain_name = plain_session.get_inputs()[0].name
            max_plain_diff = 0.0
            for row in rows_to_check.values():
                path = bundle_file(bundle, row["path"])
                standard = preprocess(path, size, job["preprocessing"])
                candidate = native(path)
                if (not np.array_equal(preprocess_pixels(path, size, job["preprocessing"]), native.pixels(path)) or
                        not np.array_equal(standard, candidate)):
                    raise ValueError(f"Native preprocessing is not bitwise equivalent: {row['path']}")
                expected = plain_session.run(None, {plain_name: standard})[0][0]
                actual = infer(candidate)
                max_plain_diff = max(max_plain_diff, float(np.max(np.abs(expected - actual))))
                if not np.allclose(expected, actual, atol=2e-4, rtol=2e-4) or expected.argmax() != actual.argmax():
                    raise ValueError(f"Optimized/plain output mismatch: {row['path']}")
            equivalence.update({"sample_count": len(rows_to_check), "rgb_pixels_bitwise_equal": True,
                                "fp32_inputs_bitwise_equal": True, "full_test_predictions_equal": len(samples) == protocol["full_test_count"],
                                "max_abs_logit_difference": max_plain_diff})
            del plain_session
        checks = job.get("validation", {}).get("export_checks", [])
        max_diff = 0.0
        for row in checks:
            expected = np.asarray(row["onnx_logits"], dtype=np.float32).reshape(-1)
            actual = infer(tensor(row))
            if expected.shape != actual.shape or not np.isfinite(expected).all():
                raise ValueError("Invalid export reference logits")
            max_diff = max(max_diff, float(np.max(np.abs(expected - actual))))
            if not np.allclose(expected, actual, atol=2e-4, rtol=2e-4) or expected.argmax() != actual.argmax():
                raise ValueError(f"Device/export prediction mismatch: {row['path']}")
        bank = [tensor(row) for row in bank_rows]
        cooldown_start = time.monotonic()
        while True:
            elapsed = time.monotonic() - cooldown_start
            current_temp = temperature()
            if elapsed >= args.cooldown_seconds and (current_temp is None or current_temp <= args.temperature_limit):
                break
            if elapsed >= args.max_cooldown_seconds:
                break
            time.sleep(min(0.5, max(0.01, args.max_cooldown_seconds - elapsed)))
        cooldown = time.monotonic() - cooldown_start
        before = hardware()
        if (protocol["profile"] == "comparison" and not args.skip_cooldown and
                before["temperature_c"] is not None and before["temperature_c"] > args.temperature_limit):
            raise RuntimeError("Comparison profile did not reach the required cooldown temperature")
        if before["throttled"] not in (None, "throttled=0x0"):
            raise RuntimeError("Device reports throttling/undervoltage before timing")
        for index in range(protocol["warmup"]):
            infer(bank[index % len(bank)])
        def benchmark():
            flat, groups = [], []
            for number in range(protocol["benchmark_rounds"]):
                one = []
                for index in range(protocol["repeats"]):
                    value = bank[index % len(bank)]
                    start = time.perf_counter_ns()
                    session.run(None, {name: value})
                    one.append((time.perf_counter_ns() - start) / 1e6)
                flat.extend(one)
                groups.append({"benchmark_round": number + 1, "latencies_ms": one, "statistics": stats(one)})
            return flat, groups
        if protocol["profile"] == "optimization":
            pure, pure_rounds = benchmark()
        predictions, records = [], []
        for row in samples:
            path = bundle_file(bundle, row["path"])
            start = time.perf_counter_ns()
            value = native(path) if optimized else preprocess(path, size, job["preprocessing"])
            after_preprocess = time.perf_counter_ns()
            logits = session.run(None, {name: value})[0][0]
            after_infer = time.perf_counter_ns()
            prediction = int(logits.argmax())
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            confidence = float(probabilities[prediction])
            end = time.perf_counter_ns()
            if not np.isfinite(logits).all():
                raise ValueError("Non-finite test logits")
            predictions.append(prediction)
            records.append({"path": row["path"], "label": row["label"], "prediction": prediction,
                            "confidence": confidence, "preprocessing_ms": (after_preprocess - start) / 1e6,
                            "inference_ms": (after_infer - after_preprocess) / 1e6,
                            "total_ms": (end - start) / 1e6})
        if protocol["profile"] == "comparison":
            pure, pure_rounds = benchmark()
        after = hardware()
    verify_artifacts(bundle, job_hash, model, model_hash,
                     args.native_dir.resolve() if optimized else None, native_hashes)
    warnings = []
    if args.skip_cooldown:
        warnings.append("Cooldown was explicitly skipped for a smoke check.")
    if not before["is_raspberry_pi5"]:
        warnings.append("This run was not identified as Raspberry Pi 5; it is not a Raspberry Pi 5 timing result.")
    if before["throttled"] is None or after["throttled"] is None:
        warnings.append("Throttling status was unavailable.")
    if before["temperature_c"] is not None and before["temperature_c"] > args.temperature_limit:
        warnings.append("Cooldown reached its time limit above the requested temperature.")
    valid = after["throttled"] in (None, "throttled=0x0")
    if not valid:
        warnings.append("Throttling/undervoltage was reported after timing; this result is invalid for aggregation.")
    if hardware_signature(before) != hardware_signature(after):
        valid = False
        warnings.append("Hardware/software configuration changed during timing.")
    result = {"schema_version": 1, "record_type": "deployment_round", "round": args._worker_round,
              "pid": os.getpid(), "started_utc": started,
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "valid": valid, "job_sha256": job_hash,
              "model_sha256": model_hash, "model_size_bytes": model.stat().st_size,
              "actual_protocol": protocol, "native_libraries_sha256": native_hashes,
              "native_equivalence": equivalence,
              "hardware_start": before, "hardware_end": after, "cooldown_seconds": cooldown,
              "peak_process_memory_mib": memory.peak / 1024 ** 2,
              "peak_process_memory_scope": "Sampled RSS of this fresh process includes imports, loaded sessions, correctness verification, input/reference data, and evaluation; optimized verification also loads a plain reference session before timing.",
              "export_consistency": {"sample_count": len(checks), "max_abs_logit_difference": max_diff,
                                     "atol": 2e-4, "rtol": 2e-4, "passed": bool(checks)},
              "metrics": classification_metrics([r["label"] for r in samples], predictions, len(job["class_names"])),
              "latency": {"inference_only": stats(pure),
                          "preprocessing": stats([r["preprocessing_ms"] for r in records]),
                          "inference_in_full_flow": stats([r["inference_ms"] for r in records]),
                          "total": stats([r["total_ms"] for r in records])},
              "pure_raw_ms": pure, "pure_rounds": pure_rounds, "samples": records, "warnings": warnings}
    write_json(args.output.resolve() / f"round_{args._worker_round:02d}.json", result)
    if not valid:
        raise RuntimeError("Throttling/undervoltage occurred; invalid round retained without aggregation")


def verify_artifacts(bundle, job_hash, model, model_hash, native_dir, native_hashes):
    if sha256(bundle / "job.json") != job_hash:
        raise ValueError("job.json changed during measurement")
    if sha256(model) != model_hash:
        raise ValueError("ONNX changed during measurement")
    for name, expected in native_hashes.items():
        if sha256(native_dir / (name + ".so")) != expected:
            raise ValueError(f"Native library changed during measurement: {name}")


def finalize(args):
    job_hash = sha256(args.bundle.resolve() / "job.json")
    bundle, job, protocol, _, samples, bank_rows = configuration(args)
    output = args.output.resolve()
    rounds = [read_json(output / f"round_{number:02d}.json") for number in range(1, protocol["rounds"] + 1)]
    if any(row["actual_protocol"] != protocol for row in rounds):
        raise ValueError("Protocol differed across independent processes")
    native_hashes = rounds[0]["native_libraries_sha256"]
    if any(row.get("job_sha256") != job_hash or
           row["model_sha256"] != job["model_sha256"] or
           row["native_libraries_sha256"] != native_hashes for row in rounds):
        raise ValueError("Job, ONNX, or native library hashes differed across timing rounds")
    verify_artifacts(bundle, job_hash, bundle_file(bundle, job["model_file"]), job["model_sha256"],
                     args.native_dir.resolve() if native_hashes else None, native_hashes)
    if len({fingerprint(hardware_signature(row["hardware_start"])) for row in rounds}) != 1:
        raise ValueError("Hardware/software settings differed across rounds")
    if len({fingerprint([r["prediction"] for r in row["samples"]]) for row in rounds}) != 1:
        raise ValueError("Predictions differed across timing rounds")
    result = {"schema_version": 1, "record_type": "deployment_result",
              "measurement_label": "new_portable_measurement", "model_name": job["model_name"],
              "dataset_name": job["dataset_name"], "train_seed": job["train_seed"],
              "variant": job.get("variant", "plain"), "class_names": job["class_names"],
              "checkpoint": job["checkpoint"], "model_sha256": rounds[0]["model_sha256"],
              "model_size_bytes": rounds[0]["model_size_bytes"],
              "native_libraries_sha256": native_hashes,
              "job_sha256": job_hash, "bundle": str(bundle),
              "test_sample_count": len(samples), "full_test_count": protocol["full_test_count"],
              "sample_fingerprint": fingerprint(samples), "pure_input_fingerprint": fingerprint(bank_rows),
              "actual_protocol": protocol, "valid": all(row["valid"] for row in rounds), "rounds": rounds,
              "aggregation": "Rounds are independent timing processes for one training seed, not additional training seeds."}
    if hasattr(args, "pair_metadata"):
        result["pairing"] = args.pair_metadata
    write_json(output / "result.json", result)
    print(json.dumps({"result": str(output / "result.json"), "seed": job["train_seed"],
                      "rounds": len(rounds), "test_sample_count": len(samples),
                      "full_test_count": protocol["full_test_count"]}, ensure_ascii=False))
    return result


def main():
    args = parse_args()
    if args._worker_round is not None:
        evaluate_round(args)
        return
    bundle, _, protocol, _, _, _ = configuration(args)
    output = args.output.resolve()
    if output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Use a separate output directory outside the immutable input bundle")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(protocol["threads"]),
                "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"})
    for number in range(1, protocol["rounds"] + 1):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--_worker-round", str(number)]
        subprocess.run(command, check=True, env=env)
    finalize(args)


if __name__ == "__main__":
    main()
