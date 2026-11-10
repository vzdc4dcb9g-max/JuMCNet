"""Aggregate new deployment results: timing rounds first, distinct seeds second."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean_sd(values):
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "training_seed_count": len(values)}


def hardware_signature(record):
    keys = ("platform", "machine", "processor", "cpu_count", "ram_total_bytes",
            "python", "versions", "board", "governor")
    return {key: record[key] for key in keys}


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def aggregate(paths):
    groups = defaultdict(list)
    for path in paths:
        result = read_json(path)
        if result.get("schema_version") != 1 or result.get("record_type") != "deployment_result":
            raise ValueError(f"Not a new portable deployment result: {path}")
        if result.get("measurement_label") != "new_portable_measurement" or not result.get("valid"):
            raise ValueError(f"Invalid or historical result: {path}")
        rounds = result.get("rounds", [])
        if len(rounds) != result["actual_protocol"]["rounds"] or not rounds:
            raise ValueError(f"Incomplete timing rounds: {path}")
        if len({r["round"] for r in rounds}) != len(rounds):
            raise ValueError(f"Duplicate timing round identifiers: {path}")
        if any(not r.get("valid") or r["actual_protocol"] != result["actual_protocol"] for r in rounds):
            raise ValueError(f"Invalid or inconsistent timing rounds: {path}")
        if (not result.get("job_sha256") or not result.get("model_sha256") or
                any(r.get("job_sha256") != result["job_sha256"] or
                    r.get("model_sha256") != result["model_sha256"] or
                    r.get("native_libraries_sha256") != result.get("native_libraries_sha256")
                    for r in rounds)):
            raise ValueError(f"Job, ONNX, or native library hashes differ across timing rounds: {path}")
        hw = {stable(hardware_signature(r["hardware_start"])) for r in rounds}
        if len(hw) != 1:
            raise ValueError(f"Hardware/software differs across timing rounds: {path}")
        if any(len(r["samples"]) != result["test_sample_count"] for r in rounds):
            raise ValueError(f"Sample count differs across timing rounds: {path}")
        signature = {
            "protocol": result["actual_protocol"], "hardware": next(iter(hw)),
            "class_names": result["class_names"], "test_samples": result["sample_fingerprint"],
            "pure_inputs": result["pure_input_fingerprint"],
            "native_libraries": result.get("native_libraries_sha256", {}),
        }
        metrics = {key: statistics.mean(r["metrics"][key] for r in rounds)
                   for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
        latency = {key + "_ms": statistics.mean(r["latency"][key]["mean_ms"] for r in rounds)
                   for key in ("inference_only", "preprocessing", "inference_in_full_flow", "total")}
        latency["peak_process_memory_mib"] = statistics.mean(r["peak_process_memory_mib"] for r in rounds)
        latency["onnx_size_mib"] = result["model_size_bytes"] / 1024 ** 2
        key = (result["dataset_name"], result["model_name"], result["variant"])
        groups[key].append({"train_seed": result["train_seed"], "timing_rounds": len(rounds),
                            "metrics": metrics, "measurements": latency, "source": str(path),
                            "checkpoint_sha256": result["checkpoint"].get("sha256"),
                            "model_sha256": result["model_sha256"], "signature": signature})
    output = []
    for key, rows in sorted(groups.items()):
        if len({row["train_seed"] for row in rows}) != len(rows):
            raise ValueError(f"More than one result for the same training seed in {key}; specify one planned measurement per seed")
        if len({stable(row["signature"]) for row in rows}) != 1:
            raise ValueError(f"Incompatible threads, protocol, samples, preprocessing, or environment in {key}")
        row0 = rows[0]
        output.append({"dataset_name": key[0], "model_name": key[1], "variant": key[2],
                       "training_seed_count": len(rows), "timing_round_count": sum(r["timing_rounds"] for r in rows),
                       "actual_protocol": row0["signature"]["protocol"],
                       "hardware": json.loads(row0["signature"]["hardware"]),
                       "metrics": {metric: mean_sd([r["metrics"][metric] for r in rows]) for metric in row0["metrics"]},
                       "measurements": {metric: mean_sd([r["measurements"][metric] for r in rows]) for metric in row0["measurements"]},
                       "seeds": [{k: v for k, v in row.items() if k != "signature"} for row in sorted(rows, key=lambda r: r["train_seed"])]})
    return {"schema_version": 1, "record_type": "deployment_summary",
            "measurement_label": "new_portable_measurement",
            "aggregation": "Within each model/variant/seed, average all independent timing rounds; then average distinct training seeds with sample SD (ddof=1). A single seed has undefined sample SD (null). Classification metrics use fractions, not percentages.",
            "groups": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="New result.json files or directories containing them")
    parser.add_argument("--output", type=Path, required=True, help="New summary JSON path")
    args = parser.parse_args()
    paths = set()
    for entry in args.inputs:
        if entry.is_dir():
            paths.update(path.resolve() for path in entry.rglob("result.json"))
        elif entry.is_file():
            paths.add(entry.resolve())
        else:
            raise FileNotFoundError(entry)
    if not paths:
        raise ValueError("No result.json files found")
    if args.output.exists():
        raise FileExistsError(f"Summary already exists: {args.output}")
    summary = aggregate(sorted(paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "groups": len(summary["groups"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
