from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import device


ORDER = (("plain", 1), ("optimized", 1), ("optimized", 2), ("plain", 2))


def command_options(args, variant: str) -> list[str]:
    bundle = args.plain_bundle if variant == "plain" else args.optimized_bundle
    options = ["--bundle", str(bundle.resolve()), "--output", str(args.output_dir.resolve() / variant), "--rounds", "2"]
    if variant == "optimized":
        options.extend(["--native-dir", str(args.native_dir.resolve())])
    for name in ("warmup", "repeats", "max_samples", "cooldown_seconds", "max_cooldown_seconds", "temperature_limit"):
        value = getattr(args, name)
        if value is not None:
            options.extend(["--" + name.replace("_", "-"), str(value)])
    if args.skip_cooldown:
        options.append("--skip-cooldown")
    return options


def comparable_protocol(protocol):
    result = copy.deepcopy(protocol)
    result.pop("preprocessing_implementation", None)
    return result


def verify_pair(configurations):
    plain_bundle, plain, plain_protocol, plain_size, plain_samples, plain_bank = configurations["plain"]
    optimized_bundle, optimized, opt_protocol, opt_size, opt_samples, opt_bank = configurations["optimized"]
    if plain.get("variant") != "plain" or optimized.get("variant") != "optimized":
        raise ValueError("Pass a plain bundle and an optimized bundle, respectively.")
    for name in ("model_name", "dataset_name", "train_seed", "class_names", "preprocessing", "input_shape"):
        if plain[name] != optimized[name]:
            raise ValueError("Paired bundles disagree on " + name)
    if not plain["checkpoint"].get("sha256") or plain["checkpoint"]["sha256"] != optimized["checkpoint"].get("sha256"):
        raise ValueError("Paired bundles must use the same checkpoint SHA256.")
    if plain_size != opt_size or plain_samples != opt_samples or plain_bank != opt_bank:
        raise ValueError("Paired bundles must contain identical ordered test samples and pure-inference input banks.")
    if plain["samples"] != optimized["samples"] or plain.get("validation", {}).get("export_checks") != optimized.get("validation", {}).get("export_checks"):
        raise ValueError("Paired bundles have different full sample lists or export references.")
    if plain_protocol["profile"] != "optimization" or opt_protocol["profile"] != "optimization":
        raise ValueError("ABBA pairing is for the optimization profile; prepare both bundles with that profile.")
    if plain_protocol["rounds"] != 2 or plain_protocol["benchmark_rounds"] != 1:
        raise ValueError("ABBA uses two independent processes per variant and one pure-inference loop per process.")
    if comparable_protocol(plain_protocol) != comparable_protocol(opt_protocol):
        raise ValueError("Paired timing protocols differ beyond the preprocessing implementation.")
    plain_model = device.bundle_file(plain_bundle, plain["model_file"])
    expected_plain_hash = device.sha256(plain_model)
    if expected_plain_hash != plain["model_sha256"] or expected_plain_hash != optimized.get("reference_plain_sha256"):
        raise ValueError("The optimized bundle does not reference the supplied plain model.")
    reference = device.bundle_file(optimized_bundle, optimized["reference_plain_model"])
    if device.sha256(reference) != expected_plain_hash:
        raise ValueError("The optimized bundle's embedded plain reference SHA256 is wrong.")


def main():
    parser = argparse.ArgumentParser(description="Measure one checkpoint's plain/optimized pair in four fresh processes: plain1, optimized1, optimized2, plain2.")
    parser.add_argument("--plain-bundle", type=Path, required=True)
    parser.add_argument("--optimized-bundle", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-samples", type=int, help="Smoke checks only; omit to use every prepared image.")
    parser.add_argument("--cooldown-seconds", type=float)
    parser.add_argument("--max-cooldown-seconds", type=float)
    parser.add_argument("--temperature-limit", type=float)
    parser.add_argument("--skip-cooldown", action="store_true", help="Smoke checks only; recorded in each result.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    options = {variant: command_options(args, variant) for variant in ("plain", "optimized")}
    parsed = {variant: device.parse_args(value) for variant, value in options.items()}
    configurations = {variant: device.configuration(value) for variant, value in parsed.items()}
    verify_pair(configurations)
    for bundle, *_ in configurations.values():
        if output.is_relative_to(bundle) or bundle.is_relative_to(output):
            raise ValueError("Use a new output directory outside both input bundles.")
    if not args.native_dir.is_dir():
        raise FileNotFoundError("Missing compiled native library directory: " + str(args.native_dir))
    from portable_native import CUSTOM_OP_LIBRARIES
    for name in (*CUSTOM_OP_LIBRARIES, "lut", "cropresize"):
        if not (args.native_dir / (name + ".so")).is_file():
            raise FileNotFoundError("Build the native libraries before starting the paired session: " + name + ".so")
    output.mkdir(parents=True, exist_ok=False)
    for variant in options:
        (output / variant).mkdir()
    pair_id = str(uuid.uuid4())
    plan = {
        "schema_version": 1, "record_type": "deployment_pair_plan", "pair_id": pair_id,
        "order": [{"variant": variant, "round": number} for variant, number in ORDER],
        "checkpoint_sha256": configurations["plain"][1]["checkpoint"]["sha256"],
        "train_seed": configurations["plain"][1]["train_seed"],
        "scope": "One training seed and one checkpoint; each variant is measured in two independent processes in this paired session.",
    }
    device.write_json(output / "pair_plan.json", plan)
    protocol = configurations["plain"][2]
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(protocol["threads"]),
        "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0",
    })
    sequence = []
    for variant, number in ORDER:
        print(f"Paired measurement {len(sequence) + 1}/4: {variant}, round {number}", flush=True)
        command = [sys.executable, "-B", str(Path(device.__file__).resolve()), *options[variant], "--_worker-round", str(number)]
        subprocess.run(command, check=True, env=environment)
        record = device.read_json(output / variant / f"round_{number:02d}.json")
        if not record["valid"] or record["round"] != number or record["actual_protocol"] != configurations[variant][2]:
            raise RuntimeError("Invalid paired round or unexpected effective protocol.")
        sequence.append({
            "variant": variant, "round": number, "pid": record["pid"],
            "started_utc": record["started_utc"], "finished_utc": record["finished_utc"],
            "record": str(Path(variant) / f"round_{number:02d}.json"),
        })
    if len({item["pid"] for item in sequence}) != 4:
        raise RuntimeError("The four rounds were not recorded as four distinct processes.")
    rounds = [device.read_json(output / item["record"]) for item in sequence]
    if len({device.fingerprint(device.hardware_signature(item["hardware_start"])) for item in rounds}) != 1:
        raise RuntimeError("Hardware or software settings changed across the paired session.")
    if len({device.fingerprint([sample["prediction"] for sample in item["samples"]]) for item in rounds}) != 1:
        raise RuntimeError("The plain and optimized paired measurements produced different predictions.")
    optimized_rounds = [record for record, item in zip(rounds, sequence) if item["variant"] == "optimized"]
    if optimized_rounds[0]["native_libraries_sha256"] != optimized_rounds[1]["native_libraries_sha256"]:
        raise RuntimeError("Compiled native libraries changed between the optimized measurements.")
    metadata = {"pair_id": pair_id, "order": plan["order"], "sequence": sequence, "scope": plan["scope"]}
    results = {}
    for variant in ("plain", "optimized"):
        parsed[variant].pair_metadata = metadata
        results[variant] = device.finalize(parsed[variant])
    device.write_json(output / "pair.json", {
        **plan, "record_type": "deployment_pair", "sequence": sequence, "completed": True,
        "results": {variant: {"path": variant + "/result.json", "sha256": device.sha256(output / variant / "result.json")} for variant in results},
        "aggregation": "Each result averages its two timing rounds. Seed-level results from separate paired sessions remain distinct measurements.",
    })
    print(json.dumps({"paired_output": str(output), "pair_id": pair_id, "order": plan["order"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
