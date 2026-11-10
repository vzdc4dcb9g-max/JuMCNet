from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


CUSTOM_OPS = [
    "dw_silu", "pw_silu", "pw24_silu", "stem_silu",
    "more_ops", "direct", "dw8", "fc",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or non-bundle path: " + relative)
    return path


def initializers(model):
    result = {tensor.name: tensor for tensor in model.graph.initializer}
    if len(result) != len(model.graph.initializer):
        raise ValueError("Duplicate ONNX initializer names.")
    return result


def signature(tensor):
    return tensor.data_type, tuple(tensor.dims)


def interface(model):
    return [
        (item.name, item.type.SerializeToString())
        for group in (model.graph.input, model.graph.output) for item in group
    ]


def graph_signature(model):
    nodes = []
    for node in model.graph.node:
        clean = copy.deepcopy(node)
        clean.ClearField("doc_string")
        nodes.append(clean.SerializeToString())
    return nodes


def build_model(plain, template_plain, template_optimized):
    import onnx

    fresh = initializers(plain)
    old_plain = initializers(template_plain)
    old_optimized = initializers(template_optimized)
    if set(fresh) != set(old_plain) or set(old_plain) != set(old_optimized):
        raise ValueError("Fresh/plain-template/optimized-template initializer names differ; no implicit remapping is permitted.")
    for name in fresh:
        if signature(fresh[name]) != signature(old_plain[name]) or signature(fresh[name]) != signature(old_optimized[name]):
            raise ValueError("Initializer shape or dtype differs: " + name)
        if old_plain[name].SerializeToString() != old_optimized[name].SerializeToString():
            raise ValueError("The supplied template pair does not preserve its initializers: " + name)
    if interface(plain) != interface(template_plain) or interface(plain) != interface(template_optimized):
        raise ValueError("The model input/output interfaces differ.")
    if graph_signature(plain) != graph_signature(template_plain):
        raise ValueError("The fresh plain graph differs from the paired plain template; use a matching export and explicitly validated optimization graph.")
    fresh_opsets = {(item.domain, item.version) for item in plain.opset_import}
    plain_opsets = {(item.domain, item.version) for item in template_plain.opset_import}
    if fresh_opsets != plain_opsets:
        raise ValueError("Fresh and template plain graphs use different ONNX opsets.")
    result = copy.deepcopy(template_optimized)
    del result.graph.initializer[:]
    result.graph.initializer.extend(copy.deepcopy(fresh[name]) for name in old_optimized)
    onnx.checker.check_model(result)
    if any(initializers(result)[name].SerializeToString() != fresh[name].SerializeToString() for name in fresh):
        raise RuntimeError("Initializer transfer verification failed.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an explicitly supplied, compatible JuMCNet optimization graph to a new plain deployment bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True, help="Existing optimized ONNX graph; it is never used as the source of trained weights.")
    parser.add_argument("--template-plain", type=Path, help="Paired plain graph; defaults to replacing '_optimized' with '_plain' in the template filename.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        import onnx
    except ImportError:
        parser.error("Install the PC deployment requirements; the onnx package is required.")
    source = args.bundle.resolve()
    output = args.output_dir.resolve()
    template_path = args.template.resolve()
    if output.exists():
        parser.error("Use a new output directory so the source bundle and old measurements remain intact.")
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    if job.get("variant") != "plain":
        parser.error("The source bundle must be a fresh plain deployment export.")
    if job.get("input_shape") != [1, 3, 224, 224]:
        parser.error("This preserved optimization graph supports fixed FP32 [1,3,224,224] input.")
    prep = job.get("preprocessing", {})
    if prep.get("resize_short_side") != 256 or prep.get("crop_size") != 224 or prep.get("crop") != "center" or prep.get("interpolation") != "bicubic":
        parser.error("Native preprocessing is restricted to the recorded Resize256 + bicubic + center crop224 protocol.")
    plain_path = local_file(source, job["model_file"])
    if digest(plain_path) != job["model_sha256"]:
        parser.error("The source ONNX SHA256 does not match its job manifest.")
    if args.template_plain:
        template_plain_path = args.template_plain.resolve()
    else:
        if "_optimized" not in template_path.stem:
            parser.error("Supply --template-plain when the template filename does not end in '_optimized.onnx'.")
        template_plain_path = template_path.with_name(template_path.name.replace("_optimized", "_plain"))
    for path in (template_path, template_plain_path):
        if not path.is_file():
            parser.error("Missing template: " + str(path))
    try:
        optimized = build_model(
            onnx.load(plain_path), onnx.load(template_plain_path), onnx.load(template_path)
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    rows = list(job["samples"]) + list(job.get("validation", {}).get("samples", []))
    resources = {}
    for row in rows:
        path = local_file(source, row["path"])
        if digest(path) != row["sha256"]:
            parser.error("Image SHA256 mismatch: " + row["path"])
        resources[row["path"]] = path
    output.mkdir(parents=True)
    for relative, path in resources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(plain_path, output / "reference_plain.onnx")
    onnx.save(optimized, output / "model.onnx")
    new_job = copy.deepcopy(job)
    new_job.update({
        "variant": "optimized",
        "model_file": "model.onnx",
        "model_sha256": digest(output / "model.onnx"),
        "reference_plain_model": "reference_plain.onnx",
        "reference_plain_sha256": digest(output / "reference_plain.onnx"),
        "custom_ops": CUSTOM_OPS,
        "native_preprocessing": "cropresize_lut",
        "optimization": {
            "template_sha256": digest(template_path),
            "template_plain_sha256": digest(template_plain_path),
            "source_plain_sha256": digest(plain_path),
            "source_checkpoint_sha256": job["checkpoint"]["sha256"],
            "initializers_transferred": len(optimized.graph.initializer),
            "initializers_equal_to_fresh_plain": True,
            "plain_graph_matches_template": True,
            "numerical_validation": "pending_device_validation",
        },
    })
    new_job.setdefault("validation", {})["optimized_graph"] = "pending_device_validation"
    (output / "job.json").write_text(
        json.dumps(new_job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared optimized bundle: " + str(output))
    print("Numerical equivalence is pending; device.py must compare both graphs before timing.")


if __name__ == "__main__":
    main()
