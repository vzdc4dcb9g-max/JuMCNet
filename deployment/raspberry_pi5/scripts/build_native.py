from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ORT_LIBRARIES = (
    "dw_silu", "pw_silu", "pw24_silu", "stem_silu",
    "more_ops", "direct", "dw8", "fc",
)
PREPROCESS_LIBRARIES = ("lut", "cropresize")
ORT_HEADERS = (
    "onnxruntime_c_api.h", "onnxruntime_cxx_api.h",
    "onnxruntime_cxx_inline.h", "onnxruntime_float16.h",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the JuMCNet FP32 custom operators on a Linux ARM64 Raspberry Pi."
    )
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "optimization" / "20260918_crop_resize" / "reference_sources",
        help="Directory containing the preserved C++ sources and ORT API 20 headers.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--cpu", default="cortex-a76")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate all source files and print commands without compiling or writing files.",
    )
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    names = ORT_LIBRARIES + PREPROCESS_LIBRARIES
    required = [source / (name + ".cpp") for name in names]
    required.extend(source / name for name in ORT_HEADERS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing native build inputs:\n" + "\n".join(missing))
    header = (source / "onnxruntime_c_api.h").read_text(encoding="utf-8")
    api = re.search(r"^#define\s+ORT_API_VERSION\s+(\d+)\s*$", header, re.M)
    if api is None or int(api.group(1)) != 20:
        parser.error("These preserved kernels require the supplied ONNX Runtime API 20 headers.")
    commands = []
    for name in names:
        command = [
            args.compiler, "-O3", "-std=c++17", "-mcpu=" + args.cpu,
            "-fPIC", "-shared", "-Wl,--no-undefined",
            "-I" + str(source), str(source / (name + ".cpp")),
            "-o", str(output / (name + ".so")),
        ]
        if name in ORT_LIBRARIES:
            command.append("-lmvec")
        command.append("-lm")
        commands.append((name, command))
    if args.dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        parser.error("Run this build on Linux ARM64; --dry-run is available on other hosts.")
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("C++ compiler not found: " + args.compiler)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "ort_api_version": 20,
        "cpu": args.cpu,
        "headers": {path.name: digest(path) for path in required if path.suffix == ".h"},
        "libraries": [],
    }
    for name, command in commands:
        command[0] = compiler
        temporary = output / (name + ".so.building")
        temporary_command = list(command)
        temporary_command[temporary_command.index("-o") + 1] = str(temporary)
        print("Building " + name + ".so", flush=True)
        try:
            subprocess.run(temporary_command, check=True)
            temporary.replace(output / (name + ".so"))
        finally:
            if temporary.exists():
                temporary.unlink()
        report["libraries"].append({
            "name": name + ".so",
            "source": name + ".cpp",
            "source_sha256": digest(source / (name + ".cpp")),
            "library_sha256": digest(output / (name + ".so")),
            "command": command,
        })
    (output / "build_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Built 10 libraries. Run the deployment input/logit equivalence checks before timing.")


if __name__ == "__main__":
    main()
