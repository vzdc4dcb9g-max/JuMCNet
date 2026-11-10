from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml, project_path  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed, stratified dataset splits")
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/jujube12000.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing split files",
    )
    return parser.parse_args()


def allocate_counts(
    sample_count: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    train_count = int(sample_count * train_ratio)
    val_count = int(sample_count * val_ratio)
    test_count = sample_count - train_count - val_count
    return train_count, val_count, test_count


def main() -> None:
    args = parse_args()
    config = load_yaml(args.dataset_config)
    data_root = project_path(config["data_root"])
    split_config = config["split"]
    split_root = project_path(split_config["output_dir"])
    class_names = list(config["class_names"])

    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1")

    target_files = {
        name: split_root / f"{name}.txt"
        for name in ("train", "val", "test")
    }
    if not args.overwrite and any(path.exists() for path in target_files.values()):
        raise FileExistsError(
            f"Split files already exist in {split_root}. "
            "Existing splits are not overwritten by default to preserve reproducibility. Add --overwrite to regenerate them."
        )

    rng = random.Random(int(split_config["seed"]))
    records: dict[str, list[tuple[str, int, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    class_summary: dict[str, dict[str, int]] = {}

    for label, class_name in enumerate(class_names):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"The class directory contains no images: {class_dir}")
        rng.shuffle(files)
        train_count, val_count, test_count = allocate_counts(
            len(files), train_ratio, val_ratio
        )
        boundaries = (train_count, train_count + val_count)
        groups = {
            "train": files[: boundaries[0]],
            "val": files[boundaries[0] : boundaries[1]],
            "test": files[boundaries[1] :],
        }
        class_summary[class_name] = {
            "total": len(files),
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        for split_name, split_files in groups.items():
            records[split_name].extend(
                (
                    path.relative_to(data_root).as_posix(),
                    label,
                    class_name,
                )
                for path in split_files
            )

    split_root.mkdir(parents=True, exist_ok=True)
    for split_name, rows in records.items():
        rng.shuffle(rows)
        with target_files[split_name].open("w", encoding="utf-8", newline="\n") as file:
            for relative_path, label, class_name in rows:
                file.write(f"{relative_path}\t{label}\t{class_name}\n")

    path_sets = {
        name: {row[0] for row in rows} for name, rows in records.items()
    }
    if path_sets["train"] & path_sets["val"]:
        raise RuntimeError("The training and validation splits overlap")
    if path_sets["train"] & path_sets["test"]:
        raise RuntimeError("The training and test splits overlap")
    if path_sets["val"] & path_sets["test"]:
        raise RuntimeError("The validation and test splits overlap")

    all_paths = path_sets["train"] | path_sets["val"] | path_sets["test"]
    digest = hashlib.sha256(
        "\n".join(sorted(all_paths)).encode("utf-8")
    ).hexdigest()
    summary = {
        "dataset_name": config["dataset_name"],
        "data_root": str(data_root),
        "split_seed": int(split_config["seed"]),
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {name: len(rows) for name, rows in records.items()},
        "total": len(all_paths),
        "class_counts": class_summary,
        "class_order": class_names,
        "path_manifest_sha256": digest,
        "label_counts": {
            name: dict(Counter(row[2] for row in rows))
            for name, rows in records.items()
        },
    }
    with (split_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

