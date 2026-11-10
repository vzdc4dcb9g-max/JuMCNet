from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import project_path
from src.reproducibility import seed_worker


class FixedRegionMask:
    """Mask a dataset-specified region identically before train/eval transforms.

    Coordinates are normalized fractions (left, top, right, bottom).
    Source files are never edited. Absent configuration preserves old behavior.
    """

    def __init__(self, box, fill):
        self.box = tuple(float(v) for v in box)
        self.fill = tuple(int(v) for v in fill)
        if len(self.box) != 4 or not (
            0 <= self.box[0] < self.box[2] <= 1
            and 0 <= self.box[1] < self.box[3] <= 1
        ):
            raise ValueError("Invalid normalized mask rectangle")
        if len(self.fill) != 3 or any(v < 0 or v > 255 for v in self.fill):
            raise ValueError("Mask fill must be an RGB triplet")

    def __call__(self, image):
        result = image.copy()
        w, h = image.size
        box = tuple(round(v * (w if i % 2 == 0 else h))
                    for i, v in enumerate(self.box))
        result.paste(self.fill, box)
        return result


class SplitImageDataset(Dataset):
    """Read images from an immutable split manifest.

    Each line is: relative/path.png<TAB>label_index<TAB>class_name
    """

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        transform: Any,
    ) -> None:
        self.data_root = data_root
        self.manifest_path = manifest_path
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {manifest_path}\n"
                "Run first: python tools/create_split.py"
            )

        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        f"{manifest_path}:{line_number} Invalid format; expected three tab-separated columns"
                    )
                relative_path, label_text, class_name = parts
                image_path = data_root / Path(relative_path)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image referenced by the split file not found: {image_path}")
                self.samples.append((image_path, int(label_text), class_name))

        if not self.samples:
            raise ValueError(f"The split file contains no valid samples: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label, _ = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label, image_path.as_posix()


class RepeatDataset(Dataset):
    """Repeat training samples with fresh stochastic transforms per visit."""

    def __init__(self, dataset: Dataset, repeat_factor: int) -> None:
        if repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")
        self.dataset = dataset
        self.repeat_factor = repeat_factor

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat_factor

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]


def build_transforms(
    dataset_config: dict[str, Any],
    experiment_config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    image_size = int(dataset_config["image"]["input_size"])
    mean = dataset_config["normalization"]["mean"]
    std = dataset_config["normalization"]["std"]
    augmentation = copy.deepcopy(experiment_config["augmentation"])
    augmentation_override = dict(
        (model_config or {}).get("augmentation_override", {})
    )
    for key, override in augmentation_override.items():
        if isinstance(override, dict) and isinstance(
            augmentation.get(key), dict
        ):
            augmentation[key].update(override)
        else:
            augmentation[key] = override

    crop = augmentation["random_resized_crop"]
    color = augmentation["color_jitter"]
    erase = augmentation["random_erasing"]

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(crop["scale"]),
                ratio=tuple(crop["ratio"]),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip"])
            ),
            transforms.RandomVerticalFlip(
                p=float(augmentation["vertical_flip"])
            ),
            transforms.RandomAffine(
                degrees=float(augmentation["rotation_degrees"]),
                shear=float(augmentation["shear_degrees"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=float(color["brightness"]),
                contrast=float(color["contrast"]),
                saturation=float(color["saturation"]),
                hue=float(color["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=float(erase["probability"]),
                scale=tuple(erase["scale"]),
                value=erase.get("value", "random"),
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                int(round(image_size / 0.875)),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    mask_config = dataset_config.get("preprocessing", {}).get("watermark_mask", {})
    if mask_config.get("enabled", False):
        mask = FixedRegionMask(mask_config["box"], mask_config["fill"])
        train_transform = transforms.Compose([mask, *train_transform.transforms])
        evaluation_transform = transforms.Compose([mask, *evaluation_transform.transforms])
    return train_transform, evaluation_transform


def create_dataloaders(
    dataset_config: dict[str, Any],
    experiment_config: dict[str, Any],
    train_seed: int,
    include_test: bool = False,
    model_config: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    data_root = project_path(dataset_config["data_root"])
    split_root = project_path(dataset_config["split"]["output_dir"])
    train_transform, evaluation_transform = build_transforms(
        dataset_config, experiment_config, model_config=model_config
    )

    manifest_names = ["train", "val"]
    if include_test:
        manifest_names.append("test")

    datasets: dict[str, Dataset] = {}
    for name in manifest_names:
        transform = train_transform if name == "train" else evaluation_transform
        dataset: Dataset = SplitImageDataset(
            data_root=data_root,
            manifest_path=split_root / f"{name}.txt",
            transform=transform,
        )
        if name == "train":
            repeat_factor = int(
                experiment_config.get("train_repeat_factor", 1)
            )
            if repeat_factor > 1:
                dataset = RepeatDataset(dataset, repeat_factor)
        datasets[name] = dataset

    loader_config = dataset_config["loader"]
    num_workers = int(loader_config["num_workers"])
    batch_size = int(experiment_config["batch_size"])
    pin_memory = bool(loader_config["pin_memory"]) and torch.cuda.is_available()
    persistent_workers = (
        bool(loader_config["persistent_workers"]) and num_workers > 0
    )

    generator = torch.Generator()
    generator.manual_seed(train_seed)

    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
            generator=generator if name == "train" else None,
            drop_last=name == "train",
        )
    return loaders
