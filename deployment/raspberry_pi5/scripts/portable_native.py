from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from PIL import Image


CUSTOM_OP_LIBRARIES = (
    "dw_silu", "pw_silu", "pw24_silu", "stem_silu",
    "more_ops", "direct", "dw8", "fc",
)


def register_custom_ops(options, native_dir: Path) -> None:
    native_dir = Path(native_dir).resolve()
    for name in CUSTOM_OP_LIBRARIES:
        path = native_dir / (name + ".so")
        if not path.is_file():
            raise FileNotFoundError("Build the native libraries first: " + str(path))
        options.register_custom_ops_library(str(path))


class NativePreprocess:
    def __init__(self, native_dir: Path, mean, std):
        native_dir = Path(native_dir).resolve()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,) or not np.all(std > 0):
            raise ValueError("Expected three RGB normalization means and positive standard deviations.")
        self.table = np.ascontiguousarray(
            (np.arange(256, dtype=np.float32)[None, :] / np.float32(255) - mean[:, None])
            / std[:, None]
        )
        self.lut = ctypes.CDLL(str(native_dir / "lut.so"))
        self.lut.lut_chw.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
        self.lut.lut_chw.restype = None
        self.resize = ctypes.CDLL(str(native_dir / "cropresize.so"))
        self.resize.crop_resize.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        self.resize.crop_resize.restype = ctypes.c_int

    def pixels(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            resized_width, resized_height = (
                (256, int(256 * height / width))
                if width <= height else (int(256 * width / height), 256)
            )
            left = int(round((resized_width - 224) / 2.0))
            top = int(round((resized_height - 224) / 2.0))
            pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        output = np.empty((224, 224, 3), dtype=np.uint8)
        status = self.resize.crop_resize(
            pixels.ctypes.data, width, height, resized_width, resized_height,
            left, top, output.ctypes.data,
        )
        if status:
            raise RuntimeError("Native crop_resize failed with status " + str(status))
        return output

    def __call__(self, path: Path) -> np.ndarray:
        pixels = self.pixels(path)
        output = np.empty((1, 3, 224, 224), dtype=np.float32)
        self.lut.lut_chw(pixels.ctypes.data, self.table.ctypes.data, output.ctypes.data, 224 * 224)
        return output
