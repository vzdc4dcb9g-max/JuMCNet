from __future__ import annotations

import os
import random

# Configure cuBLAS before this module imports PyTorch or launches CUDA work.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed RNGs and configure cuDNN reproducibility within a fixed environment.

    PYTHONHASHSEED must also be set before launching Python to control the
    current interpreter's hash randomization; setting it here covers children.
    """
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Optional strict algorithm enforcement is currently disabled.
    # Uncomment to also enforce determinism beyond the cuDNN convolution flags.
    # torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reproducibility_settings() -> dict[str, object]:
    """Record actual runtime flags alongside each new experiment."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
