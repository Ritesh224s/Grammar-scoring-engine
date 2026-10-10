"""Utility functions for reproducible execution, logging, and environment checks."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import random
import sys
import time
from typing import Any

import numpy as np


def get_logger(name: str = "shl_engine") -> logging.Logger:
    """Return a pre-configured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device_info() -> dict[str, Any]:
    """Inspect and return system hardware and accelerator status."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": 0.0,
        "device": "cpu",
    }

    try:
        import torch

        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
            info["device"] = "cuda"
    except ImportError:
        pass

    return info


# List of required and optional packages to audit
PACKAGE_REGISTRY = [
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("scipy", "scipy", True),
    ("sklearn", "scikit-learn", True),
    ("joblib", "joblib", True),
    ("librosa", "librosa", True),
    ("soundfile", "soundfile", True),
    ("transformers", "transformers", True),
    ("matplotlib", "matplotlib", True),
    ("seaborn", "seaborn", True),
    ("tqdm", "tqdm", True),
    ("torch", "torch", True),
    ("spacy", "spacy", False),
    ("catboost", "catboost", False),
]


def check_dependencies() -> list[dict[str, Any]]:
    """Verify installation status and versions of all registered dependencies."""
    results = []
    for import_name, pip_name, required in PACKAGE_REGISTRY:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            installed = True
        except ImportError:
            version = None
            installed = False

        results.append({
            "package": pip_name,
            "required": required,
            "installed": installed,
            "version": version,
        })
    return results


class Timer:
    """Context manager for measuring elapsed execution time."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.2f}s"
