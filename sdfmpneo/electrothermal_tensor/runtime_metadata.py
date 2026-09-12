"""Software and hardware metadata recorded with neural training artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy


def git_revision() -> str:
    """Return the source revision without making reproducibility depend on git."""
    explicit = os.environ.get("SDFMPNEO_GIT_REVISION")
    if explicit:
        return str(explicit)
    root = Path(__file__).resolve().parents[2]
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        ).strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def software_environment_summary() -> dict:
    result = {
        "git_revision": git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import torch
    except ImportError:
        result["torch"] = None
    else:
        result["torch"] = torch.__version__
    return result


def training_environment_summary(device: str) -> dict:
    result = software_environment_summary()
    result.update(
        {
            "requested_device": str(device),
            "cpu_count": os.cpu_count(),
        }
    )
    try:
        import torch
    except ImportError:
        result.update({"cuda_available": False, "torch_threads": None})
        return result
    result["torch_threads"] = int(torch.get_num_threads())
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        resolved = torch.device(device)
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        result["cuda_device_index"] = index
        result["cuda_device_name"] = torch.cuda.get_device_name(index)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(index))
    return result


__all__ = ["git_revision", "software_environment_summary", "training_environment_summary"]
