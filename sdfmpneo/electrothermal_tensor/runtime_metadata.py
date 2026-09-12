"""Minimal source/runtime metadata helpers used by model persistence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


def git_revision() -> str:
    explicit = os.environ.get("SDFMPNEO_GIT_REVISION")
    if explicit:
        return str(explicit)
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            text=True,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def software_environment_summary() -> dict:
    result = {
        "python": sys.version.split()[0],
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


__all__ = ["git_revision", "software_environment_summary"]
