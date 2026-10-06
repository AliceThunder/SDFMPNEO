from __future__ import annotations


def resolve_torch_device(device: str | None = "auto") -> str:
    """Resolve auto/cuda/mps/cpu without importing torch in non-neural paths."""
    requested = "auto" if device is None else str(device).strip().lower()
    if requested != "auto" and not requested.startswith("cuda") and requested != "mps":
        return str(device)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        if requested == "auto":
            return "cpu"
        raise ImportError(
            "the requested accelerator requires the 'neural' extra"
        ) from exc

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        return str(device)

    mps = getattr(torch.backends, "mps", None)
    if mps is None or not bool(mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return "mps"
