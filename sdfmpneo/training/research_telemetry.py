"""Structured one-shot telemetry for residual-optimizer diagnostics."""
from __future__ import annotations


def trial_event(monitor, **details):
    """Write one trial diagnostic row without polluting later heartbeats.

    ``TrainingMonitor`` intentionally keeps a small stable public API.  Trial
    diagnostics are optimizer-specific, so this helper writes ephemeral
    ``trial_*`` fields into one JSONL snapshot and immediately removes them.
    The monitor lock is re-entrant, making the nested ``_write`` call safe.
    """
    if monitor is None:
        return
    lock = getattr(monitor, "_lock", None)
    data = getattr(monitor, "data", None)
    writer = getattr(monitor, "_write", None)
    if lock is None or not isinstance(data, dict) or not callable(writer):
        return
    payload = {f"trial_{key}": value for key, value in details.items()}
    with lock:
        data.update(payload)
        writer()
        for key in payload:
            data.pop(key, None)


__all__ = ["trial_event"]
