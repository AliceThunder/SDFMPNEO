"""Canonical provenance hashes for neural electrothermal artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .domain import load_reachable_state_domain_report


def _canonical(value):
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, np.ndarray):
        return [_canonical(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("provenance payload cannot contain NaN or infinity")
        return value
    return value


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_domain_report_hash(report) -> str:
    return canonical_sha256(report)


def verified_state_domain_report(path: str | Path):
    """Load a report and verify its declared content hash when present.

    Reports emitted by ``sdfmpneo-domain probe`` are required to carry a hash.
    Direct legacy report mappings remain readable for library compatibility, but
    they do not provide tamper evidence.
    """
    source = Path(path).expanduser().resolve()
    envelope = json.loads(source.read_text(encoding="utf-8"))
    report = load_reachable_state_domain_report(source)
    actual = state_domain_report_hash(report)
    if isinstance(envelope, dict) and envelope.get("kind") == "reachable_state_domain":
        declared = envelope.get("report_hash")
        if declared is None:
            raise ValueError("reachable-state domain report is missing its content hash")
        if str(declared) != actual:
            raise ValueError("reachable-state domain report hash mismatch")
    return report, actual


__all__ = [
    "canonical_sha256",
    "state_domain_report_hash",
    "verified_state_domain_report",
]
