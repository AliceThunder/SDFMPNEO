from __future__ import annotations

from hashlib import sha256


_UINT64_RANGE = 1 << 64


def _validation_score(index: int, seed: int) -> int:
    """Deterministic sample-index score independent of dataset length."""
    payload = f"sdfmpneo-vnext-validation:{int(seed)}:{int(index)}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def stable_deterministic_split(samples, validation_fraction: float, seed: int):
    """Prefix-stable deterministic train/validation split by teacher index.

    Membership depends only on ``(seed, index)``.  Growing a deterministic
    teacher cache therefore never moves an existing sample between train and
    validation, so validation metrics remain comparable across dataset growth.
    """
    samples = tuple(samples)
    if len(samples) < 2:
        raise ValueError("need at least two samples for train/validation split")

    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")

    limit = int(fraction * _UINT64_RANGE)
    scores = tuple(_validation_score(index, seed) for index in range(len(samples)))
    validation_mask = [score < limit for score in scores]

    # Tiny synthetic test datasets can miss either side of a probabilistic
    # threshold.  Keep both partitions non-empty without changing real-scale
    # prefix stability except for that unavoidable degenerate case.
    if not any(validation_mask):
        validation_mask[min(range(len(samples)), key=scores.__getitem__)] = True
    if all(validation_mask):
        validation_mask[max(range(len(samples)), key=scores.__getitem__)] = False

    train = tuple(
        sample
        for index, sample in enumerate(samples)
        if not validation_mask[index]
    )
    validation = tuple(
        sample
        for index, sample in enumerate(samples)
        if validation_mask[index]
    )
    return train, validation


__all__ = ["stable_deterministic_split"]
