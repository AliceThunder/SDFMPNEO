from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ProofNode:
    """One typed node in the certified error propagation graph.

    The previous ledger only stored additive terms.  This object keeps the
    physical layer and norm explicit so incompatible estimates cannot silently
    enter the final certificate.
    """

    name: str
    layer: str
    norm: str
    bound: float
    gain_to_next: float = 1.0
    certified: bool = False
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.bound < 0 or not np.isfinite(self.bound):
            raise ValueError("proof node bound must be finite and non-negative")
        if self.gain_to_next <= 0 or not np.isfinite(self.gain_to_next):
            raise ValueError("proof gain must be finite and positive")


@dataclass(frozen=True)
class TypedPropagationCertificate:
    """Fail-closed end-to-end output certificate.

    Propagation is explicit:
    EM field -> heat source -> thermal state -> reconstructed output.
    """

    nodes: tuple[ProofNode, ...]
    output_error_bound: float | None

    @property
    def certified(self) -> bool:
        return self.output_error_bound is not None


def compose_typed_error_certificate(
    nodes: Mapping[str, ProofNode],
    *,
    output_gain: float = 1.0,
) -> TypedPropagationCertificate:
    ordered = tuple(nodes.values())
    if output_gain <= 0 or not np.isfinite(output_gain):
        raise ValueError("output_gain must be finite and positive")

    if not all(node.certified for node in ordered):
        return TypedPropagationCertificate(ordered, None)

    value = 0.0
    for node in ordered:
        value += node.bound * node.gain_to_next
    return TypedPropagationCertificate(
        ordered,
        float(np.nextafter(output_gain * value, np.inf)),
    )
