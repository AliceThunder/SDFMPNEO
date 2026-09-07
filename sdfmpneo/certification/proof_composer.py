from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .basic import residual_to_state_gain


@dataclass(frozen=True)
class ProofNode:
    """One typed bound and its proved propagation target.

    A node may either already live in the common accumulation quantity or carry
    a proved scalar gain to a declared target ``(layer,norm)``.  This prevents a
    field-energy error, a heat-source residual and a thermal-state error from
    being silently added just because all three are finite numbers.
    """

    name: str
    layer: str
    norm: str
    bound: float
    gain_to_next: float = 1.0
    certified: bool = False
    provenance: str = ""
    target_layer: str | None = None
    target_norm: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.layer).strip() or not str(self.norm).strip():
            raise ValueError("proof node name/layer/norm must be non-empty")
        if self.bound < 0 or not np.isfinite(self.bound):
            raise ValueError("proof node bound must be finite and non-negative")
        if self.gain_to_next <= 0 or not np.isfinite(self.gain_to_next):
            raise ValueError("proof gain must be finite and positive")
        if (self.target_layer is None) != (self.target_norm is None):
            raise ValueError("target_layer and target_norm must be supplied together")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("certified proof node requires provenance")

    @property
    def accumulation_type(self) -> tuple[str, str]:
        return (
            self.layer if self.target_layer is None else str(self.target_layer),
            self.norm if self.target_norm is None else str(self.target_norm),
        )

    @property
    def propagated_bound(self) -> float:
        return float(np.nextafter(self.bound * self.gain_to_next, np.inf))


@dataclass(frozen=True)
class TypedPropagationCertificate:
    nodes: tuple[ProofNode, ...]
    accumulation_layer: str | None
    accumulation_norm: str | None
    accumulated_error_bound: float | None
    output_error_bound: float | None

    @property
    def certified(self) -> bool:
        return self.output_error_bound is not None and bool(np.isfinite(self.output_error_bound))


@dataclass(frozen=True)
class ElectroThermalPropagationCertificate:
    residual_nodes: tuple[ProofNode, ...]
    state_nodes: tuple[ProofNode, ...]
    residual_error_bound: float | None
    residual_to_state_gain: float | None
    propagated_residual_state_error_bound: float | None
    total_state_error_bound: float | None
    output_error_bound: float | None
    time_horizon: float | None
    kappa: float

    @property
    def certified(self) -> bool:
        return self.output_error_bound is not None and bool(np.isfinite(self.output_error_bound))


def _require_unique_names(nodes: Sequence[ProofNode]) -> None:
    names = [node.name for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("proof node names must be unique")


def compose_typed_error_certificate(
    nodes: Mapping[str, ProofNode],
    *,
    output_gain: float = 1.0,
) -> TypedPropagationCertificate:
    """Accumulate only bounds proved to reach one common physical quantity."""

    ordered = tuple(nodes.values())
    if not ordered:
        raise ValueError("at least one proof node is required")
    _require_unique_names(ordered)
    if output_gain <= 0 or not np.isfinite(output_gain):
        raise ValueError("output_gain must be finite and positive")

    target_types = {node.accumulation_type for node in ordered}
    if len(target_types) != 1:
        raise ValueError(
            "typed proof nodes do not share a common accumulation layer/norm; "
            "add an explicit proved propagation target before combining them"
        )
    layer, norm = next(iter(target_types))
    if not all(node.certified for node in ordered):
        return TypedPropagationCertificate(ordered, layer, norm, None, None)

    accumulated = float(
        np.nextafter(sum(node.propagated_bound for node in ordered), np.inf)
    )
    output = float(np.nextafter(output_gain * accumulated, np.inf))
    return TypedPropagationCertificate(ordered, layer, norm, accumulated, output)


def compose_electrothermal_error_certificate(
    residual_nodes: Mapping[str, ProofNode],
    *,
    kappa: float,
    time_horizon: float | None,
    state_nodes: Mapping[str, ProofNode] | None = None,
    output_gain: float = 1.0,
    residual_layer: str = "thermal_residual",
    state_layer: str = "thermal_state",
    norm: str = "modal_l2",
) -> ElectroThermalPropagationCertificate:
    """Propagate thermal residual errors to state/output without norm mixing.

    Every residual node must already be certified in the same thermal residual
    norm.  Electromagnetic field, constitutive, mesh and outer-domain errors must
    therefore first be converted to heat-source/thermal-residual bounds by their
    own proved output sensitivities.  The comparison-equation gain is then
    applied exactly for the declared finite horizon; direct thermal projection
    errors may be added only as state-norm nodes afterwards.
    """

    residual = tuple(residual_nodes.values())
    direct_state = tuple(() if state_nodes is None else state_nodes.values())
    if not residual:
        raise ValueError("at least one residual proof node is required")
    _require_unique_names(residual + direct_state)
    if output_gain <= 0 or not np.isfinite(output_gain):
        raise ValueError("output_gain must be finite and positive")

    for node in residual:
        if node.layer != residual_layer or node.norm != norm or node.target_layer is not None:
            raise ValueError("residual nodes must already be in the declared thermal residual norm")
    for node in direct_state:
        if node.layer != state_layer or node.norm != norm or node.target_layer is not None:
            raise ValueError("state nodes must already be in the declared thermal state norm")

    all_nodes = residual + direct_state
    if not all(node.certified for node in all_nodes):
        return ElectroThermalPropagationCertificate(
            residual,
            direct_state,
            None,
            None,
            None,
            None,
            None,
            time_horizon,
            float(kappa),
        )

    residual_bound = float(np.nextafter(sum(node.bound for node in residual), np.inf))
    gain = float(residual_to_state_gain(float(kappa), time_horizon))
    propagated = float(np.nextafter(residual_bound * gain, np.inf))
    direct = float(sum(node.bound for node in direct_state))
    total_state = float(np.nextafter(propagated + direct, np.inf))
    output = float(np.nextafter(output_gain * total_state, np.inf))
    return ElectroThermalPropagationCertificate(
        residual_nodes=residual,
        state_nodes=direct_state,
        residual_error_bound=residual_bound,
        residual_to_state_gain=gain,
        propagated_residual_state_error_bound=propagated,
        total_state_error_bound=total_state,
        output_error_bound=output,
        time_horizon=None if time_horizon is None else float(time_horizon),
        kappa=float(kappa),
    )
