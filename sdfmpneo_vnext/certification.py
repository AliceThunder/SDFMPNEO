from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PortCertificate:
    reciprocity_defect: float
    minimum_dissipation_eigenvalue: float
    power_closure_error: float
    algebraic_residual: float
    certified: bool


def certify_port_result(
    result,
    *,
    residual_tolerance=1e-10,
    reciprocity_tolerance=1e-9,
    power_tolerance=1e-8,
    passivity_tolerance=1e-12,
) -> PortCertificate:
    Z = np.asarray(
        result.impedance,
        dtype=complex,
    )
    scale = max(
        float(np.linalg.norm(Z)),
        1e-30,
    )
    reciprocity = float(
        np.linalg.norm(Z - Z.T)
        / scale
    )
    R = 0.5 * (
        Z + Z.conj().T
    )
    minimum = float(
        np.min(
            np.linalg.eigvalsh(R)
        )
    )
    probes = [
        np.eye(
            Z.shape[0],
            dtype=complex,
        )[:, k]
        for k in range(Z.shape[0])
    ]
    if Z.shape[0] > 1:
        probes.extend(
            [
                np.ones(
                    Z.shape[0],
                    dtype=complex,
                ),
                (
                    np.arange(
                        1,
                        Z.shape[0] + 1,
                        dtype=float,
                    )
                    + 0.3j
                ),
            ]
        )
    closure = 0.0
    for i in probes:
        pp = float(
            result.port_power(i)
        )
        cp = float(
            result.conductor_power(i)
        )
        denom = max(
            abs(pp),
            abs(cp),
            1e-30,
        )
        closure = max(
            closure,
            abs(pp - cp) / denom,
        )
    residual = float(
        getattr(
            result,
            "normalized_residual",
            0.0,
        )
    )
    certified = bool(
        reciprocity
        <= reciprocity_tolerance
        and minimum
        >= -passivity_tolerance
        and closure
        <= power_tolerance
        and residual
        <= residual_tolerance
    )
    return PortCertificate(
        reciprocity,
        minimum,
        closure,
        residual,
        certified,
    )
