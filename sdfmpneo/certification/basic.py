from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContractionCertificate:
    kappa: float
    long_time_certified: bool


@dataclass(frozen=True)
class StateErrorCertificate:
    total_residual_bound: float
    kappa: float
    state_error_bound: float


def contraction_certificate(lambdas: np.ndarray, J_em: np.ndarray) -> ContractionCertificate:
    lambdas = np.asarray(lambdas, dtype=float)
    J_em = np.asarray(J_em, dtype=float)
    sym = 0.5 * (J_em + J_em.T)
    feedback = float(np.max(np.linalg.eigvalsh(sym)))
    kappa = float(np.min(lambdas) - feedback)
    return ContractionCertificate(kappa=kappa, long_time_certified=(kappa > 0.0))


def state_error_certificate(eta_nn: float, eta_rom: float, eta_em: float, kappa: float) -> StateErrorCertificate:
    total = float(eta_nn + eta_rom + eta_em)
    if kappa <= 0:
        raise ValueError("Uniform-in-time error certificate requires kappa > 0")
    return StateErrorCertificate(total, kappa, total / kappa)
