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
    time_horizon: float | None = None


def contraction_certificate(lambdas: np.ndarray, J_em: np.ndarray) -> ContractionCertificate:
    lambdas = np.asarray(lambdas, dtype=float)
    J_em = np.asarray(J_em, dtype=float)
    sym = 0.5 * (J_em + J_em.T)
    feedback = float(np.max(np.linalg.eigvalsh(sym)))
    kappa = float(np.min(lambdas) - feedback)
    return ContractionCertificate(kappa=kappa, long_time_certified=(kappa > 0.0))


def residual_to_state_gain(kappa: float, time_horizon: float | None = None) -> float:
    """Gain from a uniform additive residual/source bound to state error.

    From the one-sided stability inequality

        d||e||/dt <= -kappa ||e|| + eta,

    zero initial state error gives the exact comparison gain

        (1-exp(-kappa*T))/kappa,

    with continuous limit ``T`` at ``kappa=0``.  A missing time horizon is valid
    only for ``kappa>0`` and then returns the uniform long-time gain ``1/kappa``.
    """

    k = float(kappa)
    if not np.isfinite(k):
        raise ValueError("kappa must be finite")
    if time_horizon is None:
        if k <= 0.0:
            raise ValueError("uniform-in-time propagation requires kappa > 0")
        return 1.0 / k
    T = float(time_horizon)
    if T < 0.0 or not np.isfinite(T):
        raise ValueError("time_horizon must be finite and non-negative")
    if k == 0.0:
        return T
    # expm1 is stable both near zero and for moderate negative kappa.
    return float(-np.expm1(-k * T) / k)


def state_error_from_residual(
    residual_bound: float,
    *,
    kappa: float,
    time_horizon: float | None = None,
) -> StateErrorCertificate:
    eta = float(residual_bound)
    if eta < 0.0 or not np.isfinite(eta):
        raise ValueError("residual_bound must be finite and non-negative")
    gain = residual_to_state_gain(kappa, time_horizon)
    return StateErrorCertificate(
        total_residual_bound=eta,
        kappa=float(kappa),
        state_error_bound=float(np.nextafter(eta * gain, np.inf)),
        time_horizon=None if time_horizon is None else float(time_horizon),
    )


def state_error_certificate(
    eta_nn: float,
    eta_rom: float,
    eta_em: float,
    kappa: float,
    time_horizon: float | None = None,
) -> StateErrorCertificate:
    total = float(eta_nn + eta_rom + eta_em)
    return state_error_from_residual(
        total,
        kappa=float(kappa),
        time_horizon=time_horizon,
    )
