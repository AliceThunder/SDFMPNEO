from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaximumTemperatureErrorCertificate:
    modal_state_error_bound: float
    reconstruction_lipschitz: float
    absolute_temperature_error_bound: float
    certified: bool


def maximum_temperature_lipschitz(thermal_model) -> float:
    """Global Lipschitz constant of ``Tmax`` with respect to modal 2-norm.

    For ``T=T_b+Phi a`` and any modal perturbation ``e``, each nodal/FE test
    point represented by row ``phi(x)`` satisfies

        |phi(x)e| <= ||phi(x)||_2 ||e||_2.

    Therefore ``|max T-max T_hat|`` is bounded by the largest row norm.  No
    location of the hot spot needs to remain fixed under perturbation.
    """

    Phi = np.asarray(thermal_model.Phi, dtype=float)
    if Phi.ndim != 2 or Phi.shape[1] != thermal_model.rank:
        raise ValueError("thermal reconstruction basis shape mismatch")
    return float(np.max(np.linalg.norm(Phi, axis=1)))


def certify_maximum_temperature_error(
    thermal_model,
    modal_state_error_bound: float,
    *,
    certified_state: bool,
) -> MaximumTemperatureErrorCertificate:
    eps = float(modal_state_error_bound)
    if eps < 0.0 or not np.isfinite(eps):
        raise ValueError("modal_state_error_bound must be finite and non-negative")
    L = maximum_temperature_lipschitz(thermal_model)
    return MaximumTemperatureErrorCertificate(
        modal_state_error_bound=eps,
        reconstruction_lipschitz=L,
        absolute_temperature_error_bound=float(np.nextafter(L * eps, np.inf)),
        certified=bool(certified_state),
    )
