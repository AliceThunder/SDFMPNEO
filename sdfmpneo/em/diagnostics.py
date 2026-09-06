from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .grid3d import RectilinearComplex3D


@dataclass(frozen=True)
class RegionLossProjector:
    """Region-separated Joule loss operators in gauge-eliminated EM coordinates.

    The operators use the same electric extraction and conductivity Hodge
    construction as the coupled A-phi solve. They are diagnostics only: the
    thermal dynamics continue to use the unified projected heat-source map.
    """

    H0: Mapping[str, np.ndarray]
    H_state: Mapping[str, np.ndarray]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.H0.keys())

    def operator(self, name: str, a: np.ndarray) -> np.ndarray:
        if name not in self.H0:
            raise KeyError(name)
        state = np.asarray(a, dtype=float)
        Hs = self.H_state[name]
        if Hs.shape[0] != state.size:
            raise ValueError("thermal state dimension mismatch")
        return self.H0[name] + np.tensordot(state, Hs, axes=(0, 0))

    def evaluate_state(self, coordinate_state: np.ndarray, a: np.ndarray) -> dict[str, float]:
        x = np.asarray(coordinate_state, dtype=complex)
        return {
            name: float(np.real(np.vdot(x, self.operator(name, a) @ x)))
            for name in self.names
        }

    def evaluate_reduced_model(self, reduced_model, a: np.ndarray) -> dict[str, float]:
        return self.evaluate_state(reduced_model.state(a), a)


def build_region_loss_projector(
    grid: RectilinearComplex3D,
    discretization,
    *,
    conductivity0_cell: np.ndarray,
    conductivity_state_cell: np.ndarray,
    regions: Mapping[str, np.ndarray],
) -> RegionLossProjector:
    """Build P_region(a)=1/2 E^H M_sigma,region(a) E diagnostics.

    Region masks are cellwise booleans. Overlap is allowed mathematically, but
    callers that want an additive copper+seawater decomposition should supply
    disjoint masks. No region-specific resistance formula is introduced.
    """

    sigma0 = np.asarray(conductivity0_cell, dtype=float)
    sigma_state = np.asarray(conductivity_state_cell, dtype=float)
    if sigma0.shape != grid.shape_cells:
        raise ValueError("conductivity0_cell shape mismatch")
    if sigma_state.ndim != 4 or sigma_state.shape[1:] != grid.shape_cells:
        raise ValueError("conductivity_state_cell must have shape (n_thermal,*shape_cells)")

    L = discretization.electric_extraction()
    H0: dict[str, np.ndarray] = {}
    H_state: dict[str, np.ndarray] = {}

    for name, mask_value in regions.items():
        mask = np.asarray(mask_value, dtype=bool)
        if mask.shape != grid.shape_cells:
            raise ValueError(f"region {name!r} mask shape mismatch")

        W0 = grid.edge_hodge(sigma0 * mask).toarray()
        H0[name] = 0.5 * (L.conj().T @ W0 @ L)

        blocks = []
        for k in range(sigma_state.shape[0]):
            Wk = grid.edge_hodge(sigma_state[k] * mask).toarray()
            blocks.append(0.5 * (L.conj().T @ Wk @ L))
        H_state[name] = np.stack(blocks, axis=0)

    return RegionLossProjector(H0=H0, H_state=H_state)
