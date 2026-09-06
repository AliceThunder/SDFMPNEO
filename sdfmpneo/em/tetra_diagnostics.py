from __future__ import annotations

from typing import Mapping

import numpy as np

from sdfmpneo.spatial.nedelec_weighted import assemble_weighted_nedelec_mass
from sdfmpneo.spatial.tetra3d import TetrahedralComplex3D

from .diagnostics import RegionLossProjector
from .tetra import TetrahedralApsiDiscretization


def build_tetrahedral_region_loss_projector(
    mesh: TetrahedralComplex3D,
    discretization: TetrahedralApsiDiscretization,
    *,
    conductivity_reference_tetra: np.ndarray,
    conductivity_temperature_slope_tetra: np.ndarray,
    thermal_mode_local_values: np.ndarray,
    regions: Mapping[str, np.ndarray],
) -> RegionLossProjector:
    """Build exact affine region Joule-power diagnostics on tetrahedra.

    For region mask chi_R,

        P_R(a) = 1/2 E^H M_{sigma chi_R}(a) E.

    The reference conductivity is piecewise constant per tetrahedron and the
    temperature perturbation is P1 through the retained thermal modes. The
    corresponding Nedelec matrices are integrated analytically with the same
    barycentric formulas used by the coupled electromagnetic operator.
    """

    sigma0 = np.asarray(conductivity_reference_tetra, dtype=float)
    slope = np.asarray(conductivity_temperature_slope_tetra, dtype=float)
    modes = np.asarray(thermal_mode_local_values, dtype=float)
    if sigma0.shape != (mesh.n_tetrahedra,) or slope.shape != (mesh.n_tetrahedra,):
        raise ValueError("conductivity arrays must have shape (n_tetrahedra,)")
    if modes.shape != (discretization.n_thermal, mesh.n_tetrahedra, 4):
        raise ValueError("thermal_mode_local_values shape mismatch")

    L = discretization.electric_extraction()
    H0: dict[str, np.ndarray] = {}
    H_state: dict[str, np.ndarray] = {}

    for name, mask_value in regions.items():
        mask = np.asarray(mask_value, dtype=bool)
        if mask.shape != (mesh.n_tetrahedra,):
            raise ValueError(f"region {name!r} mask must have shape (n_tetrahedra,)")
        scale0 = sigma0 * mask.astype(float)
        W0 = assemble_weighted_nedelec_mass(mesh, scale_tetra=scale0).toarray()
        H0[name] = 0.5 * (L.conj().T @ W0 @ L)

        blocks = []
        for k in range(discretization.n_thermal):
            Wk = assemble_weighted_nedelec_mass(
                mesh,
                scale_tetra=slope * mask.astype(float),
                p1_factors=modes[k : k + 1],
            ).toarray()
            blocks.append(0.5 * (L.conj().T @ Wk @ L))
        H_state[name] = np.stack(blocks, axis=0)

    return RegionLossProjector(H0=H0, H_state=H_state)
