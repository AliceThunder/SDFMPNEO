"""Primary-secondary magnetic reconstruction for the D2 total-A Galerkin FEM.

The true vacuum B of a prescribed closed filament is computed analytically
(Biot-Savart), while the *material-induced difference* is represented by
the SAME tetrahedron's medium-minus-vacuum Nedelec curls:

    B_corrected = B_BiotSavart + curl(A_medium_h - A_vacuum_h).

This is a standard singular-source subtraction for spatial B. It removes
the *discretized vacuum-source* pollution from B probes, but cannot repair
poorly resolved metal eddies, outer-domain truncation, or copper self fields.
Do not substitute this diagnostic for E/Joule field/power extraction.
"""
from __future__ import annotations

import numpy as np

from .direct3d_probe import closed_polyline_magnetic, sample_cell_fields


def sample_primary_secondary_b(reference, source_strands, points, *,
                               prefer_conductor=False, nearest=256):
    """Return per-unit-port complex B arrays of shape (n_ports,n_probes,3).

    The source loops must be the very same strands supplied to
    direct3d_galerkin.solve_closed_ports; separate independent filaments
    are not a faithful approximation to continuous BFZI winding.
    """
    points = np.asarray(points, float)
    ports = len(reference.source_edge_load)
    if len(source_strands) != ports:
        raise ValueError("One matching set of closed strands per solved port")
    med = sample_cell_fields(reference.medium, points,
                             prefer_conductor=prefer_conductor,
                             nearest=nearest)
    vac = sample_cell_fields(reference.vacuum, points,
                             prefer_conductor=prefer_conductor,
                             nearest=nearest)
    if not np.array_equal(med["cell_index"], vac["cell_index"]):
        raise ValueError("Medium/vacuum probes must use identical tetrahedra")
    primary = np.zeros_like(med["B_secondary_T"])
    for port, strands in enumerate(source_strands):
        for weight, polygon in strands:
            if not np.isfinite(weight) or not np.isreal(weight):
                raise ValueError("Strand excitation weight must be finite and real")
            primary[port] += weight*closed_polyline_magnetic(polygon)(points)
    material = med["B_secondary_T"] - vac["B_secondary_T"]
    corrected = primary + material
    return {
        "cell_index": med["cell_index"],
        "sigma_S_m": med["sigma_S_m"],
        "B_primary_analytic_T": primary,
        "B_material_reaction_fem_T": material,
        "B_total_corrected_T": corrected,
        "B_total_raw_fem_T": med["B_secondary_T"],
        "B_vacuum_raw_fem_T": vac["B_secondary_T"],
        "J_total_fem_A_m2": med["J_A_m2"],
        # q_mean is based on cell-mean J, NOT the full quadratic Joule
        # energy. For total energy use result.medium.volume_power(I).
        "q_of_mean_J_W_m3": med["q_mean_W_m3"],
    }
