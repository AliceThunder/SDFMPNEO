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
        # Pointwise Nedelec electric field and Joule heat at the physical
        # probe. Integrating individual probes does NOT replace volumetric
        # quadrature; use result.medium.volume_power(I) for total energy.
        "E_total_fem_V_m": med["E_total_V_m"],
        "q_total_fem_W_m3": med["q_W_m3"],
    }


def sample_primary_secondary_fields(reference, source_strands, frequency_hz,
                                    points, *, prefer_conductor=False,
                                    nearest=256):
    """Primary-secondary B/E/J/q diagnostics at actual spatial probe points.

    A coherent *field reconstruction* is obtained by adding the analytic
    filament vacuum potential to the medium-minus-vacuum finite-element
    difference. This does not preserve the discrete port-power equality
    and must NOT be silently used as a certified training/thermal reference.
    """
    from .direct3d_aphi import closed_polyline_potential

    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Expected a positive finite frequency")
    p = np.asarray(points, float)
    out = sample_primary_secondary_b(
        reference, source_strands, p,
        prefer_conductor=prefer_conductor, nearest=nearest)
    vacuum = sample_cell_fields(
        reference.vacuum, p, prefer_conductor=prefer_conductor,
        nearest=nearest)
    nports = len(reference.source_edge_load)
    a_primary = np.zeros((nports, len(p), 3), dtype=complex)
    for port, strands in enumerate(source_strands):
        for weight, polygon in strands:
            a_primary[port] += weight*closed_polyline_potential(polygon)(p)
    omega = 2*np.pi*frequency_hz
    e_inc = -1j*omega*a_primary
    e_react = out["E_total_fem_V_m"] - vacuum["E_total_V_m"]
    e_corrected = e_inc + e_react
    sigma = out["sigma_S_m"]
    j_corrected = sigma[None,:,None]*e_corrected
    q_corrected = .5*sigma[None,:]*np.sum(
        abs(e_corrected)**2, axis=-1)
    out.update({
        "A_primary_analytic_T_m": a_primary,
        "E_primary_analytic_V_m": e_inc,
        "E_material_reaction_fem_V_m": e_react,
        "E_total_primary_secondary_V_m": e_corrected,
        "J_total_primary_secondary_A_m2": j_corrected,
        "q_primary_secondary_W_m3": q_corrected,
    })
    return out
