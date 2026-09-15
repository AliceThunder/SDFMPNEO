import numpy as np
import scipy.sparse.linalg as spla

from sdfmpneo.unified_geometry import UnifiedUWPTGeometry
from sdfmpneo.unified_gradient_block_maxwell import build_gradient_block
from sdfmpneo.unified_hcurl_transfer import build_hcurl_prolongation
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground
from sdfmpneo.unified_two_level_maxwell import build_two_level_maxwell


MATERIALS = {
    "tx_copper": dict(
        electrical_conductivity=5.8e7,
        resistivity_temperature_coefficient=0.00393,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=1.0,
        thermal_conductivity=400.0,
        volumetric_heat_capacity=3.45e6,
    ),
    "rx_copper": dict(
        electrical_conductivity=5.8e7,
        resistivity_temperature_coefficient=0.00393,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=1.0,
        thermal_conductivity=400.0,
        volumetric_heat_capacity=3.45e6,
    ),
    "tx_package": dict(
        electrical_conductivity=0.0,
        resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=3.0,
        thermal_conductivity=0.2,
        volumetric_heat_capacity=1.5e6,
    ),
    "rx_package": dict(
        electrical_conductivity=0.0,
        resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=3.0,
        thermal_conductivity=0.2,
        volumetric_heat_capacity=1.5e6,
    ),
    "seawater": dict(
        electrical_conductivity=5.0,
        resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=80.0,
        thermal_conductivity=0.6,
        volumetric_heat_capacity=4.1e6,
    ),
}


def _geometry():
    coil = dict(
        shape="circle",
        turns=0.5,
        outer_half_size=0.010,
        pitch=0.002,
        conductor_width=0.001,
        conductor_thickness=0.001,
        corner_radius=0.005,
        angles=[0.0, 0.0, 0.0],
    )
    return UnifiedUWPTGeometry.from_mapping({
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.012]),
        "receiver": dict(coil, translation=[0.0, 0.0, 0.012]),
        "package_half_extent": [0.016, 0.016, 0.004],
    })


def _background(axis):
    return OpenBoundaryBackground(
        axis,
        axis,
        axis,
        frequency_hz=100000.0,
        materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater",
        ambient_temperature=293.15,
    )


def _lgmres(A, rhs, x0, M):
    kwargs = dict(x0=x0, M=M, atol=0.0, maxiter=20, inner_m=20, outer_k=3)
    try:
        return spla.lgmres(A, rhs, rtol=1e-10, **kwargs)
    except TypeError:
        return spla.lgmres(A, rhs, tol=1e-10, **kwargs)


def test_two_level_cycle_solves_small_actual_open_boundary_maxwell_problem():
    geometry = _geometry()
    coarse = _background(np.array([-0.05, -0.022, 0.0, 0.026, 0.05]))
    fine = _background(np.array([-0.05, -0.033, -0.012, 0.006, 0.024, 0.039, 0.05]))

    coarse_context = coarse.geometry_context(geometry, assemble_thermal=False)
    fine_context = fine.geometry_context(geometry, assemble_thermal=False)
    Ac = coarse.em_operator(coarse_context, None)
    Af = fine.em_operator(fine_context, None)
    bc = np.asarray(coarse.rhs_matrix(coarse_context)[:, 0], complex).reshape(-1)
    bf = np.asarray(fine.rhs_matrix(fine_context)[:, 0], complex).reshape(-1)
    xc = np.asarray(spla.spsolve(Ac.tocsc(), bc), complex).reshape(-1)

    P = build_hcurl_prolongation((coarse.x, coarse.y, coarse.z), fine)
    state = {
        "axes": (coarse.x.copy(), coarse.y.copy(), coarse.z.copy()),
        "field": xc,
        "local_geometry": geometry,
        "prolongation": P,
    }
    fine_gradient = build_gradient_block(fine, fine_context, check_topology=True)
    two_level = build_two_level_maxwell(
        Af,
        fine,
        fine_gradient,
        state,
        {
            "linear_ilu_drop_tolerance": 1e-3,
            "linear_ilu_fill_factor": 8.0,
            "linear_transverse_stabilization_factor": 3e-2,
            "linear_two_level_jacobi_weight": 0.5,
        },
    )
    M = two_level.operator(coarse_corrections=2, smoother_sweeps=1)
    x0 = np.asarray(P @ xc, complex).reshape(-1)
    before = np.linalg.norm(bf - Af @ x0) / np.linalg.norm(bf)
    x, info = _lgmres(Af, bf, x0, M)
    after = np.linalg.norm(bf - Af @ x) / np.linalg.norm(bf)

    assert np.isfinite(before)
    assert info == 0
    assert after <= 1e-8
    assert after < before
