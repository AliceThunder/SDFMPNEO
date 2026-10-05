import numpy as np

from sdfmpneo.unified_gradient_block_maxwell import gradient_operator
from sdfmpneo.unified_hcurl_transfer import (
    build_hcurl_prolongation,
    build_nodal_prolongation,
)
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground


def _background(x, y, z):
    materials = {
        "wire": {
            "electrical_conductivity": 5.8e7,
            "relative_permittivity": 1.0,
            "relative_permeability": 1.0,
            "thermal_conductivity": 1.0,
            "volumetric_heat_capacity": 1.0,
        },
        "package": {
            "electrical_conductivity": 0.0,
            "relative_permittivity": 3.0,
            "relative_permeability": 1.0,
            "thermal_conductivity": 1.0,
            "volumetric_heat_capacity": 1.0,
        },
        "sea": {
            "electrical_conductivity": 5.0,
            "relative_permittivity": 80.0,
            "relative_permeability": 1.0,
            "thermal_conductivity": 1.0,
            "volumetric_heat_capacity": 1.0,
        },
    }
    return OpenBoundaryBackground(
        np.asarray(x, float),
        np.asarray(y, float),
        np.asarray(z, float),
        frequency_hz=1.0e5,
        materials=materials,
        coil_materials=("wire",),
        package_materials=("package",),
        seawater_material="sea",
        ambient_temperature=293.15,
    )


def test_nonnested_hcurl_transfer_commutes_with_gradient():
    coarse = _background(
        [-1.0, -0.25, 0.35, 1.0],
        [-1.0, -0.1, 0.45, 1.0],
        [-1.0, -0.35, 0.2, 1.0],
    )
    fine = _background(
        [-1.0, -0.62, -0.05, 0.31, 0.72, 1.0],
        [-1.0, -0.55, 0.08, 0.51, 0.78, 1.0],
        [-1.0, -0.7, -0.18, 0.27, 0.66, 1.0],
    )
    coarse_axes = (coarse.x, coarse.y, coarse.z)
    P = build_hcurl_prolongation(coarse_axes, fine)
    Q = build_nodal_prolongation(coarse_axes, fine)
    Gc = gradient_operator(coarse, gauge_fixed=False)
    Gf = gradient_operator(fine, gauge_fixed=False)

    difference = (P @ Gc - Gf @ Q).tocsr()
    difference.eliminate_zeros()
    maximum = 0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
    assert maximum <= 2e-12


def test_hcurl_transfer_preserves_constant_vector_field_line_integrals():
    coarse = _background(
        [-1.0, -0.25, 0.35, 1.0],
        [-1.0, -0.1, 0.45, 1.0],
        [-1.0, -0.35, 0.2, 1.0],
    )
    fine = _background(
        [-1.0, -0.62, -0.05, 0.31, 0.72, 1.0],
        [-1.0, -0.55, 0.08, 0.51, 0.78, 1.0],
        [-1.0, -0.7, -0.18, 0.27, 0.66, 1.0],
    )
    P = build_hcurl_prolongation((coarse.x, coarse.y, coarse.z), fine)
    vector = np.array([0.7, -1.1, 0.45])
    coarse_field = np.array(
        [vector[axis] * coarse.edge_lengths[e] for e, (axis, _i, _j, _k) in enumerate(coarse.edge_tuples)],
        dtype=complex,
    )
    expected = np.array(
        [vector[axis] * fine.edge_lengths[e] for e, (axis, _i, _j, _k) in enumerate(fine.edge_tuples)],
        dtype=complex,
    )
    actual = np.asarray(P @ coarse_field, complex).reshape(-1)
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-13)
