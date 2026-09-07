import numpy as np
import scipy.sparse.linalg as spla

from sdfmpneo import TetrahedralElectroThermalCore
from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    ReciprocalLinearResistivity,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D
from sdfmpneo.tetra_core import _build_thermal_components


def _centered_tetrahedral_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ]
    )
    tetrahedra = np.array(
        [
            [4, 1, 2, 3],
            [0, 4, 2, 3],
            [0, 1, 4, 3],
            [0, 1, 2, 4],
        ],
        dtype=int,
    )
    return TetrahedralComplex3D.build(vertices, tetrahedra)


def _cartesian_tetrahedral_mesh(cells_per_axis=3):
    n = int(cells_per_axis)
    coordinates = np.linspace(0.0, 1.0, n + 1)
    vertices = np.array(
        [
            [coordinates[i], coordinates[j], coordinates[k]]
            for i in range(n + 1)
            for j in range(n + 1)
            for k in range(n + 1)
        ],
        dtype=float,
    )

    def vid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    tetrahedra = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v000 = vid(i, j, k)
                v100 = vid(i + 1, j, k)
                v010 = vid(i, j + 1, k)
                v001 = vid(i, j, k + 1)
                v110 = vid(i + 1, j + 1, k)
                v101 = vid(i + 1, j, k + 1)
                v011 = vid(i, j + 1, k + 1)
                v111 = vid(i + 1, j + 1, k + 1)
                tetrahedra.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v100, v101, v111],
                        [v000, v010, v110, v111],
                        [v000, v010, v011, v111],
                        [v000, v001, v101, v111],
                        [v000, v001, v011, v111],
                    ]
                )
    return TetrahedralComplex3D.build(vertices, np.asarray(tetrahedra, dtype=int))


def test_top_level_nonlinear_reduction_default_uses_no_sparse_lu(monkeypatch):
    mesh = _centered_tetrahedral_mesh()
    copper = np.array([True, True, False, False])
    regions = (
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(5.8e7, 3.93e-3, 293.15),
        ),
        ConductivityRegion("seawater", ~copper, ConstantConductivity(5.0)),
    )
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    core = TetrahedralElectroThermalCore.build_nonlinear(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) / (4.0e-7 * np.pi),
        conductivity_regions=regions,
        temperature_reference_nodal=np.ones(mesh.n_nodes) * 293.15,
        constitutive_relative_error_budget=1e-10,
        rho_cp_tetra=np.ones(mesh.n_tetrahedra),
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra),
        source_current=source,
    )

    def forbidden_sparse_lu(*_args, **_kwargs):
        raise AssertionError("production default invoked sparse LU")

    monkeypatch.setattr(spla, "splu", forbidden_sparse_lu)
    model = core.build_reduced_electromagnetics(
        [np.array([-8.0]), np.array([0.0]), np.array([12.0])],
        requested_energy_state_error=1e-6,
    )
    assert model.reduction_certificate.certified
    assert model.reduction_certificate.maximum_energy_state_error_bound <= 1e-6


def test_top_level_thermal_builder_selects_partial_spectrum_before_full_fallback():
    mesh = _cartesian_tetrahedral_mesh(3)
    thermal = mesh.assemble_p1_thermal(
        rho_cp_tetra=np.ones(mesh.n_tetrahedra),
        conductivity_tetra=np.ones(mesh.n_tetrahedra),
        homogeneous_dirichlet_boundary=True,
    )
    n_free = thermal.M.shape[0]
    assert n_free > 1

    (
        _,
        full_spectrum,
        model,
        certificate,
        _,
        backend,
    ) = _build_thermal_components(
        mesh,
        rho_cp_tetra=np.ones(mesh.n_tetrahedra),
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra),
        initial_temperature_deviation_free=np.zeros(n_free),
        source_dual_bound=0.0,
        requested_state_tolerance=1e-12,
        prefer_partial_spectrum=True,
    )

    assert backend == "partial_certified"
    assert full_spectrum is None
    assert model.rank == 1
    assert certificate is not None and certificate.certified
