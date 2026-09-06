import numpy as np
import scipy.sparse.linalg as spla

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    MorseAuxiliaryPhysicalBlockPreconditioner,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SparseEnergyResidualGreedyEMReducer,
    apsi_physical_energy_metric,
    make_morse_auxiliary_physical_pcg_riesz_factory,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def _high_contrast_problem():
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
    mesh = TetrahedralComplex3D.build(vertices, tetrahedra)
    copper = np.array([True, True, False, False])
    regions = (
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(5.8e7, 3.93e-3, 293.15),
        ),
        ConductivityRegion("seawater", ~copper, ConstantConductivity(5.0)),
    )
    reference = np.ones((mesh.n_tetrahedra, 4)) * 293.15
    modes = np.zeros((1, mesh.n_tetrahedra, 4))
    modes[0, :, :] = np.array([0.2, 0.4, 0.1, 0.3])
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    mu0 = 4.0e-7 * np.pi
    return NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) / mu0,
        source_current=source,
        temperature_reference_local=reference,
        thermal_modes_local=modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=1e-10,
    )


def test_morse_physical_block_has_positive_certificate_without_magnetic_factorization():
    problem = _high_contrast_problem()
    H = apsi_physical_energy_metric(problem.operator_sparse(np.zeros(problem.n_thermal)))
    preconditioner = MorseAuxiliaryPhysicalBlockPreconditioner.build(H, problem=problem)

    assert preconditioner.lower_spectral_equivalence_bound > 0.0
    assert preconditioner.gamma_upper_bound >= 0.0
    assert preconditioner.gamma_certificate_method in (
        "normalized_gershgorin",
        "morse_face_auxiliary_residual_certified_trace",
    )
    assert preconditioner.magnetic_action.lower_spectral_equivalence_bound == 1.0


def test_morse_high_contrast_snapshot_free_rom_runs_with_sparse_lu_disabled(monkeypatch):
    def forbidden_sparse_lu(*_args, **_kwargs):
        raise AssertionError("global sparse LU was used by the Morse production Riesz path")

    monkeypatch.setattr(spla, "splu", forbidden_sparse_lu)
    problem = _high_contrast_problem()
    reducer = SparseEnergyResidualGreedyEMReducer(
        problem,
        riesz_action_factory=make_morse_auxiliary_physical_pcg_riesz_factory(problem),
    )
    states = [np.array([-8.0]), np.array([0.0]), np.array([12.0])]
    model = reducer.build(states, requested_energy_state_error=1e-6)

    assert model.reduction_certificate.certified
    assert model.reduction_certificate.maximum_energy_state_error_bound <= 1e-6
    assert problem._H_metric is None
    for state in states:
        assert model.residual_certificate(state).energy_state_error_bound <= 1e-6
