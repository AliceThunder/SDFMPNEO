import numpy as np
import pytest
import scipy.sparse.linalg as spla

from sdfmpneo.em import (
    AdaptiveAggregateEnergyPreconditioner,
    ConductivityRegion,
    ConstantConductivity,
    CoupledPairEnergyPreconditioner,
    CurlAuxiliaryPhysicalBlockPreconditioner,
    MagneticCurlSubsetEnergyPreconditioner,
    MagneticFaceCirculationEnergyPreconditioner,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SparseEnergyResidualGreedyEMReducer,
    apsi_physical_energy_metric,
    build_gauge_restricted_magnetic_curl_factor,
    make_curl_auxiliary_physical_pcg_riesz_factory,
    make_physical_block_pcg_riesz_factory,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def build_high_contrast_problem():
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
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    reference = np.ones((mesh.n_tetrahedra, 4)) * 293.15
    modes = np.zeros((1, mesh.n_tetrahedra, 4))
    modes[0, :, :] = np.array([0.2, 0.4, 0.1, 0.3])
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


def _certify_reduction(problem, factory, requested=1e-6):
    reducer = SparseEnergyResidualGreedyEMReducer(
        problem,
        riesz_action_factory=factory,
    )
    states = [np.array([-8.0]), np.array([0.0]), np.array([12.0])]
    model = reducer.build(states, requested_energy_state_error=requested)

    assert model.reduction_certificate.certified
    assert model.reduction_certificate.maximum_energy_state_error_bound <= requested
    assert problem._H_metric is None
    for state in states:
        assert model.residual_certificate(state).energy_state_error_bound <= requested
    return model


def test_physical_block_riesz_factory_certifies_high_contrast_tetrahedral_reduction():
    problem = build_high_contrast_problem()
    factory = make_physical_block_pcg_riesz_factory(problem)
    _certify_reduction(problem, factory)


def test_fixed_coupled_pairs_are_correctly_rejected_on_high_contrast_magnetic_block():
    problem = build_high_contrast_problem()
    factory = make_physical_block_pcg_riesz_factory(
        problem,
        block_action_factory=CoupledPairEnergyPreconditioner.build,
    )
    with pytest.raises(ValueError, match="coupled-pair block Gershgorin"):
        SparseEnergyResidualGreedyEMReducer(
            problem,
            riesz_action_factory=factory,
        )


def test_certificate_driven_aggregates_are_correct_but_expose_global_magnetic_fallback():
    problem = build_high_contrast_problem()
    factory = make_physical_block_pcg_riesz_factory(
        problem,
        block_action_factory=AdaptiveAggregateEnergyPreconditioner.build,
    )
    _certify_reduction(problem, factory)

    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
    magnetic = AdaptiveAggregateEnergyPreconditioner.build(K_A)
    assert magnetic.lower_spectral_equivalence_bound > 0.0
    assert magnetic.aggregation_steps > 0
    assert magnetic.maximum_block_size == K_A.shape[0]


def test_cartesian_curl_subset_is_correct_but_near_global_on_high_contrast_mesh():
    problem = build_high_contrast_problem()
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
    F_A = build_gauge_restricted_magnetic_curl_factor(problem)
    gram = (F_A.conj().T @ F_A).toarray()
    assert np.allclose(gram, K_A.toarray(), rtol=5e-13, atol=1e-8)

    auxiliary = MagneticCurlSubsetEnergyPreconditioner.build_from_problem(problem)
    assert auxiliary.lower_spectral_equivalence_bound == 1.0
    assert auxiliary.maximum_scc_size >= K_A.shape[0] - 1

    S = F_A[auxiliary.selected_rows, :].toarray()
    P = S.conj().T @ S
    remainder = K_A.toarray() - P
    minimum_remainder = float(
        np.min(np.linalg.eigvalsh(0.5 * (remainder + remainder.conj().T)).real)
    )
    scale = float(np.linalg.norm(K_A.toarray(), ord=2))
    assert minimum_remainder >= -128.0 * np.finfo(float).eps * max(scale, 1.0)


def test_face_circulation_auxiliary_proves_two_stage_local_energy_chain():
    problem = build_high_contrast_problem()
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).toarray()
    auxiliary = MagneticFaceCirculationEnergyPreconditioner.build_from_problem(problem)

    assert auxiliary.selected_faces.size == problem.n_A
    counts = np.bincount(
        auxiliary.face_tetra_assignment,
        minlength=problem.mesh.n_tetrahedra,
    )
    assert np.max(counts) <= 3
    assert auxiliary.lower_spectral_equivalence_bound > 0.0
    assert auxiliary.maximum_block_size < problem.n_A

    P_face = auxiliary.face_energy_matrix().toarray()
    Q = auxiliary.preconditioner_matrix().toarray()
    m = auxiliary.lower_spectral_equivalence_bound

    remainder_face = 0.5 * ((K_A - P_face) + (K_A - P_face).conj().T)
    minimum_face = float(np.min(np.linalg.eigvalsh(remainder_face).real))
    scale = float(np.linalg.norm(K_A, ord=2))
    assert minimum_face >= -256.0 * np.finfo(float).eps * max(scale, 1.0)

    remainder_local = 0.5 * ((P_face - m * Q) + (P_face - m * Q).conj().T)
    minimum_local = float(np.min(np.linalg.eigvalsh(remainder_local).real))
    pscale = float(np.linalg.norm(P_face, ord=2))
    assert minimum_local >= -256.0 * np.finfo(float).eps * max(pscale, 1.0)

    rhs = np.arange(1, problem.n_A + 1, dtype=float).astype(complex)
    actual = auxiliary.solve(rhs)
    expected = np.linalg.solve(Q, rhs)
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert np.isfinite(auxiliary.inverse_inf_upper_bound)
    assert auxiliary.inverse_inf_upper_bound > 0.0


def test_curl_auxiliary_gamma_certificate_uses_factorization_free_trace_bound():
    problem = build_high_contrast_problem()
    H = apsi_physical_energy_metric(problem.operator_sparse(np.zeros(problem.n_thermal)))
    preconditioner = CurlAuxiliaryPhysicalBlockPreconditioner.build(H, problem=problem)

    assert preconditioner.gamma_certificate_method == "curl_auxiliary_residual_certified_trace"
    assert preconditioner.gamma_upper_bound >= 0.0


def test_curl_auxiliary_high_contrast_rom_runs_with_splu_globally_disabled(monkeypatch):
    def forbidden_sparse_lu(*_args, **_kwargs):
        raise AssertionError("scipy sparse LU was used by the curl-auxiliary ROM path")

    monkeypatch.setattr(spla, "splu", forbidden_sparse_lu)
    problem = build_high_contrast_problem()
    factory = make_curl_auxiliary_physical_pcg_riesz_factory(problem)
    model = _certify_reduction(problem, factory)
    assert model.reduction_certificate.certified
