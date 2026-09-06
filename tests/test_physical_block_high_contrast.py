from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from sdfmpneo.em import (
    AdaptiveAggregateEnergyPreconditioner,
    ConductivityRegion,
    ConstantConductivity,
    CoupledPairEnergyPreconditioner,
    CurlAuxiliaryPhysicalBlockPreconditioner,
    HierarchicalEnergyPreconditioner,
    MagneticCurlSubsetEnergyPreconditioner,
    MagneticFaceCirculationEnergyPreconditioner,
    MorseFaceCirculationEnergyPreconditioner,
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


def build_refined_magnetic_problem(cells_per_axis=2):
    n = int(cells_per_axis)
    coordinates = np.linspace(0.0, 1.0, n + 1)
    vertices = np.array(
        [[coordinates[i], coordinates[j], coordinates[k]] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)],
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

    mesh = TetrahedralComplex3D.build(vertices, np.asarray(tetrahedra, dtype=int))
    mu0 = 4.0e-7 * np.pi
    reluctivity = np.ones(mesh.n_tetrahedra) / mu0
    magnetic_stiffness, _ = mesh.assemble_nedelec_edge_matrices(
        reluctivity,
        np.zeros(mesh.n_tetrahedra),
    )
    a_basis = mesh.gauge_basis()
    return SimpleNamespace(
        mesh=mesh,
        reluctivity_tetra=reluctivity,
        a_basis=a_basis,
        n_A=a_basis.shape[1],
        magnetic_stiffness=magnetic_stiffness,
    )


def _certify_reduction(problem, factory, requested=1e-6):
    reducer = SparseEnergyResidualGreedyEMReducer(problem, riesz_action_factory=factory)
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
    _certify_reduction(problem, make_physical_block_pcg_riesz_factory(problem))


def test_fixed_coupled_pairs_are_correctly_rejected_on_high_contrast_magnetic_block():
    problem = build_high_contrast_problem()
    factory = make_physical_block_pcg_riesz_factory(
        problem,
        block_action_factory=CoupledPairEnergyPreconditioner.build,
    )
    with pytest.raises(ValueError, match="coupled-pair block Gershgorin"):
        SparseEnergyResidualGreedyEMReducer(problem, riesz_action_factory=factory)


def test_certificate_driven_aggregates_are_correct_but_expose_global_magnetic_fallback():
    problem = build_high_contrast_problem()
    _certify_reduction(
        problem,
        make_physical_block_pcg_riesz_factory(
            problem,
            block_action_factory=AdaptiveAggregateEnergyPreconditioner.build,
        ),
    )
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
    magnetic = AdaptiveAggregateEnergyPreconditioner.build(K_A)
    assert magnetic.lower_spectral_equivalence_bound > 0.0
    assert magnetic.maximum_block_size == K_A.shape[0]


def test_cartesian_curl_subset_is_correct_but_near_global_on_high_contrast_mesh():
    problem = build_high_contrast_problem()
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
    F_A = build_gauge_restricted_magnetic_curl_factor(problem)
    assert np.allclose((F_A.conj().T @ F_A).toarray(), K_A.toarray(), rtol=5e-13, atol=1e-8)
    auxiliary = MagneticCurlSubsetEnergyPreconditioner.build_from_problem(problem)
    assert auxiliary.lower_spectral_equivalence_bound == 1.0
    assert auxiliary.maximum_scc_size >= K_A.shape[0] - 1


def test_face_circulation_auxiliary_proves_two_stage_energy_chain_on_coarse_mesh():
    problem = build_high_contrast_problem()
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).toarray()
    auxiliary = MagneticFaceCirculationEnergyPreconditioner.build_from_problem(problem)
    assert auxiliary.selected_faces.size == problem.n_A
    counts = np.bincount(auxiliary.face_tetra_assignment, minlength=problem.mesh.n_tetrahedra)
    assert np.max(counts) <= 3
    assert auxiliary.lower_spectral_equivalence_bound > 0.0
    P_face = auxiliary.face_energy_matrix().toarray()
    Q = auxiliary.preconditioner_matrix().toarray()
    m = auxiliary.lower_spectral_equivalence_bound
    rem1 = 0.5 * ((K_A - P_face) + (K_A - P_face).conj().T)
    rem2 = 0.5 * ((P_face - m * Q) + (P_face - m * Q).conj().T)
    assert np.min(np.linalg.eigvalsh(rem1).real) >= -256.0 * np.finfo(float).eps * max(np.linalg.norm(K_A, 2), 1.0)
    assert np.min(np.linalg.eigvalsh(rem2).real) >= -256.0 * np.finfo(float).eps * max(np.linalg.norm(P_face, 2), 1.0)


def test_single_level_face_energy_is_correct_but_not_scalable_after_refinement():
    problem = build_refined_magnetic_problem(cells_per_axis=2)
    auxiliary = MagneticFaceCirculationEnergyPreconditioner.build_from_problem(problem)
    assert problem.mesh.n_tetrahedra == 48
    assert problem.n_A == 72
    assert auxiliary.lower_spectral_equivalence_bound > 0.0
    assert auxiliary.maximum_block_size == problem.n_A
    assert auxiliary.final_block_count == 1


def test_topology_generated_morse_face_action_has_strictly_smaller_coarse_problem_after_refinement():
    problem = build_refined_magnetic_problem(cells_per_axis=2)
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).toarray()
    auxiliary = MorseFaceCirculationEnergyPreconditioner.build_from_problem(problem)

    assert auxiliary.dimension == problem.n_A == 72
    assert auxiliary.lower_spectral_equivalence_bound == 1.0
    assert auxiliary.fine_dimension > 0
    assert 0 <= auxiliary.coarse_dimension < auxiliary.dimension
    assert auxiliary.fine_dimension + auxiliary.coarse_dimension == auxiliary.dimension
    assert auxiliary.coarse_fraction < 1.0
    assert np.isfinite(auxiliary.inverse_inf_upper_bound)

    P = auxiliary.auxiliary_matrix().toarray()
    remainder = 0.5 * ((K_A - P) + (K_A - P).conj().T)
    scale = float(np.linalg.norm(K_A, 2))
    assert np.min(np.linalg.eigvalsh(remainder).real) >= -1024.0 * np.finfo(float).eps * max(scale, 1.0)

    rhs = np.arange(1, problem.n_A + 1, dtype=float).astype(complex)
    actual = auxiliary.solve(rhs)
    expected = np.linalg.solve(P, rhs)
    assert np.allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_local_spectral_hierarchy_is_correct_but_still_global_after_refinement():
    problem = build_refined_magnetic_problem(cells_per_axis=2)
    R = problem.a_basis
    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
    hierarchy = HierarchicalEnergyPreconditioner.build(K_A)

    assert hierarchy.dimension == problem.n_A == 72
    assert hierarchy.hierarchy_depth >= 2
    assert hierarchy.final_coarse_dimension < problem.n_A
    assert hierarchy.lower_spectral_equivalence_bound > 0.0
    # Deliberate failure-mode regression: local spectral pairing changes
    # coordinates but the certificate still collapses to one global block.
    assert hierarchy.maximum_transformed_block_size == problem.n_A
    assert hierarchy.transformed_final_block_count == 1

    B_h = hierarchy.transformed_matrix().toarray()
    Q_h = hierarchy.transformed_preconditioner_matrix().toarray()
    m = hierarchy.lower_spectral_equivalence_bound
    remainder = 0.5 * ((B_h - m * Q_h) + (B_h - m * Q_h).conj().T)
    scale = float(np.linalg.norm(B_h, 2))
    assert np.min(np.linalg.eigvalsh(remainder).real) >= -1024.0 * np.finfo(float).eps * max(scale, 1.0)


def test_cartesian_curl_factor_is_rejected_after_refinement_when_selected_scc_is_singular():
    problem = build_refined_magnetic_problem(cells_per_axis=2)
    with pytest.raises(ValueError, match="selected magnetic SCC block is singular"):
        MagneticCurlSubsetEnergyPreconditioner.build_from_problem(problem)


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
    model = _certify_reduction(problem, make_curl_auxiliary_physical_pcg_riesz_factory(problem))
    assert model.reduction_certificate.certified
