import numpy as np
import pytest

from sdfmpneo.em import (
    AdaptiveAggregateEnergyPreconditioner,
    ConductivityRegion,
    ConstantConductivity,
    CoupledPairEnergyPreconditioner,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SparseEnergyResidualGreedyEMReducer,
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


def test_certificate_driven_aggregates_certify_high_contrast_tetrahedral_reduction_without_block_sparse_lu():
    problem = build_high_contrast_problem()
    factory = make_physical_block_pcg_riesz_factory(
        problem,
        block_action_factory=AdaptiveAggregateEnergyPreconditioner.build,
    )
    _certify_reduction(problem, factory)
