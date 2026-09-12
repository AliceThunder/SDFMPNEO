import numpy as np
import scipy.sparse as sp

from sdfmpneo.em import (
    ConductivityRegion,
    ReciprocalLinearResistivity,
    SparseEnergyReducedEMModel,
)
from sdfmpneo.em.modal_heat import exact_modal_heat_source, heat_source_for_reduced_model
from sdfmpneo.em.tetra_nonlinear import NonlinearTetrahedralApsiProblem
from sdfmpneo.spatial import TetrahedralComplex3D


def _problem():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    mesh = TetrahedralComplex3D.build(vertices, np.array([[0, 1, 2, 3]], dtype=int))
    modes = np.array(
        [
            [[[0.7, 0.2, -0.1, 0.3]]],
            [[[0.1, -0.25, 0.4, 0.2]]],
        ],
        dtype=float,
    ).reshape(2, 1, 4)
    region = ConductivityRegion(
        "conductor",
        np.array([True]),
        ReciprocalLinearResistivity(5.8e7, 0.00393, 293.15),
    )
    return NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(1) / (4.0e-7 * np.pi),
        source_current=np.zeros(mesh.n_edges, dtype=complex),
        temperature_reference_local=np.full((1, 4), 293.15),
        thermal_modes_local=modes,
        conductivity_regions=(region,),
        constitutive_relative_error_budget=1.0e-10,
    )


def test_batched_modal_heat_matches_per_mode_loss_operators():
    problem = _problem()
    rng = np.random.default_rng(7)
    x = rng.normal(size=problem.n_em) + 1j * rng.normal(size=problem.n_em)
    state = np.array([0.25, -0.15])

    expected = np.array(
        [
            np.real(np.vdot(x, problem.loss_operator_sparse(j, state) @ x))
            for j in range(problem.n_thermal)
        ]
    )
    observed = exact_modal_heat_source(problem, x, state)
    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=1e-10)


def test_fused_reduced_modal_heat_avoids_full_em_state_reconstruction(monkeypatch):
    problem = _problem()
    basis = np.eye(problem.n_em, dtype=complex)
    model = SparseEnergyReducedEMModel(
        problem,
        basis,
        reference_energy_metric=sp.eye(problem.n_em, dtype=complex, format="csr"),
    )
    state = np.array([0.12, -0.08])
    rhs = np.linspace(0.3, 0.9, problem.n_em).astype(complex)
    rhs += 1j * np.linspace(-0.2, 0.1, problem.n_em)
    expected = model.heat_source_for_rhs(state, rhs)

    def forbidden(*args, **kwargs):
        raise AssertionError("fused reduced modal heat reconstructed the full EM state")

    monkeypatch.setattr(model, "state_for_rhs", forbidden)
    observed = heat_source_for_reduced_model(model, state, rhs)
    np.testing.assert_allclose(observed, expected, rtol=5e-12, atol=1e-9)


def test_fused_reduced_modal_heat_reuses_preprojected_rhs(monkeypatch):
    problem = _problem()
    basis = np.eye(problem.n_em, dtype=complex)
    model = SparseEnergyReducedEMModel(
        problem,
        basis,
        reference_energy_metric=sp.eye(problem.n_em, dtype=complex, format="csr"),
    )
    state = np.array([-0.04, 0.09])
    rhs = np.linspace(-0.5, 0.7, problem.n_em).astype(complex)
    rhs += 1j * np.linspace(0.25, -0.15, problem.n_em)
    prepared = model.rhs_reduced(rhs)
    expected = heat_source_for_reduced_model(model, state, rhs)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared candidate recomputed V^H b")

    monkeypatch.setattr(model, "rhs_reduced", forbidden)
    observed = heat_source_for_reduced_model(
        model,
        state,
        rhs,
        reduced_rhs=prepared,
    )
    np.testing.assert_allclose(observed, expected, rtol=5e-12, atol=1e-9)
