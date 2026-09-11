import numpy as np

from sdfmpneo.em import ConductivityRegion, ReciprocalLinearResistivity
from sdfmpneo.em.modal_heat import exact_modal_heat_source
from sdfmpneo.em.tetra_nonlinear import NonlinearTetrahedralApsiProblem
from sdfmpneo.spatial import TetrahedralComplex3D


def test_batched_modal_heat_matches_per_mode_loss_operators():
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
    problem = NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(1) / (4.0e-7 * np.pi),
        source_current=np.zeros(mesh.n_edges, dtype=complex),
        temperature_reference_local=np.full((1, 4), 293.15),
        thermal_modes_local=modes,
        conductivity_regions=(region,),
        constitutive_relative_error_budget=1.0e-10,
    )
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
