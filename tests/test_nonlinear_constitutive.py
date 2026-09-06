import numpy as np

from sdfmpneo.em import (
    AffineConductivity,
    CompositeCellConductivity,
    ConductivityRegion,
    NonlinearSpatialAphiProblem,
    RectilinearComplex3D,
    ReciprocalLinearResistivity,
    ResidualGreedyEMReducer,
    face_loop_source,
)


def test_reciprocal_linear_resistivity_exact_derivative():
    law = ReciprocalLinearResistivity(
        sigma_ref=500.0,
        alpha=0.004,
        temperature_ref=293.15,
    )
    T = np.array([290.0, 300.0, 330.0])
    analytic = law.derivative(T)
    h = 1e-5
    finite = (law.evaluate(T + h) - law.evaluate(T - h)) / (2 * h)
    assert np.allclose(analytic, finite, rtol=2e-9, atol=2e-9)


def make_nonlinear_problem():
    grid = RectilinearComplex3D.build(
        [0, 0.01, 0.02],
        [0, 0.01, 0.02],
        [0, 0.01],
    )
    shape = grid.shape_cells

    copper = np.zeros(shape, dtype=bool)
    package = np.zeros(shape, dtype=bool)
    copper[0, 0, 0] = True
    package[1, 0, 0] = True
    seawater = ~(copper | package)

    model = CompositeCellConductivity(
        shape,
        [
            ConductivityRegion(
                "copper",
                copper,
                ReciprocalLinearResistivity(
                    sigma_ref=500.0,
                    alpha=0.004,
                    temperature_ref=293.15,
                ),
            ),
            ConductivityRegion(
                "seawater",
                seawater,
                AffineConductivity(
                    sigma_ref=5.0,
                    beta=0.01,
                    temperature_ref=293.15,
                ),
            ),
        ],
    )

    temperature_reference = np.ones(shape) * 293.15
    thermal_modes = np.zeros((2, *shape))
    thermal_modes[0] = 1.0
    thermal_modes[1] = np.indices(shape)[0] - 0.5

    problem = NonlinearSpatialAphiProblem.build(
        grid,
        omega=2 * np.pi * 100e3,
        reluctivity_cell=np.ones(shape) / (4e-7 * np.pi),
        source_current=face_loop_source(grid, 0),
        temperature_reference=temperature_reference,
        thermal_modes=thermal_modes,
        conductivity_model=model,
        thermal_test_cell=thermal_modes,
    )
    return problem


def test_nonlinear_aphi_operator_derivative_matches_finite_difference():
    problem = make_nonlinear_problem()
    a = np.array([3.0, -1.5])
    derivatives = problem.operator_derivatives(a)
    h = 1e-6

    for k in range(problem.n_thermal):
        step = np.zeros(problem.n_thermal)
        step[k] = h
        finite = (problem.operator(a + step) - problem.operator(a - step)) / (2 * h)
        assert np.allclose(derivatives[k], finite, rtol=2e-6, atol=2e-7)


def test_nonlinear_reduced_heat_source_jacobian_matches_finite_difference():
    problem = make_nonlinear_problem()
    candidate_states = [
        np.array([x, y])
        for x in (-2.0, 0.0, 2.0)
        for y in (-1.0, 0.0, 1.0)
    ]
    reduced = ResidualGreedyEMReducer(problem).build(candidate_states, tolerance=1e-9)

    a = np.array([0.7, -0.3])
    _, jacobian = reduced.heat_source_and_jacobian(a)
    h = 1e-5
    finite = np.zeros_like(jacobian)
    for k in range(problem.n_thermal):
        step = np.zeros(problem.n_thermal)
        step[k] = h
        finite[:, k] = (
            reduced.heat_source(a + step) - reduced.heat_source(a - step)
        ) / (2 * h)

    assert np.allclose(jacobian, finite, rtol=3e-5, atol=3e-8)
