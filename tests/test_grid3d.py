import numpy as np

from sdfmpneo.em import (
    RectilinearComplex3D,
    ResidualGreedyEMReducer,
    build_compatible_aphi_from_cells,
    face_loop_source,
)


def test_rectilinear_complex_exact_sequence_and_hodges():
    grid = RectilinearComplex3D.build([0, 1, 2], [0, 1, 3], [0, 2])
    assert grid.topology_defect() == 0.0
    assert (grid.curl @ grid.grad).nnz == 0

    coefficient = np.ones(grid.shape_cells) * 4.0
    assert np.all(grid.edge_hodge(coefficient).diagonal() > 0)
    assert np.all(grid.face_hodge(coefficient).diagonal() > 0)

    gauge = grid.gauge_basis()
    assert np.allclose(grid.grad.toarray().T @ gauge, 0.0, atol=1e-12)


def test_cell_assembled_aphi_is_nonsingular_and_reducible():
    grid = RectilinearComplex3D.build(
        [0, 0.01, 0.02],
        [0, 0.01, 0.02],
        [0, 0.01],
    )
    shape = grid.shape_cells
    n_thermal = 2

    reluctivity = np.ones(shape) / (4e-7 * np.pi)
    conductivity0 = np.ones(shape) * 5.0
    conductivity_state = np.zeros((n_thermal, *shape))
    conductivity_state[0] = 0.01
    conductivity_state[1] = -0.005

    thermal_test = np.zeros((n_thermal, *shape))
    thermal_test[0] = 1.0
    thermal_test[1] = np.indices(shape)[0] - 0.5

    source = face_loop_source(grid, 0)
    assert np.allclose(grid.grad.T @ source, 0.0)

    assembly = build_compatible_aphi_from_cells(
        grid,
        omega=2 * np.pi * 1e5,
        reluctivity_cell=reluctivity,
        conductivity0_cell=conductivity0,
        conductivity_state_cell=conductivity_state,
        thermal_test_cell=thermal_test,
        source_current=source,
    )
    problem = assembly.discretization.to_parametric_problem()
    assert np.linalg.matrix_rank(problem.A0) == problem.A0.shape[0]

    states = [
        np.array([x, y])
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
    ]
    reduced = ResidualGreedyEMReducer(problem).build(states, tolerance=1e-9)
    assert max(reduced.residual_dual_norm(a) for a in states) < 1e-8


def test_high_contrast_reference_uses_h_orthonormal_riesz_coordinates():
    grid = RectilinearComplex3D.build(
        [0, 0.01, 0.02],
        [0, 0.01, 0.02],
        [0, 0.01],
    )
    shape = grid.shape_cells

    reluctivity = np.ones(shape) / (4e-7 * np.pi)
    conductivity0 = np.ones(shape) * 5.0
    conductivity0[0, 0, 0] = 5.8e7
    conductivity0[1, 0, 0] = 0.0

    conductivity_state = np.zeros((1, *shape))
    thermal_test = np.ones((1, *shape))
    source = face_loop_source(grid, 0)

    assembly = build_compatible_aphi_from_cells(
        grid,
        omega=2 * np.pi * 100e3,
        reluctivity_cell=reluctivity,
        conductivity0_cell=conductivity0,
        conductivity_state_cell=conductivity_state,
        thermal_test_cell=thermal_test,
        source_current=source,
    )
    problem = assembly.discretization.to_parametric_problem()
    reducer = ResidualGreedyEMReducer(problem)
    V = reducer.initial_basis()

    gram = V.conj().T @ problem.H_metric @ V
    assert np.allclose(gram, np.eye(1), rtol=1e-10, atol=1e-10)
