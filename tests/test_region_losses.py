import numpy as np

from sdfmpneo.em import (
    RectilinearComplex3D,
    ResidualGreedyEMReducer,
    build_compatible_aphi_from_cells,
    build_region_loss_projector,
    face_loop_source,
)


def test_disjoint_region_losses_add_to_total_joule_loss():
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

    sigma0 = np.zeros(shape)
    sigma0[copper] = 500.0
    sigma0[seawater] = 5.0

    n_thermal = 1
    sigma_state = np.zeros((n_thermal, *shape))
    thermal_test = np.ones((n_thermal, *shape))
    reluctivity = np.ones(shape) / (4e-7 * np.pi)
    source = face_loop_source(grid, 0)

    spatial = build_compatible_aphi_from_cells(
        grid,
        omega=2 * np.pi * 100e3,
        reluctivity_cell=reluctivity,
        conductivity0_cell=sigma0,
        conductivity_state_cell=sigma_state,
        thermal_test_cell=thermal_test,
        source_current=source,
    )
    problem = spatial.discretization.to_parametric_problem()
    state = np.zeros(n_thermal)
    reduced = ResidualGreedyEMReducer(problem).build([state], tolerance=1e-9)

    projector = build_region_loss_projector(
        grid,
        spatial.discretization,
        conductivity0_cell=sigma0,
        conductivity_state_cell=sigma_state,
        regions={
            "copper": copper,
            "seawater": seawater,
            "all": copper | seawater,
        },
    )
    power = projector.evaluate_reduced_model(reduced, state)

    assert power["copper"] >= 0.0
    assert power["seawater"] >= 0.0
    assert np.isclose(
        power["copper"] + power["seawater"],
        power["all"],
        rtol=1e-11,
        atol=1e-13,
    )
