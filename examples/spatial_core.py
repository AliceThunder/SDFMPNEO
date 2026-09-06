"""Run the first shared 3-D electromagnetic-thermal spatial core.

The affine conductivity law used here is intentionally a verification law for
exercising the spatial coupling interface. It is not claimed as the final copper
or seawater constitutive model; production use requires the certified
constitutive-separation layer described in the theory.
"""

import numpy as np

from sdfmpneo.em import (
    RectilinearComplex3D,
    ResidualGreedyEMReducer,
    build_compatible_aphi_from_cells,
    face_loop_source,
)
from sdfmpneo.thermal import RectilinearThermalFV3D, ThermalSpectralModel


def main():
    x = [0.0, 0.01, 0.02]
    y = [0.0, 0.01, 0.02]
    z = [0.0, 0.01]

    em_grid = RectilinearComplex3D.build(x, y, z)
    thermal_grid = RectilinearThermalFV3D.build(x, y, z)
    shape = em_grid.shape_cells

    # Four cells: one conductor-like cell, one package-like cell, two seawater-like cells.
    rho_cp = np.ones(shape) * 4.0e6
    k_thermal = np.ones(shape) * 0.6
    rho_cp[0, 0, 0] = 3.45e6
    k_thermal[0, 0, 0] = 400.0
    rho_cp[1, 0, 0] = 1.5e6
    k_thermal[1, 0, 0] = 0.2

    M_T, K_T = thermal_grid.assemble(rho_cp, k_thermal, dirichlet_outer=True)
    thermal = ThermalSpectralModel.build(M_T, K_T)
    n_thermal = thermal.lambdas.size
    thermal_test = np.stack(
        [thermal.Phi[:, j].reshape(shape) for j in range(n_thermal)],
        axis=0,
    )

    mu0 = 4.0e-7 * np.pi
    reluctivity = np.ones(shape) / mu0

    conductivity0 = np.ones(shape) * 5.0
    conductivity0[0, 0, 0] = 5.8e7
    conductivity0[1, 0, 0] = 0.0

    # Exact affine verification law sigma = sigma0 + beta * DeltaT on each
    # already-conducting material cell. This is only an executable interface test.
    beta = np.ones(shape) * 0.01
    beta[0, 0, 0] = -2.0e5
    beta[1, 0, 0] = 0.0
    conductivity_state = np.stack(
        [beta * thermal_test[j] for j in range(n_thermal)],
        axis=0,
    )

    source = face_loop_source(em_grid, face_index=0, amplitude=1.0)
    spatial = build_compatible_aphi_from_cells(
        em_grid,
        omega=2.0 * np.pi * 100e3,
        reluctivity_cell=reluctivity,
        conductivity0_cell=conductivity0,
        conductivity_state_cell=conductivity_state,
        thermal_test_cell=thermal_test,
        source_current=source,
    )
    problem = spatial.discretization.to_parametric_problem()

    a_ref = np.zeros(n_thermal)
    em = ResidualGreedyEMReducer(problem).build([a_ref], tolerance=1e-8)

    print("cells:", em_grid.n_cells)
    print("nodes/edges/faces:", em_grid.n_nodes, em_grid.n_edges, em_grid.n_faces)
    print("topology defect ||C G||_max:", em_grid.topology_defect())
    print("thermal modes:", n_thermal)
    print("thermal lambdas:", thermal.lambdas)
    print("full EM coordinate dimension:", problem.n_em)
    print("reduced EM rank at reference state:", em.V.shape[1])
    print("EM residual dual norm:", em.residual_dual_norm(a_ref))
    print("reduced thermal heat source:", em.heat_source(a_ref))


if __name__ == "__main__":
    main()
