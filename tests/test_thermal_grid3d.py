import numpy as np
import scipy.linalg

from sdfmpneo.thermal import RectilinearThermalFV3D


def test_thermal_fv_is_symmetric_positive_with_dirichlet_outer():
    grid = RectilinearThermalFV3D.build(
        [0, 0.01, 0.02],
        [0, 0.01, 0.02],
        [0, 0.01, 0.02],
    )
    rho_cp = np.ones(grid.shape_cells) * 4.0e6
    conductivity = np.ones(grid.shape_cells) * 0.6
    M, K = grid.assemble(rho_cp, conductivity, dirichlet_outer=True)

    assert np.allclose(K.toarray(), K.toarray().T)
    eigenvalues = scipy.linalg.eigvalsh(K.toarray(), M.toarray())
    assert np.all(eigenvalues > 0)
