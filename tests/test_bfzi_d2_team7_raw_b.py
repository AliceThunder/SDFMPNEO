"""One physical Whitney manufactured field for Elmer edge ordering and ASCII AV."""
import numpy as np

from examples.bfzi_d2_team7_raw_b import EL_EDGES, calc
from sdfmpneo_bfzi.direct3d_aphi import MU0


def test_original_whitney_uniform_b(tmp_path):
    x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [0., 0., 1.], [0., 0., -1.]])
    tet = np.array([[0, 1, 2, 3], [1, 0, 2, 4]])
    np.savetxt(tmp_path/"mesh.nodes",
               np.column_stack((np.arange(1, 6), -np.ones(5), x)),
               fmt=["%d", "%d", "%.16g", "%.16g", "%.16g"])
    np.savetxt(tmp_path/"mesh.elements",
               np.column_stack((np.arange(1, 3), np.ones(2),
                                np.full(2, 504), tet+1)), fmt="%d")
    pairs = np.sort(tet[:, EL_EDGES].reshape(-1, 2), axis=1)
    uniq, first = np.unique(pairs, axis=0, return_index=True)
    edges = uniq[np.argsort(first)]
    field = np.array([0., 0., 3.])
    a = np.cross(field, x)/2
    dof = [np.dot((a[i]+a[j])/2, x[j]-x[i]) for i, j in edges]
    vals = np.r_[np.zeros(len(x)), dof]
    n = len(vals)
    with (tmp_path/"edge.result").open("w") as f:
        f.write("ASCII 3\nTime: 1 1 0.1\na\n")
        f.write(f"Perm: {n} {n}\n")
        for i in range(n):
            f.write(f"{i+1} {n-i}\n")
        for v in vals:
            f.write(f"{v:.17g}\n")
    res = calc(tmp_path, tmp_path/"edge.result")
    expected = (2/6) * (3**2)/(2*MU0)
    np.testing.assert_allclose(res["magnetic_energy_J"], expected, rtol=1e-13)
    np.testing.assert_allclose(res["B_rms_T"], 3., rtol=1e-13)
    assert res["edge_dofs"] == len(edges)
