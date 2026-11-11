import numpy as np

from examples.bfzi_d2_team7_dg import mean_sq


def test_linear_field_quadrature():
    v = np.zeros((2,4,3))
    v[0,:,0] = 2
    v[1,:,0] = [0,1,2,3]
    r = mean_sq(v)
    assert np.isclose(r[0], 4)
    assert np.isclose(r[1], (0+1+4+9+6**2)/20)
