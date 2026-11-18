"""Cold-port TX/RX blending must preserve mutual and physical air pairing."""
import numpy as np
import pytest

from sdfmpneo_bfzi.geom_portblend import PortBlend


class _Port:
    def __init__(self,z,air):
        self.z=np.asarray(z,float)
        self.air=np.asarray(air,float)

    def apply(self,case,init):
        out=dict(init)
        out['z0']=self.z.copy()
        out['air']=self.air.copy()
        return out


def test_independent_terminal_blend():
    base=_Port([1,2,3,4,5,6],[7,8,9])
    new=_Port([11,12,13,14,15,16],[17,18,19])
    out=PortBlend(base,new,.25,.75).apply({},dict(z0=np.zeros(6),
                                                air=np.zeros(3)))
    np.testing.assert_allclose(out['z0'],[3.5,4.5,3,4,12.5,13.5])
    np.testing.assert_allclose(out['air'],[9.5,8,16.5])


def test_zero_weight_matches_base():
    base=_Port([1,2,3,4,5,6],[7,8,9])
    new=_Port([11,12,13,14,15,16],[17,18,19])
    out=PortBlend(base,new).apply({},dict(z0=np.zeros(6),
                                         air=np.zeros(3)))
    np.testing.assert_array_equal(out['z0'],base.z)
    np.testing.assert_array_equal(out['air'],base.air)


def test_invalid_blend_rejected():
    base=_Port(np.zeros(6),np.zeros(3))
    with pytest.raises(ValueError):
        PortBlend(base,base,tx=-.01)
    with pytest.raises(ValueError):
        PortBlend(base,base,rx=1.01)
