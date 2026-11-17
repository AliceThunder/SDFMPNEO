"""Train/test two physically distinct terminals with no BFZI queries."""
import numpy as np
from sdfmpneo_bfzi.geom_portself import PortSelf


def _case(a=.75,b=.8):
    return dict(geometry=dict(alpha1=a,alpha2=b,core1_mm=15,
        core2_mm=17,box1_mm=225,box2_mm=228))


def test_portself_has_independent_terminal_shapes(tmp_path):
    cases=[_case(.71,.83),_case(.83,.71),_case(.95,.90)]
    rows=[]
    for k,c in enumerate(cases):
        y=np.zeros(9);y[[0,1,4,5]]=[.25+k*.01,17+k,
                .27-k*.01,18-k*.5]
        y[6]=1.3e-5;y[8]=1.4e-5
        rows.append(dict(case=c,y=y))
    m=PortSelf.fit(rows,2,3.,.01)
    init=dict(z0=np.array([.28,18,.05,1,.28,18]),
              air=np.array([1e-5,1e-6,1e-5]),k=np.eye(5))
    got=m.apply(cases[0],init)
    assert got['z0'].shape==(6,)
    assert np.isfinite(got['z0']).all()
    assert got['z0'][2]==init['z0'][2]
    assert got['z0'][3]==init['z0'][3]
    assert np.all(got['air'][[0,2]]>0)
    assert got['k'] is init['k']
    assert np.allclose(init['air'],[1e-5,1e-6,1e-5])
    m.save(tmp_path/'self.npz')
    assert np.allclose(PortSelf.load(tmp_path/'self.npz').predict(cases[0]),
                       m.predict(cases[0]))
