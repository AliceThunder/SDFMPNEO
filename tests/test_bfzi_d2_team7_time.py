import numpy as np
from examples.bfzi_d2_team7_time import calc

def test_constant_horizon_and_pilot(tmp_path):
    p=tmp_path/"h10"
    for n in (64,128,256):
        d=p/f"resene200s{n}"
        d.mkdir(parents=True)
        x=np.ones((n,2))
        x[-1,:]=[1+0.2*(64/n)**2,2+0.3*(64/n)**2]
        np.savetxt(d/"phys.dat",x)
    r=calc(tmp_path)
    assert all(np.isclose(x["time_s"],.02) for x in r["rows"])
    assert np.isclose(r["comparison"]["energy_J"]["order"],2)
