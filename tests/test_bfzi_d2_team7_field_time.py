"""Checks for real-time aligned four-line TEAM-7 B/J comparison."""
import numpy as np
import pytest

from examples.bfzi_d2_team7_field_time import calc


def make(root, n, points=3, move=False):
    d = root / "h10"
    out = d / f"resene200s{n}"
    out.mkdir(parents=True, exist_ok=True)
    (d / f"case_ene200s{n}.sif").write_text("\n".join([
        f'Results Directory "resene200s{n}"',
        "BDF Order = 2",
        f"Timestep Sizes(1) = Real {0.02/n}",
        f"TimeStep Intervals(1) = {n}",
        f"Output Intervals(1) = {n}",
        'Real MATC "cos(2*3.14*200*tx[0])"'
    ]))
    a = np.zeros((n, 4, points, 27))
    a[..., 0] = np.arange(1,n+1)[:,None,None]
    a[..., 2] = np.arange(1,5)[None,:,None]
    a[..., 3] = np.arange(points)[None,None,:]
    a[..., 4] = np.linspace(0,.288,points)[None,None,:]
    if move:
        a[...,4] += .001
    a[..., 5] = np.array([.072,.144,.072,.072])[None,:,None]
    a[..., 6] = np.array([.034,.034,.019,0])[None,:,None]
    phase = 2*3.14*200*(np.arange(1,n+1)*.02/n)
    a[..., 22] = np.cos(phase)[:,None,None]*(1+1/n**2)
    a[..., 24] = np.sin(phase)[:,None,None]*(1+1/n)
    a[..., 20] = a[..., 22]
    a[..., 21] = .5*a[..., 22]
    a[..., 23] = .2*a[..., 24]
    a[..., 25] = .4*a[..., 24]
    np.savetxt(out/"f.dat", a.reshape(-1,27))
    (out/"f.dat.names").write_text(
        "23: magnetic flux density e 3\n25: current density e 2\n")


def test_vector_time_orders(tmp_path):
    for n in (64,128,256):
        make(tmp_path,n)
    d=calc(tmp_path,points=3)
    assert np.isclose(d["observables"]["A1_Bz"]["order"],2,atol=1e-8)
    assert np.isclose(d["observables"]["Jy_top"]["order"],1,atol=1e-8)


def test_interlevel_geometry_changed(tmp_path):
    for n in (64,128,256):
        make(tmp_path,n,move=n==256)
    with pytest.raises(ValueError,match="coordinates"):
        calc(tmp_path,points=3)
