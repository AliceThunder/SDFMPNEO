"""Direct checks for native 200 Hz time-series convergence."""
import math

import numpy as np
import pytest

from examples.bfzi_d2_team7_time import calc


def make_case(root, n, *, changed=False, complete=True):
    d = root / "h10"
    out = d / f"resene200s{n}"
    out.mkdir(parents=True, exist_ok=True)
    sif = "\n".join([
        f'Results Directory "resene200s{n}"',
        "TimeStepping Method = BDF", "BDF Order = 2",
        f"Timestep Sizes(1) = Real {0.02/n}",
        f"TimeStep Intervals(1) = {n}",
        f"Output Intervals(1) = {n}",
        'Component 1 = Variable Time; Real MATC "cos(2*3.14*200*tx[0])"',
        "Sigma = 1" if changed else "Sigma = 0",
    ])
    (d / f"case_ene200s{n}.sif").write_text(sif)
    (d / f"ene200s{n}.log").write_text(
        "MAIN: *** Elmer Solver: ALL DONE ***" if complete else "Time: 2")
    (out / "phys.dat.names").write_text(
        "1: res: eddy current power\n2: res: electromagnetic field energy\n")
    vals = np.tile([2 + 1/(n*n), 3 + 4/(n*n)], (n, 1))
    np.savetxt(out / "phys.dat", vals)


def test_quadratic(tmp_path):
    for n in (64, 128, 256):
        make_case(tmp_path, n)
    d = calc(tmp_path)
    for f, exact in [("eddy_W", 2), ("energy_J", 3)]:
        assert math.isclose(d["comparison"][f]["order"], 2, rel_tol=1e-9)
        assert math.isclose(d["comparison"][f]["extrapolated"], exact, rel_tol=1e-12)
    assert d["rows"][-1]["time_s"] == 0.02


def test_source_mismatch(tmp_path):
    for n in (64, 128, 256):
        make_case(tmp_path, n, changed=n == 128)
    with pytest.raises(ValueError, match="physics differ"):
        calc(tmp_path)


def test_incomplete(tmp_path):
    for n in (64, 128, 256):
        make_case(tmp_path, n, complete=n != 256)
    with pytest.raises(ValueError, match="not complete"):
        calc(tmp_path)


def test_wave_same_physical_times(tmp_path):
    for n in (64, 128, 256):
        make_case(tmp_path, n)
        path = tmp_path / "h10" / f"resene200s{n}" / "phys.dat"
        v = np.loadtxt(path)
        t = (np.arange(n) + 1)*.02/n
        v[:, 0] = 10 + 2*np.cos(4*3.14*200*t) + 1/n**2
        v[:, 1] = .5 + .2*np.sin(4*3.14*200*t) + 1/n**2
        np.savetxt(path, v)
    res = calc(tmp_path)["waveform"]
    assert res["eddy"]["raw_waveform"]["rel128_256"] < 1e-5
    assert res["energy"]["raw_waveform"]["rel128_256"] < 1e-5
