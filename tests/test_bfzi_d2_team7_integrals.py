"""Checks for Elmer native integrated scalar output."""
import numpy as np
from examples.bfzi_d2_team7_integrals import compare, read_phys, input_sif


def test_integral_parse(tmp_path):
    (tmp_path / "phys.dat").write_text("9.0 0.4\n10.0 0.5\n")
    (tmp_path / "phys.dat.names").write_text(
        "   1: res: eddy current power\n"
        "   2: res: electromagnetic field energy\n")
    x = read_phys(tmp_path / "phys.dat")
    assert x == {"eddy_power_W": 10., "em_energy_J": 0.5, "rows": 2}


def test_integral_relative():
    data = {k: {"em_energy_J": v, "eddy_power_W": 2*v}
            for k,v in (("h15",1.),("h10",2.),("h075",4.),("r15",3.),("r30",5.))}
    rows = compare(data)
    assert len(rows) == 4
    assert np.isclose(rows[0]["em_energy_J_rel"], 0.5)
    assert np.isclose(rows[0]["eddy_power_W_rel"], 0.5)


def test_period_metric(tmp_path):
    a = np.ones((64, 2))
    a[-1] = [1.01, 1.02]
    (tmp_path / "phys.dat.names").write_text(
        "1: res: eddy current power\n2: res: electromagnetic field energy\n")
    np.savetxt(tmp_path / "phys.dat", a)
    r = read_phys(tmp_path / "phys.dat")
    assert np.isclose(r["power_period_rel"], .01 / 1.01)
    assert np.isclose(r["energy_period_rel"], .02 / 1.02)