"""GeomFracET neural sea impedance and heat-step smoke/integration checks."""
import numpy as np
import torch

from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_neural import GeomFracET, rollout
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from test_bfzi_geom_heat import case
from examples.bfzi_geom_neural import read, score, scales


def test_neural_heat_head_trainable():
    torch.manual_seed(2)
    net = GeomFracET(ng=22, rank=10)
    h = net.init_state(2)
    u = torch.full((2, 3), 0.2)
    g = torch.zeros(2, 22)
    a = torch.randn(2, 10)
    _, y, nxt = net.step_et(h, u, 15., g, a, torch.ones(10))
    assert y.shape == (2, 14) and nxt.shape == (2, 10)
    assert torch.allclose(nxt, a)
    loss = (nxt-a-0.2).square().mean()
    loss.backward()
    assert net.heat_head[-1].weight.grad.abs().sum() > 0


def test_neural_heat_rollout_preserves_five_budgets():
    full = case()
    rom = GeomHeatROM.build(full, rank=10)
    net = GeomFracET(ng=22, rank=rom.rank)
    for p in net.net.parameters():
        torch.nn.init.zeros_(p)
    ck = dict(src_scale=np.ones(3), y_scale=np.ones(14),
              a_scale=np.ones(rom.rank), port_sea=True)
    t = [0., 15., 30., 45., 60.]
    v = [0., 60., 60., 0., 0.]
    pred = rollout(rom, net, ck, v, t)
    assert pred["temp_K"].shape == (len(t), len(rom.cap))
    assert pred["Jsea_A_m2"].shape == (len(t), rom.ns, 3)
    assert pred["material_budget_W"].shape == (len(t), 5)
    assert np.max(abs(pred["material_budget_W"])) < 1e-6
    assert np.allclose(pred["qsea_W_m3"] @ rom.vol,
                       pred["Psea_W"], atol=1e-10)
    assert np.min(pred["temp_K"]) >= rom.amb-1e-8
    assert pred["heat_implicit_solves_online"] == 0
    assert pred["em_directional_solves_online"] == 0


def test_teacher_training_data_and_loss(tmp_path):
    full = case()
    rom = GeomHeatROM.build(full, rank=10)
    base = GeomFrac()
    for p in base.net.parameters():
        torch.nn.init.zeros_(p)
    ck = dict(src_scale=np.ones(3), y_scale=np.ones(14), port_sea=True)
    t = np.array([0., 15., 30., 45.])
    v = np.array([0., 60., 60., 0.])
    truth = rom.rollout(base, ck, v, t)
    path = tmp_path / "trace.npz"
    np.savez_compressed(path, time_s=t, voltage_v=v,
                        z6=truth["z6"], temp_K=truth["temp_K"])
    row = read(path, rom)
    sx, sy, sa = scales([row])
    g = torch.tensor(np.zeros((1, 22)), dtype=torch.float32)
    net = GeomFracET(ng=22, rank=rom.rank)
    loss = score(net, [row], g, sx, sy, sa, rom=rom, therm_w=.03)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.heat_head[-1].weight.grad.abs().sum() > 0


def test_autonomous_validation_reports_real_error(tmp_path):
    full = case()
    rom = GeomHeatROM.build(full, rank=10)
    base = GeomFrac()
    for p in base.net.parameters():
        torch.nn.init.zeros_(p)
    ck0 = dict(src_scale=np.ones(3), y_scale=np.ones(14), port_sea=True)
    t = np.array([0., 15., 30., 45.])
    v = np.array([0., 60., 60., 0.])
    truth = rom.rollout(base, ck0, v, t)
    path = tmp_path / "history.npz"
    np.savez_compressed(path, time_s=t, voltage_v=v,
                        z6=truth["z6"], temp_K=truth["temp_K"])
    from examples.bfzi_geom_neural import eval_rollout
    row = read(path, rom)
    net = GeomFracET(ng=22, rank=rom.rank)
    ck = dict(src_scale=np.ones(3), y_scale=np.ones(14),
              a_scale=np.ones(rom.rank), port_sea=True)
    err = eval_rollout(rom, net, ck, [row])
    assert len(err) == 1
    assert np.isfinite(err[0]["dynamic_Z_rel_L2"])
    assert np.isfinite(err[0]["temperature_max_K"])
    assert err[0]["material_budget_max_W"] < 1e-6


def test_batched_bfzi_teacher_selects_complete_history(tmp_path):
    full = case()
    rom = GeomHeatROM.build(full, rank=10)
    base = GeomFrac()
    for p in base.net.parameters():
        torch.nn.init.zeros_(p)
    ck = dict(src_scale=np.ones(3), y_scale=np.ones(14), port_sea=True)
    t = np.array([0., 15., 30., 45.])
    v = np.array([[0., 60., 60., 0.], [0., 45., 45., 0.]])
    a = [rom.rollout(base, ck, vv, t) for vv in v]
    path = tmp_path / "two.npz"
    np.savez_compressed(path, time_s=t, voltage_v=v,
                        z6=np.stack([r["z6"] for r in a]),
                        temp_K=np.stack([r["temp_K"] for r in a]))
    row = read(f"{path}#1", rom)
    assert row["q"].shape == (len(t), rom.rank)
    assert row["src"].shape == (len(t), 3)
    assert np.allclose(row["z6"], a[1]["z6"])
    assert row["a"].shape == (len(t), rom.rank)
    import pytest
    with pytest.raises(ValueError, match="needs #history_index"):
        read(path, rom)

def test_left_interval_voltage_right_time_observation():
    """Heat the preceding interval even when present voltage turns off."""
    from sdfmpneo_bfzi.frac_memory import zmat
    from sdfmpneo_bfzi.geom_memory import heat_drive
    full = case()
    rom = GeomHeatROM.build(full, rank=10)
    net = GeomFracET(ng=22, rank=rom.rank)
    for p in net.net.parameters():
        torch.nn.init.zeros_(p)
    ck = dict(src_scale=np.ones(3), y_scale=np.ones(14),
              a_scale=np.ones(rom.rank), port_sea=True)
    t = np.array([0., 15., 30., 45.])
    v = np.array([60., 0., 0., 0.])
    pred = rollout(rom, net, ck, v, t)
    assert np.max(pred["temp_K"][1] - rom.amb) > 1e-3
    assert np.max(abs(pred["current_A"][1])) < 1e-9
    assert np.max(abs(pred["material_budget_W"])) < 1e-6
    # The teacher step (0,15] must see the initial ON voltage.
    exp = heat_drive(v[0], zmat(pred["z6"][0]), rom.ext, rom.port,
                     pred["temp_K"][0, rom.cu], rom.cap[rom.cu])
    assert np.any(exp > 0)
    assert np.allclose(heat_drive(v[1], zmat(pred["z6"][1]), rom.ext,
                                  rom.port, pred["temp_K"][1, rom.cu],
                                  rom.cap[rom.cu]), 0)
