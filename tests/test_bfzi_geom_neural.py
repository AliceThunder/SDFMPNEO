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
    assert np.allclose(read(str(path).replace("/", "\\\\"), rom)["a"],row["a"])
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
    assert "q" not in row
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


def test_relative_heat_loss_cancels_teacher_source_and_gradient():
    """The exact residual subtraction needs no expensive teacher heat source."""
    torch.manual_seed(41)
    a = torch.randn(3,32,dtype=torch.float64)
    K = torch.randn(32,32,dtype=torch.float64)
    q = torch.randn(32,dtype=torch.float64)
    prev = torch.randn(32,dtype=torch.float64)
    x = torch.randn(32,dtype=torch.float64,requires_grad=True)
    dt = 15.
    old = ((x-prev)/dt+K@x-q)-((a[2]-a[1])/dt+K@a[2]-q)
    delta = x-a[2]
    new = (delta-(prev-a[1]))/dt+K@delta
    assert torch.allclose(old,new,atol=1e-12,rtol=0)
    grad_old = torch.autograd.grad(old.square().sum(),x,retain_graph=True)[0]
    grad_new = torch.autograd.grad(new.square().sum(),x)[0]
    assert torch.allclose(grad_old,grad_new,atol=1e-11,rtol=0)


def test_closed_loss_uses_post_balance_state(monkeypatch):
    """On a single interval the training loss must see corrected heat, not raw head."""
    import examples.bfzi_geom_neural as mod
    full = case()
    rom = GeomHeatROM.build(full, rank=10)
    net = GeomFracET(ng=22, rank=rom.rank)
    row = dict(
        time=np.array([0.,15.]),voltage=np.array([0.,0.]),
        src=np.zeros((2,3)), y=np.zeros((2,14)),
        a=np.zeros((2,rom.rank)),geom=np.zeros(22),rom=rom,
        temp_K=np.full((2,len(rom.cap)),rom.amb))
    sx,sy,sa=np.ones(3),np.ones(14),np.ones(rom.rank)
    baseline=mod.score(net,[row],None,sx,sy,sa,closed=True,src_mix=1.)
    monkeypatch.setattr(mod,"_heat_balance",
                        lambda row,rom,old,pred,y,sy,dt,voltage:
                        pred+0.1)
    changed=mod.score(net,[row],None,sx,sy,sa,closed=True,src_mix=1.)
    assert changed>baseline+0.001


def test_long_history_weighting_and_gradient():
    from examples.bfzi_geom_neural import _hist_mean
    x=torch.tensor(1.,requires_grad=True)
    losses=[2*x,6*x]
    assert torch.allclose(_hist_mean(losses,[4,40],"equal"),4*x)
    expected=(4*2+40*6)/44
    val=_hist_mean(losses,[4,40],"step")
    assert torch.allclose(val,expected*x)
    val.backward()
    assert torch.allclose(x.grad,torch.tensor(expected))


def test_dynamic_z_loss_matches_full_copper_and_seawater():
    """Training Z loss matches the full port quantity used in acceptance."""
    from types import SimpleNamespace
    from examples.bfzi_geom_neural import _zrel_full

    amb=293.15
    class Port:
        def copper(self,temp):
            t=np.asarray(temp)-amb
            return np.array([.10+.003*t[0]+1j*.001*t[0],
                             .12+.004*t[1]-1j*.002*t[1]])
    rom=SimpleNamespace(amb=amb,port=Port())
    sy=np.ones(14)
    sea=torch.tensor([[.006,.002,.001,-.0001,.004,.001],
                      [.008,.003,.002,-.0002,.006,.0015]],
                     dtype=torch.float32)
    tc=torch.tensor([[3.,6.],[5.,8.]],dtype=torch.float32)
    d=np.asarray(sea)
    dc=np.stack([.003*tc[:,0].numpy(),.001*tc[:,0].numpy(),
                 np.zeros(2),np.zeros(2),
                 .004*tc[:,1].numpy(),-.002*tc[:,1].numpy()],axis=1)
    cold=np.array([.5,1.,.1,.2,.6,1.1])
    row=dict(z6=np.vstack([cold,cold+d+dc]))
    exact=_zrel_full(rom,row,sea,tc,sy)
    assert float(exact)<1e-11
    bad=torch.zeros_like(tc,requires_grad=True)
    err=_zrel_full(rom,row,sea,bad,sy)
    assert float(err)>.01
    err.backward()
    assert torch.isfinite(bad.grad).all()
    assert bad.grad.abs().sum()>0.
