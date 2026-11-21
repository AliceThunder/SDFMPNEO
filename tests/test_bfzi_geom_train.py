import numpy as np
import torch

from sdfmpneo_bfzi.geom_train import shared_step


def test_shared_step_matches_deployment_block_solve():
    rng = np.random.default_rng(12)
    rank = 48
    a = rng.normal(size=(rank, rank))
    k = a.T @ a / rank + np.eye(rank) * 0.4
    prev = rng.normal(size=(3, rank))
    guess = rng.normal(size=(3, rank))
    q = rng.normal(size=(3, rank))
    dt = 1.875
    mix = 0.5

    expected = []
    for p, g, src in zip(prev, guess, q):
        mat_s = np.eye(rank - 5) / dt + k[5:, 5:]
        phys = np.linalg.solve(
            mat_s, p[5:] / dt + src[5:] - k[5:, :5] @ g[:5])
        shape = (1. - mix) * g[5:] + mix * phys
        mat_m = np.eye(5) / dt + k[:5, :5]
        mean = np.linalg.solve(
            mat_m, p[:5] / dt + src[:5] - k[:5, 5:] @ shape)
        expected.append(np.r_[mean, shape])

    out = shared_step(
        torch.tensor(prev, dtype=torch.float64),
        torch.tensor(guess, dtype=torch.float64),
        dt,
        torch.tensor(k, dtype=torch.float64),
        torch.tensor(q, dtype=torch.float64),
        mix)
    assert np.allclose(out.detach().numpy(), expected, rtol=1e-11, atol=1e-11)


def test_shared_step_has_gradients_through_neural_guess():
    rank = 48
    k = torch.eye(rank, dtype=torch.float64)
    prev = torch.zeros((2, rank), dtype=torch.float64)
    guess = torch.randn((2, rank), dtype=torch.float64, requires_grad=True)
    q = torch.randn((2, rank), dtype=torch.float64)
    out = shared_step(prev, guess, 1.875, k, q, 0.5)
    out.square().mean().backward()
    assert guess.grad is not None
    assert torch.isfinite(guess.grad).all()
    assert float(guess.grad.norm()) > 0.


def _ctx(rank=48,dtype=torch.float64):
    from sdfmpneo_bfzi.frac_port import PortModel
    from sdfmpneo_bfzi.geom_train import make_ctx
    omega=2*np.pi*200000.
    port=PortModel(omega,np.array([[1e-6,.1e-6],[.1e-6,1.1e-6]]),
        np.array([1.5,1.6]),np.array([.00079,.00081]),
        np.array([5.8e7,5.7e7]),np.array([.00393,.00393]),
        np.array([4e-7*np.pi]*2),np.array([293.15]*2))
    rng=np.random.default_rng(8)
    a=rng.normal(size=(rank,rank))
    k=a.T@a/rank*.01+np.eye(rank)*.02
    cap=np.array([1000.,120.,125.,5.,5.5])
    init=dict(
        cap=cap,k=k,src=np.full((rank,3),1e-3),
        z0=np.array([.08,.03,.004,.002,.09,.035]),
        ext=np.array([[.12+.01j,.002j],[.002j,.13+.01j]]),
        air=np.array([1e-6,.1e-6,1.1e-6]),
        copper0=port.copper(np.full(2,293.15)),
        guard=np.empty(0),gmat=np.empty(0,dtype=int))
    return make_ctx(init,np.zeros(45),port,rank=rank,dtype=dtype),init,port


def test_torch_copper_matches_original_bessel_model_and_gradient():
    from sdfmpneo_bfzi.geom_train import copper_t
    ctx,_,port=_ctx(dtype=torch.float64)
    temp=torch.tensor([[293.15,294.7],[296.2,301.1]],
                      dtype=torch.float64,requires_grad=True)
    got=copper_t(temp,port)
    ref=np.stack([port.copper(x) for x in temp.detach().numpy()])
    assert np.allclose(got.detach().numpy(),ref,rtol=2e-10,atol=2e-12)
    lookup=copper_t(temp,ctx)
    assert np.allclose(lookup.detach().numpy(),ref,rtol=2e-7,atol=2e-9)
    lookup.real.sum().backward()
    assert temp.grad is not None and torch.isfinite(temp.grad).all()


def test_shared_closure_matches_deployment_feedback_step():
    from sdfmpneo_bfzi.geom_train import closure_t
    from sdfmpneo_bfzi.geom_free import shape_step,thermal_step
    from sdfmpneo_bfzi.geom_canon import labels
    ctx,init,port=_ctx(dtype=torch.float64)
    rank=48
    scale=np.sqrt(init["cap"][labels(rank)])
    prev=np.zeros(rank)
    guess=np.zeros(rank)
    guess[:5]=np.array([.5,.25,.2,.15,.1])*np.sqrt(init["cap"])
    guess[5:]=np.random.default_rng(5).normal(0,.015,rank-5)*scale[5:]
    dt=1.875
    voltage=50.
    mix=.5
    geom=np.zeros((1,45))
    phys=shape_step(prev,guess,dt,voltage,init,port,None,geom,
                    scale,293.15,None,dt)
    g=guess.copy()
    g[5:]=(1.-mix)*g[5:]+mix*phys
    expected,_=thermal_step(prev,g,dt,voltage,init,port,None,geom,
                            scale,293.15,None,dt)
    got=closure_t(torch.tensor(prev[None],dtype=torch.float64),
        torch.tensor(guess[None],dtype=torch.float64),dt,ctx,None,None,
        dt,voltage,mix)
    assert np.allclose(got.detach().numpy()[0],expected,rtol=2e-7,atol=2e-8)


def test_shared_history_loss_backpropagates_to_heat_head():
    from sdfmpneo_bfzi.geom_train import hist_loss
    from sdfmpneo_bfzi.geom_neural import GeomFracET
    ctx,_,_=_ctx(dtype=torch.float32)
    net=GeomFracET(ng=45,rank=48,soe_quad="trap",heat_dt=1.875)
    row=dict(
        time=np.array([0.,1.875,3.75]),
        voltage=np.array([50.,50.,50.]),
        src=np.array([[0.,0.,0.],[1.,.01,.01],[1.,.01,.01]]),
        a=np.zeros((3,48)),
        z6=np.tile(np.array([.08,.03,.004,.002,.09,.035]),(3,1)))
    row["a"][1:,0]=.2
    row["a"][2,18]=.1
    row["z6"][1:,0]+=.001
    loss=hist_loss(net,row,ctx,np.ones(3),np.ones(48),
                   tan=None,par=None,shape_mix=.5,src_mix=1.)
    loss.backward()
    grad=net.heat_head[2].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.norm())>0.


def test_shared_history_geometry_weights_are_balanced():
    from sdfmpneo_bfzi.geom_train import _hist_weights
    rows=[
        {"file":"runs/bfzi_s5_A01_drive1.npz","time":np.arange(11)},
        {"file":"runs/bfzi_s5_A01_drive2.npz","time":np.arange(21)},
        {"file":"runs/bfzi_s5_A01_drive3.npz","time":np.arange(31)},
        {"file":"runs/bfzi_s5_B01_drive1.npz","time":np.arange(11)},
    ]
    w=_hist_weights(rows,"geometry")
    assert np.allclose(w.numpy(),[1/3,1/3,1/3,1.])
    assert np.isclose(float(w[:3].sum()),float(w[3]))
    assert np.allclose(_hist_weights(rows,"equal").numpy(),1.)
    assert np.allclose(_hist_weights(rows,"step").numpy(),[10.,20.,30.,10.])
