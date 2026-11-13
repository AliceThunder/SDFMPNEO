"""Cold BFZI heat-response subspace and real directional solve accounting."""
from types import SimpleNamespace
import numpy as np
import pytest
from sdfmpneo_bfzi.geom_tangent_subspace import response, directions, build


class EM:
    def __init__(self, cold, vol):
        self.cold, self.vol, self.n = cold, vol, 0

    def solve(self, sea, cu):
        self.n += 1
        delta = np.asarray(sea)-293.15
        x = (np.arange(1, len(delta)+1)*delta*self.vol).sum()
        z = self.cold+x*np.array([[2+1j, 1-.4j],
                                   [1-.4j, 4+.2j]])
        return SimpleNamespace(sea_z=z)


def fixture():
    rng = np.random.default_rng(7)
    ns, n, k = 12, 18, 10
    phi, _ = np.linalg.qr(rng.normal(size=(n, k)))
    sea, _ = np.linalg.qr(phi[:ns, :8])
    rom = SimpleNamespace(phi=phi, K=np.diag(np.linspace(.001,.2,k)))
    head = SimpleNamespace(
        ns=ns, cap=np.ones(n), vol=np.ones(ns),
        weak=rng.normal(size=(ns,2,2)), cu=slice(n-2,n),
        sea0=np.eye(2, dtype=complex), geom=np.arange(16),
        amb=293.15)
    return head, rom, sea


def test_cold_heat_response_probe_reduction():
    head, rom, sea = fixture()
    data = response(head, rom, sea)
    assert data.shape == (8, 50)
    for rank in (2, 4, 8):
        dirs, info = directions(head, rom, sea, rank)
        assert np.allclose(dirs.T@(head.vol[:,None]*dirs),
                           np.eye(rank),atol=1e-10)
        assert 0 < info["explained_fraction"] <= 1+1e-12
        em = EM(head.sea0, head.vol)
        tan, stats = build(em, head, rom, sea, rank)
        assert em.n == tan.nsolve == stats["n_probes"] == rank
        assert tan.method == "cold-response-forward"
        if rank == 8:
            rng = np.random.default_rng(23)
            theta = sea@rng.normal(size=8)
            expected = em.solve(head.amb+theta,
                                np.full(2,head.amb)).sea_z-head.sea0
            assert np.allclose(tan.delta(theta),expected,atol=1e-10)


def test_invalid_thermal_subspace():
    head, rom, sea = fixture()
    with pytest.raises(ValueError):
        directions(head,rom,sea,9)
    with pytest.raises(ValueError):
        build(EM(head.sea0,head.vol),head,rom,sea,4,amplitude=-1)
    rom.K = -np.eye(rom.K.shape[0])
    with pytest.raises(ValueError,match="non-passive"):
        response(head,rom,sea)
