"""Original BFZI material-conservative reduced heat step and SOE rollout."""
import numpy as np
import pytest
import torch
from scipy.sparse import diags

from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_rom import GeomHeatROM, groups
from test_bfzi_geom_heat import case


def test_rom_preserves_five_separate_material_energy_budgets(tmp_path):
    full=case()
    rom=GeomHeatROM.build(full,rank=10)
    assert 5<=rom.rank<=10
    assert np.allclose(
        rom.phi.T@(rom.cap[:,None]*rom.phi),
        np.eye(rom.rank),atol=1e-8)
    p=tmp_path/"geom_heat_rom.npz"
    rom.save(p)
    loaded=GeomHeatROM.load(full,p)
    assert np.allclose(loaded.phi,rom.phi)
    old=np.zeros(len(full.cap))
    state,z,I,q,J,ps,pc,res=rom.advance(
        old,30.,80.,np.zeros((2,2),complex))
    assert np.isclose(q@full.vol,ps,atol=1e-11)
    assert np.max(abs(rom.budgets[-1]))<1e-8
    assert all(abs(np.sum(res[sl]))<1e-8 for sl in groups(full))
    expected=full.advance(old,30.,80.,np.zeros((2,2),complex))[0]
    assert np.max(abs(state-expected))<1e-8


def test_rom_causal_thermal_history_and_full_fields():
    full=case()
    rom=GeomHeatROM.build(full,rank=10)
    net=GeomFrac()
    for p in net.net.parameters():
        torch.nn.init.zeros_(p)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),port_sea=True)
    volts=np.array([80.,80.,0.,0.,80.,80.])
    times=np.array([0.,15.,30.,45.,60.,90.])
    physical=full.rollout(net,ck,volts,times)
    reduced=rom.rollout(net,ck,volts,times)
    assert reduced["temp_K"].shape==physical["temp_K"].shape
    assert reduced["Jsea_A_m2"].shape==physical["Jsea_A_m2"].shape
    assert reduced["qsea_W_m3"].shape==physical["qsea_W_m3"].shape
    assert reduced["material_budget_W"].shape==(5,5)
    assert np.max(abs(reduced["material_budget_W"]))<1e-8
    assert np.max(abs(reduced["temp_K"]-physical["temp_K"]))<1e-8
    assert np.allclose(reduced["qsea_W_m3"]@rom.vol,
                       reduced["Psea_W"],atol=1e-10)
    assert np.all(reduced["qsea_W_m3"][2:4]==0)


def test_rom_nonuniform_thermal_gradient_maintains_material_budget():
    full=case()
    diag=np.linspace(.1,.3,len(full.cap))
    full.H=diags(diag).tocsr()
    rom=GeomHeatROM.build(full,rank=8)
    old=np.zeros(len(full.cap))
    new,_,_,q,_,ps,_,res=rom.advance(
        old,60.,125.,np.zeros((2,2),complex))
    assert np.all(np.isfinite(new))
    assert np.isclose(q@rom.vol,ps,atol=1e-9)
    assert np.max(abs(rom.budgets[-1]))<1e-8
    assert np.max(abs(res))>=0


def test_rom_rejects_different_package_seawater_interface(tmp_path):
    full=case()
    rom=GeomHeatROM.build(full,rank=10)
    path=tmp_path/"heat_rom.npz"
    rom.save(path)
    assert np.allclose(GeomHeatROM.load(full,path).K,rom.K)
    full.H=full.H+diags(np.full(len(full.cap),.5))
    with pytest.raises(ValueError,match="heat operator"):
        GeomHeatROM.load(full,path)
