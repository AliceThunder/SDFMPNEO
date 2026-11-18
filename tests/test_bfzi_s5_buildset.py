"""Non-frozen S5 teacher generation plan and geometry split checks."""
import copy
import json
from pathlib import Path

import pytest

from examples.bfzi_s5_buildset import check_plan


PLAN = Path(__file__).resolve().parents[1] / "runs/geomfrac_s5_plan.json"


def test_s5_plan_separates_geometries_and_histories():
    data=json.loads(PLAN.read_text(encoding="utf-8"))
    tr,val=check_plan(data)
    assert len(tr)==3 and len(val)==1
    assert sum(len(data["cases"][k]["drives"]) for k in tr+val)==8
    for key in tr+val:
        geo=data["cases"][key]["geometry"]
        for j in (1,2):
            alpha=geo[f"alpha{j}"]
            core=geo[f"core{j}_mm"]*.001
            assert .09-core-alpha*.05>0
            assert core-(1-alpha)*.05>=0


def test_train_val_cannot_share_geometry_via_different_materials():
    d=json.loads(PLAN.read_text(encoding="utf-8"))
    bad=copy.deepcopy(d)
    bad["cases"]["T04"]["geometry"]=bad["cases"]["T01"]["geometry"]
    with pytest.raises(ValueError,match="duplicate physical geometry"):
        check_plan(bad)


def test_frozen_case_is_rejected_before_teacher_generation():
    d=json.loads(PLAN.read_text(encoding="utf-8"))
    bad=copy.deepcopy(d)
    bad["train"][0]="U01"
    bad["cases"]["U01"]=bad["cases"]["T01"]
    with pytest.raises(ValueError,match="non-frozen"):
        check_plan(bad)


def test_failed_history_does_not_stop_later_candidates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import examples.bfzi_s5_buildset as gen

    plan={"train":["ANCHOR_BAD","TRACE_BAD","PASS"],"val":[],"test":[],"cases":{}}
    for i,name in enumerate(plan["train"]):
        plan["cases"][name]={
            "geometry":{"case_id":name,"shift":i},
            "thermal":{"ins_t":0.0004},
            "donor":"g1","alpha_per_K":0.01,
            "case_tag":name,
            "drives":[
                {"voltage":60.,"off":[[0.,15.]],"duration":15.}
            ]}
    plan["cases"]["TRACE_BAD"]["drives"].append(
        {"voltage":70.,"off":[[0.,15.]],"duration":15.})

    monkeypatch.setattr(gen,"originals",lambda root:(object(),object()))
    monkeypatch.setattr(gen,"GRIDS",{"ultra":(np.array([0.]),np.array([0.]))})
    monkeypatch.setattr(gen.GeomHeat,"load",
                        staticmethod(lambda path:SimpleNamespace(ext=object())))

    def create(bf,v150,model,**geo):
        if geo["case_id"]=="ANCHOR_BAD":
            raise RuntimeError("synthetic anchor failure")
        return object(),None
    monkeypatch.setattr(gen,"create",create)

    class Head:
        z0=np.array([1.+0.j])
        def save(self,path):
            Path(path).write_bytes(b"anchor")
    head=Head()
    monkeypatch.setattr(gen,"build",lambda *args:object())
    monkeypatch.setattr(gen,"from_skin",lambda *args,**kwargs:head)
    monkeypatch.setattr(gen,"pack",lambda z:z)
    monkeypatch.setattr(gen,"metrics",lambda *args,**kwargs:{})
    monkeypatch.setattr(gen,"teacher_temp",lambda temp,head:temp)
    monkeypatch.setattr(gen,"drive",lambda spec,t:(complex(1.),None))

    class Et:
        model=object()
        cu_ins_layers=np.array([0.0004])
        def zeros(self):return np.zeros(1)
        def em(self,*args):
            return SimpleNamespace(total_z=np.array([1.+0.j])),None,None
    monkeypatch.setattr(gen,"build",lambda *args:Et())

    def march(et,spec,step,inner_step):
        if spec["case_tag"]=="TRACE_BAD" and spec["voltage"]==60.:
            raise RuntimeError("synthetic first-history failure")
        return ({"status":"candidate"},
                {"time_s":np.array([0.,15.]),"temp_K":np.zeros((2,1))})
    monkeypatch.setattr(gen,"march",march)

    manifest,report=gen.generate(plan,bfzi_root=tmp_path,runs=tmp_path,
                                 grid="ultra",duration=15.)
    assert report["cases"]["ANCHOR_BAD"]["status"]=="anchor_error"
    bad=report["cases"]["TRACE_BAD"]
    assert bad["status"]=="partial"
    assert [x["physics"]["status"] for x in bad["traces"]]==[
        "failed","candidate"]
    assert report["cases"]["PASS"]["status"]=="complete"
    assert len(manifest["train"])==2
    assert {Path(x["anchor"]).name for x in manifest["train"]}=={
        "bfzi_s5_TRACE_BAD_ultra_anchor.npz",
        "bfzi_s5_PASS_ultra_anchor.npz"}
