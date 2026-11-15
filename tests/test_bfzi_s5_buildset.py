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
