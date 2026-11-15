"""Focused V4 split and manifest checks."""
from pathlib import Path

from examples.bfzi_geom_v4 import manifest
from examples.bfzi_s5_buildset import check_plan
from examples.bfzi_s5_v4_plan import design


def test_v4_split():
    plan = design()
    tr, val, test = plan["train"], plan["val"], plan["test"]
    assert (len(tr), len(val), len(test)) == (30, 5, 5)
    assert len(set(tr + val + test)) == 40
    assert not set(tr + val + test) & {f"U{i:02d}" for i in range(1, 13)}
    geo = [plan["cases"][k]["geometry"] for k in tr + val + test]
    assert len({str(g) for g in geo}) == 40
    # The native M4 coil has ten turns at 5 mm pitch.
    for g in geo:
        for j in (1, 2):
            assert g[f"core{j}_mm"] + 1e-8 >= (1. - g[f"alpha{j}"]) * 50.
    check_plan({**plan, "train": ["N01"], "val": []})
    assert [v["duration"] for v in plan["cases"]["N01"]["drives"]] == [60, 300, 600]
    assert check_plan({**plan,"train":[],"val":[],"test":["E01"]}) == ([], [])


def test_v4_manifest(tmp_path):
    plan = {"train": ["T01", "N01"], "val": ["T04"], "test": ["E01"]}
    for name in ("T01", "T04"):
        pre = f"bfzi_s5_{name}_ultra_"
        for suffix in ("pilot60_anchor", "pilot60_drive1", "pilot60_drive2",
                       "long600_drive1"):
            (tmp_path / (pre + suffix + ".npz")).touch()
    for suffix in ("anchor", "drive1", "drive2", "drive3"):
        (tmp_path / ("bfzi_s5_N01_ultra_v4_" + suffix + ".npz")).touch()
    out = manifest(plan, ntrain=2, nval=1, runs=tmp_path)
    assert len(out["train"]) == 2
    assert len(out["train"][1]["teacher"]) == 3
    assert out["geometry_split"]["val"] == ["T04"]
