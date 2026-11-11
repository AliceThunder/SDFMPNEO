"""The h/R versus time budget compares matched field definitions."""
from examples.bfzi_d2_team7_budget import calc, OBS


def test_matched_endpoint_budget():
    row = {key + "_rel": .12 for key in OBS}
    row.update({"from": "h15", "to": "h10"})
    t = {"observables": {key: {"endpoint": {"rel64_128": .003}}
                         for key in OBS}}
    h = {"runs": {"200": {"h": [row], "R": [{**row, "from": "r15"}]}}}
    out = calc(h, t)
    assert len(out["observations"]) == 2*len(OBS)
    assert all(abs(z["spatial_over_time"]-40) < 1e-10
               for z in out["observations"])
    assert out["status"] == "D2 OPEN"
