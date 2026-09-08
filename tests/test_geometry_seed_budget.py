from pathlib import Path

from sdfmpneo.geometry_research import _geometry_seed_budget


def test_geometry_seed_budget_reserves_correction_capacity():
    assert _geometry_seed_budget(48) == 36
    assert _geometry_seed_budget(64) == 48
    assert _geometry_seed_budget(80) == 60
    assert _geometry_seed_budget(96) == 72


def test_default_uwpt_node_budget_is_64():
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")
    assert '"max_nodes": 64, "max_degree": 3' in source
