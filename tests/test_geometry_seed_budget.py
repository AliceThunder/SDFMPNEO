from pathlib import Path

from sdfmpneo.geometry_research import _geometry_seed_budget


def test_legacy_geometry_seed_budget_reserves_scalar_prefix_capacity():
    # This helper belongs to the legacy scalar intermediate used by geometry
    # seeding. The residual-state runtime temporarily decouples that source
    # prefix budget from the final independent-state max_nodes budget.
    assert _geometry_seed_budget(48) == 36
    assert _geometry_seed_budget(64) == 48
    assert _geometry_seed_budget(80) == 60
    assert _geometry_seed_budget(96) == 72


def test_default_uwpt_state_budget_is_256():
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")
    assert '"max_nodes": 256, "max_degree": 3' in source
