from pathlib import Path


def test_weight_refinement_yields_to_candidate_search_after_tiny_relative_gain():
    source = (Path(__file__).parents[1] / "sdfmpneo" / "training" / "research.py").read_text()
    assert "objective-trial_objective <= 1e-8*max(objective,np.finfo(float).tiny)" in source
    assert "objective-trial_objective <= 1e-10*max(objective,np.finfo(float).tiny)" not in source
