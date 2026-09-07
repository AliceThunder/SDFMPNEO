import numpy as np

from sdfmpneo.analytic import (
    AnalyticRealization,
    analyze_realization_redundancy,
)


def test_realization_redundancy_report_detects_full_rank_system():
    r = AnalyticRealization.decay(0.5, 2.0)
    report = analyze_realization_redundancy(r)
    assert report.original_dimension == 1
    assert report.is_minimal


def test_realization_redundancy_detects_unobservable_state():
    r = AnalyticRealization(
        np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=complex),
        np.array([1.0, 0.0], dtype=complex),
        np.array([1.0, 0.0], dtype=complex),
    )
    report = analyze_realization_redundancy(r)
    assert report.original_dimension == 2
    assert report.redundant_dimension == 1
