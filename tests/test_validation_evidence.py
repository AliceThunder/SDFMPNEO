import numpy as np
import pytest

from sdfmpneo.validation import (
    IndependentValidationCase,
    ValidationEvidenceManifest,
    evaluate_independent_validation,
)


def test_external_validation_requires_traceable_manifest():
    with pytest.raises(ValueError, match="requires an evidence manifest"):
        IndependentValidationCase(
            name="case",
            predicted=np.array([1.0]),
            reference=np.array([1.1]),
            reference_kind="comsol",
        )


def test_validation_manifest_is_training_excluded_and_reported():
    manifest = ValidationEvidenceManifest(
        reference_kind="maxwell",
        source_identifier="independent/maxwell/case-001",
        source_version="2026-09-07",
        provenance="independently generated reference solve; not visible to trainer",
    )
    case = IndependentValidationCase(
        name="z11",
        predicted=np.array([1.0, 2.0]),
        reference=np.array([1.05, 1.95]),
        reference_kind="maxwell",
        certificate_abs_bound=np.array([0.1, 0.1]),
        evidence=manifest,
    )
    report = evaluate_independent_validation([case])
    assert report.all_external_cases_have_provenance
    assert report.all_certified_cases_contain_reference
    assert report.cases[0].source_identifier == manifest.source_identifier


def test_validation_manifest_cannot_be_marked_as_training_data():
    with pytest.raises(ValueError, match="may not be used for training"):
        ValidationEvidenceManifest(
            reference_kind="experiment",
            source_identifier="tank-run-1",
            source_version="v1",
            provenance="lab notebook",
            used_for_training=True,
        )
