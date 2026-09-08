from types import SimpleNamespace

import numpy as np

from sdfmpneo.certification import (
    CertifiedGeometryHeatFeedbackBound,
    certify_geometry_contractivity,
)


class _Model:
    lower = np.array([-1.0])
    upper = np.array([1.0])
    certificate = SimpleNamespace(
        p1_stiffness_ratio=(0.75, 1.3),
        p1_mass_ratio=(0.8, 1.25),
        certified_nondegenerate=True,
    )

    def context(self, geometry):
        return SimpleNamespace(
            M=np.diag([2.0, 3.0]),
            K=np.diag([4.0, 12.0]),
        )


def test_continuous_geometry_contractivity_fails_closed_without_feedback_proof():
    cert = certify_geometry_contractivity(_Model())
    assert cert.status == "indeterminate"
    assert not cert.certified
    # Center generalized thermal eigenvalues are 2 and 4; chart lower bound is
    # (0.75/1.25)*2 = 1.2.
    assert cert.thermal_diffusion_margin_lower_bound <= 1.2
    assert cert.thermal_diffusion_margin_lower_bound > 1.19
    assert cert.contraction_margin_lower_bound is None


def test_continuous_geometry_contractivity_composes_certified_heat_feedback():
    feedback = CertifiedGeometryHeatFeedbackBound(
        0.25,
        certified=True,
        provenance="interval Joule-Jacobian proof over G,a,U box",
    )
    cert = certify_geometry_contractivity(_Model(), heat_feedback=feedback)
    assert cert.certified
    assert cert.contraction_margin_lower_bound > 0.94
    assert cert.contraction_margin_lower_bound < 0.96


def test_nonpositive_composed_margin_remains_indeterminate_not_false_violation():
    feedback = CertifiedGeometryHeatFeedbackBound(
        2.0,
        certified=True,
        provenance="conservative interval proof",
    )
    cert = certify_geometry_contractivity(_Model(), heat_feedback=feedback)
    assert cert.status == "indeterminate"
    assert cert.contraction_margin_lower_bound < 0.0
