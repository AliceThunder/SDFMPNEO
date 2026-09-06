from sdfmpneo.certification import build_unified_error_certificate


def test_unified_ledger_refuses_full_certificate_when_mesh_or_outer_is_missing():
    common = {
        "constitutive": (1e-4, True, "certified reciprocal remainder"),
        "algebraic": (2e-4, True, "physical residual"),
        "em_rom": (3e-4, True, "Riesz residual"),
        "thermal_rom": (4e-4, True, "spectral tail"),
        "analytic_residual": (5e-4, True, "continuous analytic residual"),
    }
    incomplete = build_unified_error_certificate(common)
    assert not incomplete.fully_certified
    assert set(incomplete.missing_terms) == {"mesh", "outer"}
    assert incomplete.output_error_bound is None

    complete = build_unified_error_certificate(
        {
            **common,
            "mesh": (6e-4, True, "proved spatial reliability"),
            "outer": (7e-4, True, "proved open-domain reliability"),
        },
        output_lipschitz=2.0,
    )
    assert complete.fully_certified
    assert complete.state_error_bound is not None
    assert complete.output_error_bound == 2.0 * complete.state_error_bound
