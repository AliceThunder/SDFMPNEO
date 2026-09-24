import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    MQSConfig,
    Scene,
    SuperellipseSpiral,
    mixed_reference_convergence,
)


def _scene():
    return Scene(
        (
            CoilObject(
                SuperellipseSpiral(
                    0.022,
                    0.019,
                    0.6,
                    0.001,
                    0.001,
                    exponent=3.0,
                    conductor_width=0.8e-3,
                    conductor_thickness=0.6e-3,
                ),
                ConductorMaterial(
                    5.8e7
                ),
            ),
        ),
        HomogeneousMedium(),
    )


def test_mixed_reference_convergence_resolves_three_independent_axes():
    config = MQSConfig(
        segments_per_turn=5,
        min_segments=5,
        section_degree=0,
        radial_order=3,
        angular_order=8,
        line_order=1,
        section_basis_family="adaptive",
    )
    report = (
        mixed_reference_convergence(
            _scene(),
            5_000.0,
            config,
            tolerance=10.0,
        )
    )
    assert {
        direction.name
        for direction
        in report.directions
    } == {
        "longitudinal",
        "cross_section",
        "quadrature",
    }
    for direction in report.directions:
        assert np.isfinite(
            direction.impedance_relative_change
        )
        assert np.isfinite(
            direction.channel_relative_change
        )
        assert np.isfinite(
            direction.local_loss_relative_change
        )
        assert np.isclose(
            direction.maximum_relative_change,
            max(
                direction.impedance_relative_change,
                direction.channel_relative_change,
                direction.local_loss_relative_change,
            ),
        )
    assert np.isclose(
        report.maximum_relative_change,
        max(
            direction.maximum_relative_change
            for direction
            in report.directions
        ),
    )
    assert report.converged
