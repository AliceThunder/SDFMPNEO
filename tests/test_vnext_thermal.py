import numpy as np

from sdfmpneo_vnext import (
    StableThermalModel,
    lumped_thermal_step,
)


def test_stable_thermal_exact_lumped_step():
    C, G, P = 12.0, 3.0, 7.0
    model = StableThermalModel(
        np.array([[C]]),
        np.array([[G]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        reference_temperature=300.0,
    )
    z = model.advance_constant_power(
        np.array([0.0]),
        np.array([P]),
        1.7,
    )
    expected = lumped_thermal_step(
        1.7,
        P,
        C,
        G,
    )
    assert np.allclose(
        z,
        [expected],
        rtol=2e-13,
        atol=2e-14,
    )
    assert np.allclose(
        model.temperature(z),
        [300.0 + expected],
    )
    assert np.allclose(
        model.steady_state(np.array([P])),
        [P / G],
    )


def test_thermal_rejects_nonpassive_matrices():
    try:
        StableThermalModel(
            np.array([[1.0]]),
            np.array([[-0.1]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "negative thermal conductance must be rejected"
        )
