import numpy as np

from sdfmpneo.electrothermal_pipeline import ElectroThermalSurrogate


class _Evolution:
    n_modes = 2

    def evaluate(self, state, time, operating=None):
        return np.asarray(state) + time


class _Vector:
    n_modes = 2

    def evaluate(self, state, operating=None):
        class Result:
            vector_field = -np.asarray(state)
        return Result()


def test_production_pipeline_delegates_continuous_evolution():
    model = ElectroThermalSurrogate(_Vector(), _Evolution())
    result = model.state(np.array([1.0, 2.0]), 0.5)
    assert np.allclose(result.coordinates, [1.5, 2.5])
    assert np.allclose(model.residual(np.array([1.0, 2.0])), [-1.0, -2.0])


def test_pipeline_rejects_dimension_mismatch():
    class BadEvolution:
        n_modes = 3

    try:
        ElectroThermalSurrogate(_Vector(), BadEvolution())
    except ValueError:
        return
    raise AssertionError("dimension mismatch must fail")
