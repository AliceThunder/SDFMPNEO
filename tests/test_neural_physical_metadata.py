import json

import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.physical_metadata import compact_summary


def test_compact_summary_encodes_small_complex_arrays_as_json():
    value = np.array([[1.0 + 2.0j, -3.0 + 0.5j]])
    summary = compact_summary(value)
    assert summary["kind"] == "complex_ndarray"
    assert summary["shape"] == [1, 2]
    assert summary["real"] == [[1.0, -3.0]]
    assert summary["imag"] == [[2.0, 0.5]]
    encoded = json.dumps(summary, allow_nan=False)
    assert "complex_ndarray" in encoded


def test_compact_summary_encodes_complex_scalar_as_json():
    summary = compact_summary(np.complex128(2.5 - 0.75j))
    assert summary == {"kind": "complex", "real": 2.5, "imag": -0.75}
    json.dumps(summary, allow_nan=False)


def test_compact_summary_rejects_nonfinite_float_provenance():
    with pytest.raises(ValueError, match="non-finite"):
        compact_summary(np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="non-finite"):
        compact_summary(float("nan"))
