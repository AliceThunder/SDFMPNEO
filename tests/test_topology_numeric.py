import numpy as np
import pytest
import scipy.sparse as sp

from sdfmpneo.em.topology_numeric import (
    certified_integer_topology_matrix,
    fail_closed_lu_factor,
)


def test_exact_complex_dtype_integer_topology_converts_without_warning():
    matrix = sp.csr_matrix(np.array([[1.0 + 0.0j, -1.0 + 0.0j], [0.0, 2.0 + 0.0j]]))
    converted = certified_integer_topology_matrix(matrix, name="test topology")
    assert converted.dtype.kind in {"i", "u"}
    assert np.array_equal(converted.toarray(), np.array([[1, -1], [0, 2]]))


def test_topology_conversion_rejects_nonzero_imaginary_component():
    matrix = sp.csr_matrix(np.array([[1.0 + 1e-30j]]))
    with pytest.raises(ValueError, match="nonzero imaginary"):
        certified_integer_topology_matrix(matrix)


def test_topology_conversion_rejects_noninteger_real_entry():
    matrix = sp.csr_matrix(np.array([[0.5 + 0.0j]]))
    with pytest.raises(ValueError, match="non-integer"):
        certified_integer_topology_matrix(matrix)


def test_fail_closed_lu_factor_translates_exact_singularity():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    with pytest.raises(ValueError, match="declared singular"):
        fail_closed_lu_factor(singular, singular_message="declared singular")
