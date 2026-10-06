import numpy as np

from sdfmpneo_vnext.electric_tensor import PreparedTensorElectricTransmission


def test_tensor_mfs_projects_raw_collocation_response_to_reciprocal_operator():
    transmission = PreparedTensorElectricTransmission.__new__(
        PreparedTensorElectricTransmission
    )
    transmission.source_positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        dtype=float,
    )
    transmission.source_radii = np.asarray([1.0e-4, 2.0e-4], dtype=float)
    transmission.source_region = np.asarray([-1, -1], dtype=int)
    transmission.regions = (object(),)
    transmission._coefficients = np.asarray([[0.0, 1.0]], dtype=complex)
    transmission.maximum_raw_reciprocity_defect = 1.0e-6
    transmission.raw_reciprocity_defect = 0.0
    transmission.raw_reciprocity_target_exceeded = False

    transmission._active_region = lambda points: np.full(
        len(np.atleast_2d(points)),
        -1,
        dtype=int,
    )
    transmission._coefficient = lambda region_index: np.eye(3, dtype=complex)

    def asymmetric_collocation_basis(
        points,
        *,
        region_index,
        normals=None,
        field=False,
    ):
        assert not field
        assert normals is None
        assert region_index == -1
        return np.asarray([[1.0], [0.0]], dtype=complex), None

    transmission._region_basis = asymmetric_collocation_basis

    projected = transmission.potential_matrix()

    assert transmission.raw_reciprocity_defect > 1.0e-6
    assert transmission.raw_reciprocity_target_exceeded
    assert np.allclose(projected, projected.T, rtol=0.0, atol=0.0)
