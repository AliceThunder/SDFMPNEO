import numpy as np
import pytest

from sdfmpneo_vnext import (
    HomogeneousMedium,
    IsotropicMaterial,
    RigidPose,
    SmoothHeterogeneousMedium,
    SmoothMaterialAnchor,
    TensorElectricMaterial,
)


def test_smooth_heterogeneous_medium_convexly_blends_electric_properties():
    base = HomogeneousMedium(
        relative_permittivity=2.0,
        conductivity=1.0e-4,
    )
    target = IsotropicMaterial(
        relative_permittivity=8.0,
        conductivity=5.0e-4,
    )
    medium = SmoothHeterogeneousMedium(
        base,
        (
            SmoothMaterialAnchor(
                target,
                np.asarray([0.01, 0.02, 0.03]),
                strength=3.0,
            ),
        ),
    )

    epsilon, sigma, mu = medium.material_tensors_at(
        np.asarray([0.0, 0.0, 0.0]),
        80_000.0,
    )
    assert np.allclose(epsilon, 6.5 * np.eye(3))
    assert np.allclose(sigma, 4.0e-4 * np.eye(3))
    assert np.isclose(mu, 1.0)

    epsilon_far, sigma_far, _ = medium.material_tensors_at(
        np.asarray([1.0, 1.0, 1.0]),
        80_000.0,
    )
    assert np.allclose(epsilon_far, 2.0 * np.eye(3), rtol=0.0, atol=1e-12)
    assert np.allclose(sigma_far, 1.0e-4 * np.eye(3), rtol=0.0, atol=1e-12)


def test_tensor_anchor_rotates_with_its_pose_and_remains_passive():
    rotation = RigidPose.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0)
    target = TensorElectricMaterial(
        relative_permittivity_tensor=np.diag([2.0, 4.0, 8.0]),
        conductivity_tensor=np.diag([1.0e-4, 2.0e-4, 4.0e-4]),
    )
    medium = SmoothHeterogeneousMedium(
        HomogeneousMedium(),
        (
            SmoothMaterialAnchor(
                target,
                np.ones(3),
                pose=rotation,
                strength=1.0,
            ),
        ),
    )

    epsilon, sigma, _ = medium.material_tensors_at(np.zeros(3), 100_000.0)
    expected_epsilon_target = (
        rotation.rotation
        @ target.relative_permittivity_tensor
        @ rotation.rotation.T
    )
    expected_sigma_target = (
        rotation.rotation
        @ target.conductivity_tensor
        @ rotation.rotation.T
    )
    assert np.allclose(epsilon, 0.5 * (np.eye(3) + expected_epsilon_target))
    assert np.allclose(sigma, 0.5 * expected_sigma_target)
    assert np.min(np.linalg.eigvalsh(epsilon)) > 0.0
    assert np.min(np.linalg.eigvalsh(sigma)) >= -1e-14


def test_heterogeneous_anchor_common_pose_transform_preserves_local_profile():
    anchor = SmoothMaterialAnchor(
        IsotropicMaterial(relative_permittivity=5.0),
        np.asarray([0.02, 0.01, 0.03]),
        pose=RigidPose.from_axis_angle(
            (1.0, 0.0, 0.0),
            0.3,
            translation=(0.04, -0.01, 0.02),
        ),
    )
    medium = SmoothHeterogeneousMedium(HomogeneousMedium(), (anchor,))
    query = np.asarray([0.05, -0.012, 0.027])
    reference = medium.relative_permittivity_tensor_at(query, 50_000.0)

    common = RigidPose.from_axis_angle(
        (0.0, 1.0, 0.0),
        -0.4,
        translation=(0.2, -0.1, 0.3),
    )
    moved = medium.transformed(common)
    actual = moved.relative_permittivity_tensor_at(
        common.apply(query),
        50_000.0,
    )
    assert np.allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_heterogeneous_medium_rejects_spatial_mu_variation_for_now():
    with pytest.raises(ValueError, match="uniform relative_permeability"):
        SmoothHeterogeneousMedium(
            HomogeneousMedium(relative_permeability=1.0),
            (
                SmoothMaterialAnchor(
                    IsotropicMaterial(relative_permeability=2.0),
                    np.ones(3),
                ),
            ),
        )


def test_scalar_electric_interface_fails_closed():
    medium = SmoothHeterogeneousMedium(
        HomogeneousMedium(),
        (
            SmoothMaterialAnchor(
                IsotropicMaterial(relative_permittivity=3.0),
                np.ones(3),
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="no single relative permittivity"):
        medium.relative_permittivity_at(100_000.0)
    with pytest.raises(NotImplementedError, match="no single loss conductivity"):
        medium.loss_conductivity(100_000.0)
