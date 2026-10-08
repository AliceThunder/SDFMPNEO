import numpy as np
import pytest

from sdfmpneo_vnext import HybridSceneSamplerConfig
from sdfmpneo_vnext.hybrid_domain import validate_hybrid_geometry_domain
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)


def _background_only_sampler():
    return TensorHybridSceneSamplerConfig(
        base=HybridSceneSamplerConfig(
            package_count_range=(1, 1),
            nested_package_probability=0.0,
            graded_package_probability=0.0,
            free_inclusion_probability=0.0,
            debye_package_probability=0.0,
            multi_debye_package_probability=0.0,
            dc_probability=0.0,
        ),
        tensor_package_probability=0.0,
        tensor_background_probability=0.0,
        background_only_probability=1.0,
        tensor_lossless_probability=0.0,
    )


def test_geometry_domain_accepts_declared_zero_package_topology():
    sampler = _background_only_sampler()
    scene, frequency_hz = sample_tensor_hybrid_scene(
        np.random.default_rng(1901),
        sampler,
    )
    assert scene.packages == ()

    domain = sampler.base.geometry_domain_metadata()
    domain["n_packages_range"] = [0, 1]
    validate_hybrid_geometry_domain(scene, frequency_hz, domain)

    rejected = dict(domain)
    rejected["n_packages_range"] = [1, 1]
    with pytest.raises(ValueError, match="package count"):
        validate_hybrid_geometry_domain(scene, frequency_hz, rejected)
