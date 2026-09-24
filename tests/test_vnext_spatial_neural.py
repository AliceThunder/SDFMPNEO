import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
    analytic_port_baseline,
    encode_scene_invariant,
)
from sdfmpneo_vnext.neural import (
    NeuralResidualArtifact,
    PhysicsFactoredResidualNet,
    ResidualNormalizer,
)
from sdfmpneo_vnext.spatial_neural import (
    NeuralSpatialLossArtifact,
    SpatialLossShapeNet,
)


def _scene():
    copper = ConductorMaterial(5.8e7)
    first = CoilObject(
        SuperellipseSpiral(
            0.028,
            0.024,
            0.8,
            0.0012,
            0.0012,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
        ),
        copper,
        "a",
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.022,
            0.019,
            0.7,
            0.0010,
            0.0010,
            exponent=4.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.004, 0.0, 0.018]),
            ),
        ),
        copper,
        "b",
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def _port_artifact():
    scene = _scene()
    frequency = 60_000.0
    baseline = analytic_port_baseline(
        scene,
        frequency,
        segments_per_coil=32,
    )
    target = (
        baseline.resistance
        + 1j
        * 2.0
        * np.pi
        * frequency
        * baseline.inductance
    )
    sample = TeacherSample(
        scene,
        frequency,
        encode_scene_invariant(
            scene,
            frequency,
        ),
        baseline.resistance,
        target.imag,
        target,
        32,
    )
    normalizer = ResidualNormalizer.fit(
        (sample,)
    )
    model = PhysicsFactoredResidualNet(
        hidden_dim=16,
        factor_rank=2,
        depth=1,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return NeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=32,
    )


def test_spatial_decoder_is_locally_psd_and_closes_port_channels():
    torch.manual_seed(3)
    port = _port_artifact()
    field_model = SpatialLossShapeNet(
        hidden_dim=16,
        pair_dim=15,
        field_hidden_dim=16,
        factor_rank=2,
        depth=1,
    )
    artifact = NeuralSpatialLossArtifact(
        port,
        field_model,
        longitudinal_points=6,
        radial_order=2,
        angular_order=8,
    )
    prepared = artifact.prepare(
        _scene(),
        60_000.0,
    )
    assert (
        prepared.normalization_closure_error
        < 5e-5
    )
    for coil in (0, 1):
        matrix = (
            prepared.local_dissipation_matrix(
                coil,
                0.45,
                (0.0, 0.0),
            )
        )
        assert np.allclose(
            matrix,
            matrix.conj().T,
            atol=2e-6,
        )
        assert (
            np.min(
                np.linalg.eigvalsh(
                    matrix
                )
            )
            >= -2e-6
        )
    outside = (
        prepared.local_dissipation_matrix(
            0,
            0.5,
            (1.0, 1.0),
        )
    )
    assert np.allclose(
        outside,
        0.0,
    )
