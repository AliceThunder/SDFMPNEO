import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    RigidPose,
    Scene,
    StructuredPortPrediction,
    SuperellipseSpiral,
    encode_scene_invariant,
)
from sdfmpneo_vnext.neural import ResidualNormalizer
from sdfmpneo_vnext.spatial_neural import (
    SpatialLossArtifact,
    SpatialLossShapeNet,
)


class _PortArtifact:
    def __init__(self, scene, frequency):
        encoded = encode_scene_invariant(scene, frequency)
        self.normalizer = ResidualNormalizer(
            np.zeros(encoded.node_features.shape[-1]),
            np.ones(encoded.node_features.shape[-1]),
            np.zeros(encoded.pair_features.shape[-1]),
            np.ones(encoded.pair_features.shape[-1]),
            1.0,
            1.0,
        )

    def fingerprint(self):
        return "test-port-artifact"

    def predict_structured(self, scene, frequency_hz):
        n = len(scene.coils)
        resistance = np.eye(n) * 0.2
        if n > 1:
            resistance += 0.03 * (np.ones((n, n)) - np.eye(n))
        reactance = np.eye(n) * 0.8
        impedance = resistance + 1j * reactance
        channels = np.zeros((n, n, n), dtype=complex)
        for coil in range(n):
            raw = np.eye(n) * (0.03 + 0.01 * coil)
            raw[coil, coil] += 0.05
            channels[coil] = raw
        total = np.sum(channels, axis=0)
        # exact common congruence into resistance
        w, v = np.linalg.eigh(total)
        inv = v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ v.T
        wr, vr = np.linalg.eigh(resistance)
        root = vr @ np.diag(np.sqrt(np.maximum(wr, 0.0))) @ vr.T
        C = root @ inv
        channels = np.stack([C @ d @ C.T for d in channels])
        return StructuredPortPrediction(impedance, channels)


def _scene(swapped=False):
    copper = ConductorMaterial(5.8e7)
    a = CoilObject(
        SuperellipseSpiral(
            0.030, 0.026, 0.8, 0.0015, 0.0015,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
        ),
        copper,
        "a",
    )
    b = CoilObject(
        SuperellipseSpiral(
            0.024, 0.020, 0.7, 0.0012, 0.0012,
            exponent=4.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.35,
                translation=(0.006, 0.0, 0.02),
            ),
        ),
        copper,
        "b",
    )
    return Scene((b, a) if swapped else (a, b), HomogeneousMedium())


def test_spatial_neural_field_is_psd_and_integrates_to_port_channels():
    scene = _scene()
    port = _PortArtifact(scene, 60_000.0)
    torch.manual_seed(13)
    model = SpatialLossShapeNet(
        hidden_dim=24,
        factor_rank=2,
        depth=1,
    )
    artifact = SpatialLossArtifact(
        model,
        port,
        normalization_segments=8,
        radial_order=3,
        angular_order=12,
    )
    integrated = artifact.integrated_channels(scene, 60_000.0)
    expected = port.predict_structured(scene, 60_000.0).dissipation_channels
    assert np.allclose(integrated, expected, rtol=2e-5, atol=2e-6)

    matrix = artifact.local_dissipation_matrix(
        scene,
        60_000.0,
        0,
        0.4,
        (0.0, 0.0),
    )
    assert np.allclose(matrix, matrix.conj().T, atol=2e-6)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -2e-6
    assert (
        artifact.local_joule_density(
            scene,
            60_000.0,
            0,
            0.4,
            (0.0, 0.0),
            np.array([1.0 + 0.2j, -0.3 + 0.1j]),
        )
        >= 0.0
    )


def test_spatial_field_respects_port_and_object_permutation():
    scene = _scene(False)
    swapped = _scene(True)
    port_a = _PortArtifact(scene, 60_000.0)
    port_b = _PortArtifact(swapped, 60_000.0)
    torch.manual_seed(5)
    model_a = SpatialLossShapeNet(hidden_dim=20, factor_rank=2, depth=1)
    model_b = SpatialLossShapeNet(hidden_dim=20, factor_rank=2, depth=1)
    model_b.load_state_dict(model_a.state_dict())
    a = SpatialLossArtifact(
        model_a, port_a,
        normalization_segments=6,
        radial_order=3,
        angular_order=10,
    )
    b = SpatialLossArtifact(
        model_b, port_b,
        normalization_segments=6,
        radial_order=3,
        angular_order=10,
    )
    H = a.local_dissipation_matrix(scene, 60_000.0, 0, 0.35, (0.0, 0.0))
    Hs = b.local_dissipation_matrix(swapped, 60_000.0, 1, 0.35, (0.0, 0.0))
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(Hs, P @ H @ P.T, rtol=5e-5, atol=5e-6)



def test_spatial_neural_prepared_field_matches_unified_runtime_contract():
    scene = _scene()
    port = _PortArtifact(scene, 60_000.0)
    torch.manual_seed(17)
    artifact = SpatialLossArtifact(
        SpatialLossShapeNet(
            hidden_dim=20,
            factor_rank=2,
            depth=1,
        ),
        port,
        normalization_segments=6,
        radial_order=3,
        angular_order=10,
    )
    prepared = artifact.prepare(
        scene,
        60_000.0,
    )
    assert prepared.port_prediction.impedance.shape == (2, 2)
    assert prepared.normalization_closure_error < 5e-5
    matrices = prepared.local_dissipation_matrices(
        np.array([0, 1]),
        np.array([0.25, 0.75]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    )
    assert matrices.shape == (2, 2, 2)
    assert np.allclose(
        matrices,
        matrices.conj().transpose(0, 2, 1),
        atol=3e-6,
    )
