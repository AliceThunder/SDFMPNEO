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
    analytic_port_baseline,
    encode_scene_invariant,
)
from sdfmpneo_vnext.neural import (
    NeuralResidualArtifact,
    PhysicsFactoredResidualNet,
    ResidualNormalizer,
    TeacherSample,
)


def _scene(swapped=False):
    copper = ConductorMaterial(5.8e7)
    a = CoilObject(
        SuperellipseSpiral(
            0.030,
            0.026,
            0.9,
            0.0015,
            0.0015,
            exponent=3.0,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
        "a",
    )
    b = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.020,
            0.75,
            0.0012,
            0.0012,
            exponent=4.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.4,
                translation=(0.006, 0.0, 0.02),
            ),
        ),
        copper,
        "b",
    )
    coils = (b, a) if swapped else (a, b)
    return Scene(coils, HomogeneousMedium())


def _forward(model, scene):
    encoded = encode_scene_invariant(
        scene,
        70_000.0,
    )
    baseline = analytic_port_baseline(
        scene,
        70_000.0,
        segments_per_coil=48,
    )
    node = torch.as_tensor(
        encoded.node_features,
        dtype=torch.float32,
    )
    pair = torch.as_tensor(
        encoded.pair_features,
        dtype=torch.float32,
    )
    R, X = model(
        node,
        pair,
        torch.as_tensor(
            baseline.resistance,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            2.0
            * np.pi
            * 70_000.0
            * baseline.inductance,
            dtype=torch.float32,
        ),
        resistance_scale=0.01,
        reactance_scale=0.1,
    )
    return (
        R.detach().cpu().numpy(),
        X.detach().cpu().numpy(),
    )


def test_random_network_is_reciprocal_passive_and_permutation_equivariant():
    torch.manual_seed(4)
    model = PhysicsFactoredResidualNet(
        hidden_dim=24,
        factor_rank=3,
        depth=1,
    )
    R, X = _forward(
        model,
        _scene(False),
    )
    assert np.allclose(
        R,
        R.T,
        atol=1e-6,
    )
    assert np.allclose(
        X,
        X.T,
        atol=1e-6,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(R)
        )
        >= -1e-7
    )

    Rs, Xs = _forward(
        model,
        _scene(True),
    )
    P = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    assert np.allclose(
        Rs,
        P @ R @ P.T,
        rtol=2e-5,
        atol=2e-6,
    )
    assert np.allclose(
        Xs,
        P @ X @ P.T,
        rtol=2e-5,
        atol=2e-6,
    )


def test_zero_residual_artifact_round_trip_matches_physics_baseline(tmp_path):
    scene = _scene(False)
    frequency = 70_000.0
    encoded = encode_scene_invariant(
        scene,
        frequency,
    )
    baseline = analytic_port_baseline(
        scene,
        frequency,
        segments_per_coil=48,
    )
    target = (
        baseline.resistance
        + 1j
        * (
            2.0
            * np.pi
            * frequency
            * baseline.inductance
        )
    )
    sample = TeacherSample(
        scene,
        frequency,
        encoded,
        baseline.resistance,
        target.imag,
        target,
        48,
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
    artifact = NeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=48,
    )
    predicted = artifact.predict(
        scene,
        frequency,
    )
    assert np.allclose(
        predicted,
        target,
        rtol=2e-6,
        atol=2e-7,
    )

    path = tmp_path / "vnext.pt"
    artifact.save(path)
    loaded = NeuralResidualArtifact.load(
        path
    )
    assert np.allclose(
        loaded.predict(
            scene,
            frequency,
        ),
        predicted,
        rtol=0,
        atol=1e-12,
    )
