import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
    analytic_port_baseline,
    encode_scene_invariant,
)
from sdfmpneo_vnext.bundle import (
    load_bundle,
    publish_bundle,
)
from sdfmpneo_vnext.neural import (
    NeuralResidualArtifact,
    PhysicsFactoredResidualNet,
    ResidualNormalizer,
)
from sdfmpneo_vnext.uncertainty import FastErrorCalibrator


def _scene():
    copper = ConductorMaterial(5.8e7)
    coil = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.022,
            0.8,
            0.001,
            0.001,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
    )


def _artifact():
    scene = _scene()
    frequency = 50_000.0
    baseline = analytic_port_baseline(
        scene,
        frequency,
        segments_per_coil=32,
    )
    target = baseline.impedance
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


def test_bundle_round_trip_and_calibrated_fast(tmp_path):
    port = _artifact()
    calibrator = FastErrorCalibrator(
        quantile=0.95,
        scale=1.5,
        indicator_floor=1e-3,
        validation_samples=10,
        ensemble_size=1,
        baseline_weight=0.5,
        ensemble_weight=0.0,
    )
    root = tmp_path / "bundle"
    manifest = publish_bundle(
        root,
        port,
        calibrator=calibrator,
        metadata={
            "test": True
        },
    )
    assert set(
        manifest["files"]
    ) == {
        "port",
        "calibrator",
    }

    loaded = load_bundle(
        root
    )
    scene = _scene()
    direct = port.predict(
        scene,
        50_000.0,
    )
    restored = (
        loaded.system.fast_ports(
            scene,
            50_000.0,
        ).impedance
    )
    assert np.allclose(
        direct,
        restored,
        rtol=0,
        atol=1e-12,
    )
    calibrated = loaded.calibrated_fast(
        scene,
        50_000.0,
        baseline_segments=32,
    )
    assert (
        calibrated.relative_error_bound
        >= 0.0
    )


def test_bundle_detects_artifact_tampering(tmp_path):
    root = tmp_path / "bundle"
    publish_bundle(
        root,
        _artifact(),
    )
    port_path = (
        root
        / "port.pt"
    )
    with port_path.open(
        "ab"
    ) as handle:
        handle.write(
            b"tamper"
        )
    with pytest.raises(
        ValueError,
        match="checksum mismatch",
    ):
        load_bundle(
            root
        )



def test_bundle_manifest_port_fingerprint_is_verified(tmp_path):
    root = tmp_path / "bundle"
    port = _artifact()
    manifest = publish_bundle(
        root,
        port,
    )
    assert (
        manifest["port_fingerprint"]
        == port.fingerprint()
    )

    manifest_path = root / "manifest.json"
    payload = __import__("json").loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )
    payload["port_fingerprint"] = "0" * 64
    manifest_path.write_text(
        __import__("json").dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="port fingerprint mismatch",
    ):
        load_bundle(
            root
        )



def test_development_bundle_is_rejected_when_release_is_required(tmp_path):
    root = tmp_path / "development"
    manifest = publish_bundle(
        root,
        _artifact(),
    )
    assert (
        manifest["release_status"]
        == "development"
    )
    load_bundle(
        root,
        require_release=False,
    )
    with pytest.raises(
        ValueError,
        match="development-only",
    ):
        load_bundle(
            root,
            require_release=True,
        )


def test_release_bundle_requires_all_locked_audits_to_pass(tmp_path):
    good_gate = {
        "port": {
            "passed": True,
        },
        "spatial": {
            "passed": True,
        },
        "certified": {
            "passed": True,
        },
    }
    root = tmp_path / "released"
    manifest = publish_bundle(
        root,
        _artifact(),
        release_gate=good_gate,
    )
    assert (
        manifest["release_status"]
        == "released"
    )
    loaded = load_bundle(
        root,
        require_release=True,
    )
    assert (
        loaded.manifest["release_gate"]
        == good_gate
    )

    bad_gate = {
        **good_gate,
        "certified": {
            "passed": False,
        },
    }
    with pytest.raises(
        ValueError,
        match="failed audits",
    ):
        publish_bundle(
            tmp_path / "bad",
            _artifact(),
            release_gate=bad_gate,
        )
