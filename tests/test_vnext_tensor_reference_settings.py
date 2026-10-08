from types import SimpleNamespace

import sdfmpneo_vnext.tensor_reference as tensor_reference


def test_tensor_reference_spatial_defaults_follow_artifact_energy_settings(monkeypatch):
    artifact = tensor_reference.TensorElectricReferenceArtifact(
        energy_volume_axial_order=10,
        energy_volume_radial_order=7,
        energy_volume_azimuthal_order=28,
        energy_background_radial_order=14,
        energy_background_angular_order=56,
        maximum_raw_energy_closure_error=0.17,
        normalized_energy_closure_tolerance=3e-7,
    )
    teacher = SimpleNamespace(solve=lambda: object())
    captured = {}
    sentinel = object()

    monkeypatch.setattr(artifact, "supports_scene", lambda scene: True)
    monkeypatch.setattr(
        artifact,
        "_raw_teacher",
        lambda scene, frequency_hz: teacher,
    )

    def fake_prepare(resolved_teacher, result, **kwargs):
        assert resolved_teacher is teacher
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        tensor_reference,
        "prepare_tensor_reference_loss_field",
        fake_prepare,
    )

    assert artifact.prepare_spatial(object(), 123_000.0) is sentinel
    assert captured == {
        "volume_axial_order": 10,
        "volume_radial_order": 7,
        "volume_azimuthal_order": 28,
        "background_radial_order": 14,
        "background_angular_order": 56,
        "maximum_raw_closure_error": 0.17,
        "normalized_closure_tolerance": 3e-7,
    }
