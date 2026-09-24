import numpy as np

from sdfmpneo_vnext import (
    ConductorMaterial,
    CoilObject,
    HomogeneousMedium,
    ImmutableTeacherDataset,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
    analytic_port_baseline,
    deterministic_split,
    encode_scene_invariant,
    sample_two_coil_mvp_scene,
    scene_from_dict,
    scene_to_dict,
)


def _scene():
    copper = ConductorMaterial(5.8e7)
    a = CoilObject(
        SuperellipseSpiral(
            0.03,
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
            0.025,
            0.021,
            0.75,
            0.0012,
            0.0012,
            exponent=4.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.005, 0.0, 0.02]),
            ),
        ),
        copper,
        "b",
    )
    return Scene((a, b), HomogeneousMedium())


def _sample():
    scene = _scene()
    frequency = 60_000.0
    encoded = encode_scene_invariant(scene, frequency)
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
    return TeacherSample(
        scene,
        frequency,
        encoded,
        baseline.resistance,
        target.imag,
        target,
        32,
    )


def test_scene_serialization_round_trip():
    scene = _scene()
    restored = scene_from_dict(
        scene_to_dict(scene)
    )
    assert len(restored.coils) == len(scene.coils)
    for original, loaded in zip(
        scene.coils,
        restored.coils,
    ):
        assert original.name == loaded.name
        assert np.allclose(
            original.geometry.pose.rotation,
            loaded.geometry.pose.rotation,
        )
        assert np.allclose(
            original.geometry.pose.translation,
            loaded.geometry.pose.translation,
        )
        assert np.isclose(
            original.material.conductivity,
            loaded.material.conductivity,
        )


def test_dataset_is_content_addressed_and_idempotent(tmp_path):
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset",
        split_seed=23,
    )
    sample = _sample()
    cfg = MQSConfig(
        segments_per_turn=8,
        min_segments=10,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )
    first = dataset.add_sample(
        sample,
        teacher_config=cfg,
        split="validation",
    )
    second = dataset.add_sample(
        sample,
        teacher_config=cfg,
        split="validation",
    )
    assert first.sample_id == second.sample_id
    assert len(dataset.records) == 1
    loaded = dataset.load_sample(
        first.sample_id
    )
    assert np.allclose(
        loaded.target_impedance,
        sample.target_impedance,
    )
    assert loaded.baseline_segments == 32


def test_active_learning_cannot_enter_release_split(tmp_path):
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset"
    )
    sample = _sample()
    try:
        dataset.add_sample(
            sample,
            source="active",
            split="release",
        )
    except ValueError as exc:
        assert "train" in str(exc)
    else:
        raise AssertionError(
            "active sample must not enter release split"
        )


def test_deterministic_split_and_sampler_are_reproducible():
    assert (
        deterministic_split(
            "abc",
            seed=17,
        )
        == deterministic_split(
            "abc",
            seed=17,
        )
    )
    rng_a = np.random.default_rng(31)
    rng_b = np.random.default_rng(31)
    scene_a, f_a = sample_two_coil_mvp_scene(
        rng_a
    )
    scene_b, f_b = sample_two_coil_mvp_scene(
        rng_b
    )
    assert np.isclose(f_a, f_b)
    assert scene_to_dict(scene_a) == scene_to_dict(scene_b)
