import numpy as np

from sdfmpneo_vnext import (
    ConductorMaterial,
    CoilObject,
    DenseMixedConductorTeacher,
    HomogeneousMedium,
    ImmutableTeacherDataset,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SpatialLossSamples,
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
        reference_backend="mqs",
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



def test_dataset_round_trips_spatial_loss_samples(tmp_path):
    sample = _sample()
    n_ports = sample.target_impedance.shape[0]
    spatial = SpatialLossSamples(
        np.array([0, 1], dtype=int),
        np.array([0.25, 0.75], dtype=float),
        np.array(
            [
                [0.0, 0.0],
                [1e-4, -2e-4],
            ],
            dtype=float,
        ),
        np.array([1e-9, 2e-9], dtype=float),
        np.stack(
            [
                np.eye(n_ports) * 2.0,
                np.eye(n_ports) * 3.0,
            ]
        ).astype(complex),
    )
    sample = TeacherSample(
        sample.scene,
        sample.frequency_hz,
        sample.encoded,
        sample.baseline_resistance,
        sample.baseline_reactance,
        sample.target_impedance,
        sample.baseline_segments,
        sample.target_dissipation_channels,
        spatial,
    )
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset"
    )
    record = dataset.add_sample(
        sample,
        split="train",
    )
    loaded = dataset.load_sample(
        record.sample_id
    )
    assert loaded.spatial_loss is not None
    assert np.array_equal(
        loaded.spatial_loss.coil_index,
        spatial.coil_index,
    )
    assert np.allclose(
        loaded.spatial_loss.arc_fraction,
        spatial.arc_fraction,
    )
    assert np.allclose(
        loaded.spatial_loss.xy,
        spatial.xy,
    )
    assert np.allclose(
        loaded.spatial_loss.weights,
        spatial.weights,
    )
    assert np.allclose(
        loaded.spatial_loss.dissipation_matrix,
        spatial.dissipation_matrix,
    )



def test_minimal_scene_json_uses_physical_defaults_and_pitch_shorthand():
    compact = {
        "coils": [
            {
                "geometry": {
                    "outer_a": 0.03,
                    "outer_b": 0.025,
                    "turns": 0.8,
                    "pitch": 0.0012,
                    "conductor_width": 0.001,
                    "conductor_thickness": 0.0008,
                },
                "material": {
                    "conductivity": 5.8e7,
                },
            }
        ]
    }
    scene = scene_from_dict(
        compact
    )
    assert len(scene.coils) == 1
    coil = scene.coils[0]
    assert coil.name == "coil_0"
    assert np.isclose(
        coil.geometry.pitch_a,
        0.0012,
    )
    assert np.isclose(
        coil.geometry.pitch_b,
        0.0012,
    )
    assert np.isclose(
        coil.geometry.exponent,
        2.0,
    )
    assert np.isclose(
        coil.geometry.cross_section_exponent,
        2.0,
    )
    assert np.allclose(
        coil.geometry.pose.rotation,
        np.eye(3),
    )
    assert np.allclose(
        coil.geometry.pose.translation,
        np.zeros(3),
    )
    assert np.isclose(
        coil.material.relative_permeability,
        1.0,
    )
    assert np.isclose(
        coil.material.resistance_temperature_coefficient,
        0.0,
    )
    assert np.isclose(
        coil.material.reference_temperature,
        293.15,
    )
    assert np.isclose(
        scene.medium.relative_permittivity,
        1.0,
    )
    assert np.isclose(
        scene.medium.relative_permeability,
        1.0,
    )
    assert np.isclose(
        scene.medium.conductivity,
        0.0,
    )



def test_mixed_reference_backend_generates_mixed_truth_and_records_backend(tmp_path):
    scene = _scene()
    frequency = 12_000.0
    config = MQSConfig(
        segments_per_turn=6,
        min_segments=6,
        section_degree=0,
        radial_order=2,
        angular_order=8,
        line_order=2,
    )
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset"
    )
    record = dataset.generate_and_add(
        scene,
        frequency,
        teacher_config=config,
        baseline_segments=24,
        reference_backend="mixed",
        split="train",
    )
    assert record.reference_backend == "mixed"
    loaded = dataset.load_sample(
        record.sample_id
    )
    expected = DenseMixedConductorTeacher(
        scene,
        frequency,
        config,
    ).solve()
    assert np.allclose(
        loaded.target_impedance,
        expected.impedance,
        rtol=2e-11,
        atol=2e-12,
    )
    assert loaded.target_dissipation_channels is not None
    assert np.allclose(
        loaded.target_dissipation_channels,
        expected.coil_dissipation_matrices(),
        rtol=2e-11,
        atol=2e-12,
    )



def test_dataset_reports_reference_backends_per_split(tmp_path):
    sample = _sample()
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset"
    )
    dataset.add_sample(
        sample,
        reference_backend="mqs",
        split="validation",
    )
    dataset.add_sample(
        TeacherSample(
            sample.scene,
            sample.frequency_hz + 1.0,
            encode_scene_invariant(
                sample.scene,
                sample.frequency_hz + 1.0,
            ),
            sample.baseline_resistance,
            sample.baseline_reactance,
            sample.target_impedance,
            sample.baseline_segments,
            sample.target_dissipation_channels,
            sample.spatial_loss,
            "mixed",
        ),
        reference_backend="mixed",
        split="train",
    )
    assert dataset.reference_backends(
        "train"
    ) == ("mixed",)
    assert dataset.reference_backends(
        "validation"
    ) == ("mqs",)
    assert dataset.reference_backends() == (
        "mixed",
        "mqs",
    )



def test_default_generated_dataset_truth_is_canonical_mixed(tmp_path):
    scene = _scene()
    frequency = 11_000.0
    config = MQSConfig(
        segments_per_turn=6,
        min_segments=6,
        section_degree=0,
        radial_order=2,
        angular_order=8,
        line_order=2,
    )
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset_default_mixed"
    )
    record = dataset.generate_and_add(
        scene,
        frequency,
        teacher_config=config,
        baseline_segments=20,
        split="train",
    )
    assert record.reference_backend == "mixed"
    loaded = dataset.load_sample(
        record.sample_id
    )
    assert loaded.reference_backend == "mixed"
    expected = DenseMixedConductorTeacher(
        scene,
        frequency,
        config,
    ).solve()
    assert np.allclose(
        loaded.target_impedance,
        expected.impedance,
        rtol=2e-12,
        atol=2e-13,
    )
    assert np.allclose(
        loaded.target_dissipation_channels,
        expected.coil_dissipation_matrices(),
        rtol=2e-11,
        atol=2e-13,
    )


def test_dataset_rejects_reference_backend_provenance_mismatch(tmp_path):
    sample = _sample()
    dataset = ImmutableTeacherDataset.create(
        tmp_path / "dataset_provenance"
    )
    try:
        dataset.add_sample(
            sample,
            reference_backend="mixed",
            split="train",
        )
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError(
            "dataset must reject a backend label that disagrees with sample provenance"
        )
