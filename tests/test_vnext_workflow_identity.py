from copy import deepcopy

from sdfmpneo_vnext.workflow import _training_dataset_key, _training_history_summary
from sdfmpneo_vnext.workflow_cache import teacher_cache_key


def _config():
    return {
        "DATA": {
            "count": 64,
            "seed": 37,
            "workers": 4,
            "native_threads_per_worker": 1,
        },
        "SAMPLER": {
            "base": {"conductor": {}, "dc_probability": 0.0},
            "tensor_package_probability": 1.0,
            "tensor_background_probability": 0.0,
        },
        "TEACHER": {"segments_per_turn": 8, "min_segments": 8},
        "TRUTH": {
            "baseline_segments": 16,
            "surface_vertical_order": 4,
            "surface_azimuthal_order": 8,
            "include_spatial": False,
            "energy_volume_axial_order": 8,
            "energy_volume_radial_order": 6,
            "energy_volume_azimuthal_order": 24,
            "energy_background_radial_order": 12,
            "energy_background_angular_order": 48,
            "maximum_spatial_quadrature_refinements": 0,
        },
        "PORT_TRAINING": {
            "seed": 17,
            "validation_fraction": 0.15,
            "channel_loss_weight": 1.0,
            "device": "cpu",
        },
        "SPATIAL_TRAINING": {
            "seed": 47,
            "validation_fraction": 0.15,
            "end_to_end_validation": True,
            "validation_interval": 1,
            "device": "cpu",
        },
    }


def test_count_reuses_teacher_cache_but_changes_training_dataset_identity():
    config = _config()
    cache_key = teacher_cache_key(config)
    training_key = _training_dataset_key(cache_key, 64, config)

    expanded = deepcopy(config)
    expanded["DATA"]["count"] = 128
    expanded_cache_key = teacher_cache_key(expanded)
    expanded_training_key = _training_dataset_key(expanded_cache_key, 128, expanded)

    assert expanded_cache_key == cache_key
    assert expanded_training_key != training_key


def test_device_change_keeps_teacher_and_training_dataset_identity():
    config = _config()
    cache_key = teacher_cache_key(config)
    training_key = _training_dataset_key(cache_key, 64, config)

    moved = deepcopy(config)
    moved["PORT_TRAINING"]["device"] = "cuda"
    moved["SPATIAL_TRAINING"]["device"] = "cuda"
    moved_cache_key = teacher_cache_key(moved)
    moved_training_key = _training_dataset_key(moved_cache_key, 64, moved)

    assert moved_cache_key == cache_key
    assert moved_training_key == training_key


def test_partition_change_keeps_teacher_cache_but_invalidates_model_dataset_key():
    config = _config()
    cache_key = teacher_cache_key(config)
    training_key = _training_dataset_key(cache_key, 64, config)

    changed = deepcopy(config)
    changed["PORT_TRAINING"]["seed"] = 99
    changed_cache_key = teacher_cache_key(changed)
    changed_training_key = _training_dataset_key(changed_cache_key, 64, changed)

    assert changed_cache_key == cache_key
    assert changed_training_key != training_key


def test_energy_truth_quadrature_changes_teacher_cache_identity():
    config = _config()
    changed = deepcopy(config)
    changed["TRUTH"]["energy_volume_axial_order"] = 10

    assert teacher_cache_key(changed) != teacher_cache_key(config)


def test_ignored_legacy_refinement_budget_does_not_change_teacher_cache_identity():
    config = _config()
    changed = deepcopy(config)
    changed["TRUTH"]["maximum_spatial_quadrature_refinements"] = 9

    assert teacher_cache_key(changed) == teacher_cache_key(config)


def test_training_history_summary_reports_selected_and_final_quality():
    history = [
        {
            "epoch": 1,
            "train_loss": 0.8,
            "validation_loss": 0.7,
            "best_validation_loss": 0.7,
            "best_epoch": 1,
            "device": "cpu",
            "dtype": "float64",
        },
        {
            "epoch": 2,
            "train_loss": 0.5,
            "validation_loss": 0.6,
            "best_validation_loss": 0.6,
            "best_epoch": 2,
            "device": "cpu",
            "dtype": "float64",
        },
        {
            "epoch": 3,
            "train_loss": 0.4,
            "validation_loss": 0.65,
            "best_validation_loss": 0.6,
            "best_epoch": 2,
            "device": "cpu",
            "dtype": "float64",
        },
    ]

    summary = _training_history_summary(history, 10)

    assert summary["epochs_completed"] == 3
    assert summary["final_epoch"] == 3
    assert summary["best_epoch"] == 2
    assert summary["stopped_early"] is True
    assert summary["best_validation_loss"] == 0.6
    assert summary["best_epoch_train_loss"] == 0.5
    assert summary["final_validation_loss"] == 0.65
    assert summary["device"] == "cpu"
    assert summary["dtype"] == "float64"
