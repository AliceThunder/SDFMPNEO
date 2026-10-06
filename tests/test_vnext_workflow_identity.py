from copy import deepcopy

from sdfmpneo_vnext.workflow import _training_dataset_key
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
