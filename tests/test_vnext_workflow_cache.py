from copy import deepcopy
from pathlib import Path

from sdfmpneo_vnext.workflow_cache import TensorTeacherCache, teacher_cache_key


def _config(tmp_path):
    return {
        "DATA": {
            "count": 4,
            "seed": 37,
            "workers": 2,
            "native_threads_per_worker": 1,
            "generation_chunk_size": 2,
        },
        "SAMPLER": {
            "base": {
                "conductor": {},
                "dc_probability": 0.0,
            },
            "tensor_package_probability": 1.0,
            "tensor_background_probability": 0.0,
        },
        "TEACHER": {
            "segments_per_turn": 8,
            "min_segments": 8,
        },
        "TRUTH": {
            "baseline_segments": 16,
            "surface_vertical_order": 4,
            "surface_azimuthal_order": 8,
            "include_spatial": False,
        },
    }


def test_cache_identity_ignores_count_workers_and_training_runtime(tmp_path):
    config = _config(tmp_path)
    original = teacher_cache_key(config)

    changed = deepcopy(config)
    changed["DATA"]["count"] = 128
    changed["DATA"]["workers"] = 16
    changed["DATA"]["native_threads_per_worker"] = 4
    changed["DATA"]["generation_chunk_size"] = 64
    changed["RUNTIME"] = {"device": "cuda"}
    changed["PORT_TRAINING"] = {"epochs": 999, "batch_size": 64}
    assert teacher_cache_key(changed) == original


def test_cache_identity_changes_with_parameter_space_or_teacher_truth(tmp_path):
    config = _config(tmp_path)
    original = teacher_cache_key(config)

    changed_space = deepcopy(config)
    changed_space["SAMPLER"]["tensor_background_probability"] = 0.5
    assert teacher_cache_key(changed_space) != original

    changed_teacher = deepcopy(config)
    changed_teacher["TEACHER"]["segments_per_turn"] = 12
    assert teacher_cache_key(changed_teacher) != original

    changed_truth = deepcopy(config)
    changed_truth["TRUTH"]["baseline_segments"] = 24
    assert teacher_cache_key(changed_truth) != original


def test_cache_shards_are_reusable_and_checksum_guarded(tmp_path):
    config = _config(tmp_path)
    cache = TensorTeacherCache(tmp_path / "cache", config, verify_checksums=True)
    cache.store(0, {"value": 11})
    cache.store(2, {"value": 22})

    assert cache.load(0) == {"value": 11}
    assert cache.existing_indices() == (0, 2)
    assert cache.missing_indices(4) == (1, 3)

    sample_path = cache.path / cache._manifest["samples"]["0"]["path"]
    sample_path.write_bytes(b"corrupt")
    assert 0 not in cache.existing_indices()
    assert 0 in cache.missing_indices(4)
