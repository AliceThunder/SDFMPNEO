import json
import numpy as np
import pytest

from sdfmpneo_vnext import (
    ImmutableTeacherDataset,
    migrate_dataset_v3_to_v4,
)


def _legacy_dataset(root):
    samples = root / "samples"
    samples.mkdir(
        parents=True,
        exist_ok=True,
    )
    old_id = "1" * 64
    np.savez_compressed(
        samples / f"{old_id}.npz",
        placeholder=np.asarray(
            [1.0]
        ),
    )
    manifest = {
        "schema": 3,
        "split_seed": 17,
        "split_fractions": [
            0.70,
            0.15,
            0.10,
            0.05,
        ],
        "records": [
            {
                "sample_id": old_id,
                "split": "train",
                "source": "initial",
                "frequency_hz": 85_000.0,
                "baseline_segments": 64,
                "file": f"samples/{old_id}.npz",
                "scene": {
                    "coils": [],
                    "medium": {},
                },
                "teacher_config": {
                    "segments_per_turn": 12,
                },
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )
    return old_id


def test_legacy_dataset_requires_explicit_migration(tmp_path):
    root = tmp_path / "dataset"
    _legacy_dataset(
        root
    )
    with pytest.raises(
        ValueError,
        match="migrate_dataset_v3_to_v4",
    ):
        ImmutableTeacherDataset(
            root
        )


def test_schema3_migration_preserves_sample_bytes_and_marks_mqs_backend(tmp_path):
    root = tmp_path / "dataset"
    old_id = _legacy_dataset(
        root
    )
    old_path = (
        root
        / "samples"
        / f"{old_id}.npz"
    )
    original = old_path.read_bytes()

    report = migrate_dataset_v3_to_v4(
        root
    )
    assert report["migrated"]
    assert (
        root
        / "manifest.schema3.json"
    ).exists()

    dataset = ImmutableTeacherDataset(
        root
    )
    assert len(
        dataset.records
    ) == 1
    record = dataset.records[
        0
    ]
    assert (
        record.reference_backend
        == "mqs"
    )
    assert (
        record.sample_id
        != old_id
    )
    new_path = (
        root
        / record.file
    )
    assert new_path.exists()
    assert not old_path.exists()
    assert (
        new_path.read_bytes()
        == original
    )

    second = migrate_dataset_v3_to_v4(
        root
    )
    assert not second["migrated"]
