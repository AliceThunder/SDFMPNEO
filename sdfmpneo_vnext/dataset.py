from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from .em import MQSConfig
from .features import EncodedScene
from .serialization import (
    canonical_json,
    content_hash,
    scene_from_dict,
    scene_to_dict,
)
from .training_data import (
    CANONICAL_REFERENCE_BACKEND,
    SpatialLossSamples,
    TeacherSample,
)


DATASET_SCHEMA = 4
SPLITS = (
    "train",
    "validation",
    "test",
    "release",
)
REFERENCE_BACKENDS = (
    "mqs",
    "mixed",
)


def deterministic_split(
    sample_id: str,
    *,
    seed: int,
    fractions=(
        0.70,
        0.15,
        0.10,
        0.05,
    ),
) -> str:
    fractions = np.asarray(
        fractions,
        dtype=float,
    )
    if (
        fractions.shape != (4,)
        or np.any(
            fractions < 0.0
        )
        or not np.isclose(
            np.sum(
                fractions
            ),
            1.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "fractions must be four nonnegative values summing to one"
        )
    digest = sha256(
        f"{int(seed)}:{sample_id}".encode(
            "utf-8"
        )
    ).digest()
    value = (
        int.from_bytes(
            digest[:8],
            "big",
        )
        / float(
            2**64
        )
    )
    edge = np.cumsum(
        fractions
    )
    index = int(
        np.searchsorted(
            edge,
            value,
            side="right",
        )
    )
    index = min(
        index,
        3,
    )
    return SPLITS[
        index
    ]


def _normalize_reference_backend(
    value,
) -> str:
    backend = str(
        value
    ).lower()
    if backend not in (
        REFERENCE_BACKENDS
    ):
        raise ValueError(
            "reference_backend must be 'mqs' or 'mixed'"
        )
    return backend


def _identity_payload(
    *,
    scene,
    frequency_hz: float,
    baseline_segments: int,
    teacher_config,
    reference_backend: str,
):
    return {
        "schema": DATASET_SCHEMA,
        "scene": scene,
        "frequency_hz": float(
            frequency_hz
        ),
        "baseline_segments": int(
            baseline_segments
        ),
        "teacher_config": dict(
            teacher_config
        ),
        "reference_backend": (
            _normalize_reference_backend(
                reference_backend
            )
        ),
        "output_schema": (
            "mvp_port_impedance_loss_channels_and_spatial_v4"
        ),
    }


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    split: str
    source: str
    frequency_hz: float
    baseline_segments: int
    file: str
    scene: dict
    teacher_config: dict
    reference_backend: str


class ImmutableTeacherDataset:
    """Content-addressed vNext teacher dataset with frozen split semantics."""

    def __init__(
        self,
        root,
    ):
        self.root = Path(
            root
        )
        self.manifest_path = (
            self.root
            / "manifest.json"
        )
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"dataset manifest does not exist: {self.manifest_path}"
            )
        self._manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )
        schema = self._manifest.get(
            "schema"
        )
        if schema == 3:
            raise ValueError(
                "vNext dataset schema 3 requires explicit "
                "migrate_dataset_v3_to_v4(...) before loading"
            )
        if (
            schema
            != DATASET_SCHEMA
        ):
            raise ValueError(
                "unsupported vNext dataset schema"
            )

    @classmethod
    def create(
        cls,
        root,
        *,
        split_seed: int = 17,
        split_fractions=(
            0.70,
            0.15,
            0.10,
            0.05,
        ),
    ):
        root = Path(
            root
        )
        root.mkdir(
            parents=True,
            exist_ok=True,
        )
        manifest = (
            root
            / "manifest.json"
        )
        if manifest.exists():
            raise FileExistsError(
                f"dataset already exists: {manifest}"
            )
        fractions = [
            float(
                x
            )
            for x in split_fractions
        ]
        deterministic_split(
            "0" * 64,
            seed=split_seed,
            fractions=fractions,
        )
        payload = {
            "schema": DATASET_SCHEMA,
            "split_seed": int(
                split_seed
            ),
            "split_fractions": (
                fractions
            ),
            "records": [],
        }
        manifest.write_text(
            canonical_json(
                payload
            ),
            encoding="utf-8",
        )
        return cls(
            root
        )

    @property
    def split_seed(
        self,
    ) -> int:
        return int(
            self._manifest[
                "split_seed"
            ]
        )

    @property
    def split_fractions(
        self,
    ):
        return tuple(
            float(
                value
            )
            for value
            in self._manifest[
                "split_fractions"
            ]
        )

    @property
    def records(
        self,
    ):
        return tuple(
            DatasetRecord(
                **record
            )
            for record
            in self._manifest[
                "records"
            ]
        )

    def counts(
        self,
    ):
        out = {
            split: 0
            for split
            in SPLITS
        }
        for record in (
            self.records
        ):
            out[
                record.split
            ] += 1
        return out

    def reference_backends(
        self,
        split: str | None = None,
    ) -> tuple[str, ...]:
        if (
            split is not None
            and split not in SPLITS
        ):
            raise ValueError(
                f"unknown split: {split}"
            )
        return tuple(
            sorted(
                {
                    record.reference_backend
                    for record
                    in self.records
                    if (
                        split is None
                        or record.split
                        == split
                    )
                }
            )
        )


    def _write_manifest(
        self,
    ):
        temporary = (
            self.manifest_path.with_suffix(
                ".json.tmp"
            )
        )
        temporary.write_text(
            canonical_json(
                self._manifest
            ),
            encoding="utf-8",
        )
        temporary.replace(
            self.manifest_path
        )

    @staticmethod
    def _teacher_dict(
        teacher_config,
    ):
        if teacher_config is None:
            teacher_config = (
                MQSConfig()
            )
        if not isinstance(
            teacher_config,
            MQSConfig,
        ):
            raise TypeError(
                "teacher_config must be MQSConfig"
            )
        return asdict(
            teacher_config
        )

    def _identity(
        self,
        sample: TeacherSample,
        teacher_config,
        reference_backend: str,
    ):
        teacher_dict = (
            self._teacher_dict(
                teacher_config
            )
        )
        identity = (
            _identity_payload(
                scene=scene_to_dict(
                    sample.scene
                ),
                frequency_hz=(
                    sample.frequency_hz
                ),
                baseline_segments=(
                    sample.baseline_segments
                ),
                teacher_config=(
                    teacher_dict
                ),
                reference_backend=(
                    reference_backend
                ),
            )
        )
        return (
            content_hash(
                identity
            ),
            identity,
            teacher_dict,
        )

    def add_sample(
        self,
        sample: TeacherSample,
        *,
        teacher_config: MQSConfig | None = None,
        reference_backend: str | None = None,
        source: str = "initial",
        split: str | None = None,
    ) -> DatasetRecord:
        sample_backend = getattr(
            sample,
            "reference_backend",
            None,
        )
        if reference_backend is None:
            if sample_backend is None:
                raise ValueError(
                    "reference_backend is required for a TeacherSample without provenance"
                )
            reference_backend = sample_backend
        reference_backend = (
            _normalize_reference_backend(
                reference_backend
            )
        )
        if (
            sample_backend is not None
            and _normalize_reference_backend(
                sample_backend
            )
            != reference_backend
        ):
            raise ValueError(
                "reference_backend disagrees with TeacherSample provenance"
            )
        (
            sample_id,
            identity,
            teacher_dict,
        ) = self._identity(
            sample,
            teacher_config,
            reference_backend,
        )
        existing = {
            record.sample_id: (
                record
            )
            for record
            in self.records
        }
        if (
            sample_id
            in existing
        ):
            return existing[
                sample_id
            ]

        if source == "active":
            if (
                split is not None
                and split
                != "train"
            ):
                raise ValueError(
                    "active-learning samples may only enter the train split"
                )
            split = "train"
        elif split is None:
            split = (
                deterministic_split(
                    sample_id,
                    seed=(
                        self.split_seed
                    ),
                    fractions=(
                        self.split_fractions
                    ),
                )
            )
        if split not in SPLITS:
            raise ValueError(
                f"unknown split: {split}"
            )

        relative = (
            Path(
                "samples"
            )
            / f"{sample_id}.npz"
        )
        path = (
            self.root
            / relative
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if path.exists():
            raise FileExistsError(
                f"orphan sample file already exists: {path}"
            )

        n_ports = int(
            sample.target_impedance.shape[
                0
            ]
        )
        spatial = (
            sample.spatial_loss
        )
        if spatial is None:
            spatial_coil = np.empty(
                0,
                dtype=int,
            )
            spatial_arc = np.empty(
                0,
                dtype=float,
            )
            spatial_xy = np.empty(
                (
                    0,
                    2,
                ),
                dtype=float,
            )
            spatial_weights = np.empty(
                0,
                dtype=float,
            )
            spatial_matrix = np.empty(
                (
                    0,
                    n_ports,
                    n_ports,
                ),
                dtype=complex,
            )
        else:
            spatial_coil = (
                spatial.coil_index
            )
            spatial_arc = (
                spatial.arc_fraction
            )
            spatial_xy = (
                spatial.xy
            )
            spatial_weights = (
                spatial.weights
            )
            spatial_matrix = (
                spatial.dissipation_matrix
            )

        np.savez_compressed(
            path,
            node_features=(
                sample.encoded.node_features
            ),
            pair_features=(
                sample.encoded.pair_features
            ),
            length_scale=np.asarray(
                sample.encoded.length_scale,
                dtype=float,
            ),
            baseline_resistance=(
                sample.baseline_resistance
            ),
            baseline_reactance=(
                sample.baseline_reactance
            ),
            target_impedance=(
                sample.target_impedance
            ),
            target_dissipation_channels=(
                np.asarray(
                    [],
                    dtype=complex,
                )
                if (
                    sample.target_dissipation_channels
                    is None
                )
                else (
                    sample.target_dissipation_channels
                )
            ),
            spatial_coil_index=(
                spatial_coil
            ),
            spatial_arc_fraction=(
                spatial_arc
            ),
            spatial_xy=(
                spatial_xy
            ),
            spatial_weights=(
                spatial_weights
            ),
            spatial_dissipation_matrix=(
                spatial_matrix
            ),
        )

        record = DatasetRecord(
            sample_id=(
                sample_id
            ),
            split=split,
            source=str(
                source
            ),
            frequency_hz=float(
                sample.frequency_hz
            ),
            baseline_segments=int(
                sample.baseline_segments
            ),
            file=str(
                relative
            ),
            scene=identity[
                "scene"
            ],
            teacher_config=(
                teacher_dict
            ),
            reference_backend=(
                reference_backend
            ),
        )
        self._manifest[
            "records"
        ].append(
            asdict(
                record
            )
        )
        self._manifest[
            "records"
        ].sort(
            key=lambda item: (
                item[
                    "sample_id"
                ]
            )
        )
        self._write_manifest()
        return record

    def generate_and_add(
        self,
        scene,
        frequency_hz: float,
        *,
        teacher_config: MQSConfig | None = None,
        baseline_segments: int = 96,
        reference_backend: str = CANONICAL_REFERENCE_BACKEND,
        source: str = "initial",
        split: str | None = None,
    ) -> DatasetRecord:
        reference_backend = (
            _normalize_reference_backend(
                reference_backend
            )
        )
        sample = (
            TeacherSample.generate(
                scene,
                frequency_hz,
                teacher_config=(
                    teacher_config
                ),
                baseline_segments=(
                    baseline_segments
                ),
                reference_backend=(
                    reference_backend
                ),
            )
        )
        return self.add_sample(
            sample,
            teacher_config=(
                teacher_config
            ),
            reference_backend=(
                reference_backend
            ),
            source=source,
            split=split,
        )

    def _record(
        self,
        sample_id: str,
    ) -> DatasetRecord:
        for record in (
            self.records
        ):
            if (
                record.sample_id
                == sample_id
            ):
                return record
        raise KeyError(
            f"unknown sample id: {sample_id}"
        )

    def load_sample(
        self,
        sample_id: str,
    ) -> TeacherSample:
        record = self._record(
            sample_id
        )
        path = (
            self.root
            / record.file
        )
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            encoded = (
                EncodedScene(
                    np.asarray(
                        data[
                            "node_features"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "pair_features"
                        ],
                        dtype=float,
                    ),
                    float(
                        data[
                            "length_scale"
                        ]
                    ),
                )
            )
            baseline_r = (
                np.asarray(
                    data[
                        "baseline_resistance"
                    ],
                    dtype=float,
                )
            )
            baseline_x = (
                np.asarray(
                    data[
                        "baseline_reactance"
                    ],
                    dtype=float,
                )
            )
            target = np.asarray(
                data[
                    "target_impedance"
                ],
                dtype=complex,
            )
            stored_channels = (
                np.asarray(
                    data[
                        "target_dissipation_channels"
                    ],
                    dtype=complex,
                )
            )
            channels = (
                None
                if (
                    stored_channels.size
                    == 0
                )
                else (
                    stored_channels
                )
            )
            spatial_coil = (
                np.asarray(
                    data[
                        "spatial_coil_index"
                    ],
                    dtype=int,
                )
            )
            if (
                spatial_coil.size
                == 0
            ):
                spatial = None
            else:
                spatial = (
                    SpatialLossSamples(
                        spatial_coil,
                        np.asarray(
                            data[
                                "spatial_arc_fraction"
                            ],
                            dtype=float,
                        ),
                        np.asarray(
                            data[
                                "spatial_xy"
                            ],
                            dtype=float,
                        ),
                        np.asarray(
                            data[
                                "spatial_weights"
                            ],
                            dtype=float,
                        ),
                        np.asarray(
                            data[
                                "spatial_dissipation_matrix"
                            ],
                            dtype=complex,
                        ),
                    )
                )
        return TeacherSample(
            scene_from_dict(
                record.scene
            ),
            record.frequency_hz,
            encoded,
            baseline_r,
            baseline_x,
            target,
            record.baseline_segments,
            channels,
            spatial,
            record.reference_backend,
        )

    def iter_samples(
        self,
        split: str | None = None,
    ):
        if (
            split is not None
            and split
            not in SPLITS
        ):
            raise ValueError(
                f"unknown split: {split}"
            )
        for record in (
            self.records
        ):
            if (
                split is None
                or record.split
                == split
            ):
                yield self.load_sample(
                    record.sample_id
                )


def migrate_dataset_v3_to_v4(
    root,
):
    """Migrate metadata only; expensive truth NPZ bytes are preserved."""
    root = Path(
        root
    )
    manifest_path = (
        root
        / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset manifest does not exist: {manifest_path}"
        )
    original_bytes = (
        manifest_path.read_bytes()
    )
    manifest = json.loads(
        original_bytes.decode(
            "utf-8"
        )
    )
    schema = manifest.get(
        "schema"
    )
    if (
        schema
        == DATASET_SCHEMA
    ):
        return {
            "migrated": False,
            "from_schema": (
                DATASET_SCHEMA
            ),
            "to_schema": (
                DATASET_SCHEMA
            ),
            "records": len(
                manifest.get(
                    "records",
                    ()
                )
            ),
        }
    if schema != 3:
        raise ValueError(
            "only vNext dataset schema 3 can be migrated to schema 4"
        )

    backup = (
        root
        / "manifest.schema3.json"
    )
    if not backup.exists():
        backup.write_bytes(
            original_bytes
        )

    migrated_records = []
    for record in manifest.get(
        "records",
        ()
    ):
        identity = (
            _identity_payload(
                scene=record[
                    "scene"
                ],
                frequency_hz=record[
                    "frequency_hz"
                ],
                baseline_segments=record[
                    "baseline_segments"
                ],
                teacher_config=record[
                    "teacher_config"
                ],
                reference_backend="mqs",
            )
        )
        new_id = content_hash(
            identity
        )
        old_path = (
            root
            / record[
                "file"
            ]
        )
        if not old_path.is_file():
            raise FileNotFoundError(
                f"dataset sample is missing: {old_path}"
            )
        new_relative = (
            Path(
                "samples"
            )
            / f"{new_id}.npz"
        )
        new_path = (
            root
            / new_relative
        )
        new_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if (
            old_path.resolve()
            != new_path.resolve()
        ):
            if new_path.exists():
                if (
                    new_path.read_bytes()
                    != old_path.read_bytes()
                ):
                    raise FileExistsError(
                        f"migration target already exists with different bytes: {new_path}"
                    )
                old_path.unlink()
            else:
                old_path.replace(
                    new_path
                )

        migrated = dict(
            record
        )
        migrated[
            "sample_id"
        ] = new_id
        migrated[
            "file"
        ] = str(
            new_relative
        )
        migrated[
            "reference_backend"
        ] = "mqs"
        migrated_records.append(
            migrated
        )

    migrated_records.sort(
        key=lambda item: (
            item[
                "sample_id"
            ]
        )
    )
    new_manifest = {
        "schema": (
            DATASET_SCHEMA
        ),
        "split_seed": int(
            manifest[
                "split_seed"
            ]
        ),
        "split_fractions": [
            float(
                value
            )
            for value
            in manifest[
                "split_fractions"
            ]
        ],
        "records": (
            migrated_records
        ),
    }
    temporary = (
        manifest_path.with_suffix(
            ".json.tmp"
        )
    )
    temporary.write_text(
        canonical_json(
            new_manifest
        ),
        encoding="utf-8",
    )
    temporary.replace(
        manifest_path
    )
    return {
        "migrated": True,
        "from_schema": 3,
        "to_schema": (
            DATASET_SCHEMA
        ),
        "records": len(
            migrated_records
        ),
        "backup": str(
            backup
        ),
    }
