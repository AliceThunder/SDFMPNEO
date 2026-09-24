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
from .training_data import SpatialLossSamples, TeacherSample


DATASET_SCHEMA = 3
SPLITS = (
    "train",
    "validation",
    "test",
    "release",
)


def deterministic_split(
    sample_id: str,
    *,
    seed: int,
    fractions=(0.70, 0.15, 0.10, 0.05),
) -> str:
    fractions = np.asarray(
        fractions,
        dtype=float,
    )
    if (
        fractions.shape != (4,)
        or np.any(fractions < 0.0)
        or not np.isclose(
            np.sum(fractions),
            1.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "fractions must be four nonnegative values summing to one"
        )
    digest = sha256(
        f"{int(seed)}:{sample_id}".encode("utf-8")
    ).digest()
    value = (
        int.from_bytes(
            digest[:8],
            "big",
        )
        / float(2**64)
    )
    edge = np.cumsum(fractions)
    index = int(
        np.searchsorted(
            edge,
            value,
            side="right",
        )
    )
    index = min(index, 3)
    return SPLITS[index]


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


class ImmutableTeacherDataset:
    """Content-addressed vNext teacher dataset with frozen split semantics."""

    def __init__(
        self,
        root,
    ):
        self.root = Path(root)
        self.manifest_path = (
            self.root / "manifest.json"
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
        if (
            self._manifest.get("schema")
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
        root = Path(root)
        root.mkdir(
            parents=True,
            exist_ok=True,
        )
        manifest = root / "manifest.json"
        if manifest.exists():
            raise FileExistsError(
                f"dataset already exists: {manifest}"
            )
        fractions = [
            float(x)
            for x in split_fractions
        ]
        # Reuse the split validator before writing persistent metadata.
        deterministic_split(
            "0" * 64,
            seed=split_seed,
            fractions=fractions,
        )
        payload = {
            "schema": DATASET_SCHEMA,
            "split_seed": int(split_seed),
            "split_fractions": fractions,
            "records": [],
        }
        manifest.write_text(
            canonical_json(payload),
            encoding="utf-8",
        )
        return cls(root)

    @property
    def split_seed(self) -> int:
        return int(
            self._manifest["split_seed"]
        )

    @property
    def split_fractions(self):
        return tuple(
            float(x)
            for x in self._manifest[
                "split_fractions"
            ]
        )

    @property
    def records(self):
        return tuple(
            DatasetRecord(**record)
            for record in self._manifest[
                "records"
            ]
        )

    def counts(self):
        out = {
            split: 0
            for split in SPLITS
        }
        for record in self.records:
            out[record.split] += 1
        return out

    def _write_manifest(self):
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
            teacher_config = MQSConfig()
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
    ):
        teacher_dict = (
            self._teacher_dict(
                teacher_config
            )
        )
        identity = {
            "schema": DATASET_SCHEMA,
            "scene": scene_to_dict(
                sample.scene
            ),
            "frequency_hz": float(
                sample.frequency_hz
            ),
            "baseline_segments": int(
                sample.baseline_segments
            ),
            "teacher_config": teacher_dict,
            "output_schema": (
                "mvp_port_impedance_loss_channels_and_spatial_v3"
            ),
        }
        return (
            content_hash(identity),
            identity,
            teacher_dict,
        )

    def add_sample(
        self,
        sample: TeacherSample,
        *,
        teacher_config: MQSConfig | None = None,
        source: str = "initial",
        split: str | None = None,
    ) -> DatasetRecord:
        sample_id, identity, teacher_dict = (
            self._identity(
                sample,
                teacher_config,
            )
        )
        existing = {
            record.sample_id: record
            for record in self.records
        }
        if sample_id in existing:
            return existing[sample_id]

        if source == "active":
            if (
                split is not None
                and split != "train"
            ):
                raise ValueError(
                    "active-learning samples may only enter the train split"
                )
            split = "train"
        elif split is None:
            split = deterministic_split(
                sample_id,
                seed=self.split_seed,
                fractions=self.split_fractions,
            )
        if split not in SPLITS:
            raise ValueError(
                f"unknown split: {split}"
            )

        relative = (
            Path("samples")
            / f"{sample_id}.npz"
        )
        path = self.root / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if path.exists():
            raise FileExistsError(
                f"orphan sample file already exists: {path}"
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
                if sample.target_dissipation_channels is None
                else sample.target_dissipation_channels
            ),
            spatial_coil_index=(
                np.asarray(
                    [],
                    dtype=int,
                )
                if sample.spatial_loss is None
                else sample.spatial_loss.coil_index
            ),
            spatial_arc_fraction=(
                np.asarray(
                    [],
                    dtype=float,
                )
                if sample.spatial_loss is None
                else sample.spatial_loss.arc_fraction
            ),
            spatial_xy=(
                np.zeros(
                    (0, 2),
                    dtype=float,
                )
                if sample.spatial_loss is None
                else sample.spatial_loss.xy
            ),
            spatial_weights=(
                np.asarray(
                    [],
                    dtype=float,
                )
                if sample.spatial_loss is None
                else sample.spatial_loss.weights
            ),
            spatial_dissipation_matrix=(
                np.zeros(
                    (
                        0,
                        sample.target_impedance.shape[0],
                        sample.target_impedance.shape[1],
                    ),
                    dtype=complex,
                )
                if sample.spatial_loss is None
                else sample.spatial_loss.dissipation_matrix
            ),
        )

        record = DatasetRecord(
            sample_id=sample_id,
            split=split,
            source=str(source),
            frequency_hz=float(
                sample.frequency_hz
            ),
            baseline_segments=int(
                sample.baseline_segments
            ),
            file=str(relative),
            scene=identity["scene"],
            teacher_config=teacher_dict,
        )
        self._manifest["records"].append(
            asdict(record)
        )
        self._manifest["records"].sort(
            key=lambda item: item[
                "sample_id"
            ]
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
        source: str = "initial",
        split: str | None = None,
    ) -> DatasetRecord:
        sample = TeacherSample.generate(
            scene,
            frequency_hz,
            teacher_config=teacher_config,
            baseline_segments=baseline_segments,
        )
        return self.add_sample(
            sample,
            teacher_config=teacher_config,
            source=source,
            split=split,
        )

    def _record(
        self,
        sample_id: str,
    ) -> DatasetRecord:
        for record in self.records:
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
            encoded = EncodedScene(
                np.asarray(
                    data["node_features"],
                    dtype=float,
                ),
                np.asarray(
                    data["pair_features"],
                    dtype=float,
                ),
                float(
                    data["length_scale"]
                ),
            )
            baseline_r = np.asarray(
                data["baseline_resistance"],
                dtype=float,
            )
            baseline_x = np.asarray(
                data["baseline_reactance"],
                dtype=float,
            )
            target = np.asarray(
                data["target_impedance"],
                dtype=complex,
            )
            stored_channels = np.asarray(
                data["target_dissipation_channels"],
                dtype=complex,
            )
            channels = (
                None
                if stored_channels.size == 0
                else stored_channels
            )
            spatial_coil = np.asarray(
                data["spatial_coil_index"],
                dtype=int,
            )
            if spatial_coil.size == 0:
                spatial = None
            else:
                spatial = SpatialLossSamples(
                    spatial_coil,
                    np.asarray(
                        data["spatial_arc_fraction"],
                        dtype=float,
                    ),
                    np.asarray(
                        data["spatial_xy"],
                        dtype=float,
                    ),
                    np.asarray(
                        data["spatial_weights"],
                        dtype=float,
                    ),
                    np.asarray(
                        data["spatial_dissipation_matrix"],
                        dtype=complex,
                    ),
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
        )

    def iter_samples(
        self,
        split: str | None = None,
    ):
        if (
            split is not None
            and split not in SPLITS
        ):
            raise ValueError(
                f"unknown split: {split}"
            )
        for record in self.records:
            if (
                split is None
                or record.split == split
            ):
                yield self.load_sample(
                    record.sample_id
                )
