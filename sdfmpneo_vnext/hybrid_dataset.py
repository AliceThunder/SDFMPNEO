from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from .dataset import (
    SPLITS,
    deterministic_split,
)
from .em import MQSConfig
from .features import EncodedScene
from .hybrid_features import EncodedHybridScene
from .hybrid_training_data import (
    HYBRID_REFERENCE_BACKEND,
    BackgroundSpatialLossSamples,
    HybridTeacherSample,
    PackageSpatialLossSamples,
)
from .training_data import SpatialLossSamples
from .serialization import (
    canonical_json,
    content_hash,
    scene_from_dict,
    scene_to_dict,
)


HYBRID_DATASET_SCHEMA = 3
_HYBRID_DATASET_READ_SCHEMAS = (
    1,
    2,
    HYBRID_DATASET_SCHEMA,
)


@dataclass(frozen=True)
class HybridDatasetRecord:
    sample_id: str
    split: str
    source: str
    frequency_hz: float
    baseline_segments: int
    surface_vertical_order: int
    surface_azimuthal_order: int
    file: str
    scene: dict
    teacher_config: dict
    reference_backend: str
    package_volume_axial_order: int = 0
    package_volume_radial_order: int = 0
    package_volume_azimuthal_order: int = 0
    background_radial_order: int = 0
    background_angular_order: int = 0


class ImmutableHybridTeacherDataset:
    """Content-addressed package-aware teacher dataset.

    Validation/test/release membership is frozen at sample insertion. Active
    learning may append only to the training split.
    """

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
        self.schema = int(
            self._manifest.get(
                "schema",
                -1,
            )
        )
        if (
            self.schema
            not in _HYBRID_DATASET_READ_SCHEMAS
        ):
            raise ValueError(
                "unsupported hybrid dataset schema"
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
        domain_metadata=None,
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
                value
            )
            for value
            in split_fractions
        ]
        deterministic_split(
            "0" * 64,
            seed=split_seed,
            fractions=fractions,
        )
        payload = {
            "schema": (
                HYBRID_DATASET_SCHEMA
            ),
            "kind": (
                "hybrid_dielectric"
            ),
            "split_seed": int(
                split_seed
            ),
            "split_fractions": (
                fractions
            ),
            "domain_metadata": (
                {}
                if domain_metadata
                is None
                else json.loads(
                    canonical_json(
                        domain_metadata
                    )
                )
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
    def domain_metadata(
        self,
    ):
        value = self._manifest.get(
            "domain_metadata",
            {},
        )
        return json.loads(
            canonical_json(
                value
            )
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
            HybridDatasetRecord(
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
        config = (
            teacher_config
            or MQSConfig()
        )
        if not isinstance(
            config,
            MQSConfig,
        ):
            raise TypeError(
                "teacher_config must be MQSConfig"
            )
        return asdict(
            config
        )

    def _identity(
        self,
        sample: HybridTeacherSample,
        teacher_config,
    ):
        teacher_dict = (
            self._teacher_dict(
                teacher_config
            )
        )
        identity = {
            "schema": (
                HYBRID_DATASET_SCHEMA
            ),
            "kind": (
                "hybrid_dielectric"
            ),
            "scene": scene_to_dict(
                sample.scene
            ),
            "frequency_hz": float(
                sample.frequency_hz
            ),
            "baseline_segments": int(
                sample.baseline_segments
            ),
            "surface_vertical_order": int(
                sample.surface_vertical_order
            ),
            "surface_azimuthal_order": int(
                sample.surface_azimuthal_order
            ),
            "package_volume_axial_order": int(
                sample.package_volume_axial_order
            ),
            "package_volume_radial_order": int(
                sample.package_volume_radial_order
            ),
            "package_volume_azimuthal_order": int(
                sample.package_volume_azimuthal_order
            ),
            "background_radial_order": int(
                sample.background_radial_order
            ),
            "background_angular_order": int(
                sample.background_angular_order
            ),
            "teacher_config": (
                teacher_dict
            ),
            "reference_backend": (
                HYBRID_REFERENCE_BACKEND
            ),
            "output_schema": (
                "hybrid_port_channels_and_spatial_v3"
            ),
        }
        return (
            content_hash(
                identity
            ),
            identity,
            teacher_dict,
        )

    def add_sample(
        self,
        sample: HybridTeacherSample,
        *,
        teacher_config: MQSConfig | None = None,
        source: str = "initial",
        split: str | None = None,
    ) -> HybridDatasetRecord:
        if self.schema != HYBRID_DATASET_SCHEMA:
            raise RuntimeError(
                "legacy hybrid datasets are read-only; create a schema-v3 "
                "dataset before appending spatial truth"
            )
        if (
            sample.reference_backend
            != HYBRID_REFERENCE_BACKEND
        ):
            raise ValueError(
                "hybrid dataset accepts only canonical dielectric mixed-SIE truth"
            )
        (
            sample_id,
            identity,
            teacher_dict,
        ) = self._identity(
            sample,
            teacher_config,
        )
        existing = {
            record.sample_id: record
            for record
            in self.records
        }
        if sample_id in existing:
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
            / (
                f"{sample_id}.npz"
            )
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

        encoded = (
            sample.encoded
        )
        np.savez_compressed(
            path,
            coil_node_features=(
                encoded.coil.node_features
            ),
            coil_pair_features=(
                encoded.coil.pair_features
            ),
            package_features=(
                encoded.package_features
            ),
            coil_package_features=(
                encoded.coil_package_features
            ),
            package_pair_features=(
                encoded.package_pair_features
            ),
            length_scale=np.asarray(
                encoded.length_scale,
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
                sample.target_dissipation_channels
            ),
            surface_residual=np.asarray(
                sample.surface_residual,
                dtype=float,
            ),
            raw_potential_reciprocity_defect=np.asarray(
                sample.raw_potential_reciprocity_defect,
                dtype=float,
            ),
            power_closure_error=np.asarray(
                sample.power_closure_error,
                dtype=float,
            ),
            conductor_spatial_coil_index=(
                np.asarray(
                    [],
                    dtype=int,
                )
                if sample.conductor_spatial_loss
                is None
                else sample.conductor_spatial_loss.coil_index
            ),
            conductor_spatial_arc_fraction=(
                np.asarray(
                    [],
                    dtype=float,
                )
                if sample.conductor_spatial_loss
                is None
                else sample.conductor_spatial_loss.arc_fraction
            ),
            conductor_spatial_xy=(
                np.empty(
                    (
                        0,
                        2,
                    ),
                    dtype=float,
                )
                if sample.conductor_spatial_loss
                is None
                else sample.conductor_spatial_loss.xy
            ),
            conductor_spatial_weights=(
                np.asarray(
                    [],
                    dtype=float,
                )
                if sample.conductor_spatial_loss
                is None
                else sample.conductor_spatial_loss.weights
            ),
            conductor_spatial_matrix=(
                np.empty(
                    (
                        0,
                        sample.target_impedance.shape[
                            0
                        ],
                        sample.target_impedance.shape[
                            1
                        ],
                    ),
                    dtype=complex,
                )
                if sample.conductor_spatial_loss
                is None
                else sample.conductor_spatial_loss.dissipation_matrix
            ),
            package_spatial_package_index=(
                np.asarray(
                    [],
                    dtype=int,
                )
                if sample.package_spatial_loss
                is None
                else sample.package_spatial_loss.package_index
            ),
            package_spatial_local_position=(
                np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=float,
                )
                if sample.package_spatial_loss
                is None
                else sample.package_spatial_loss.local_position
            ),
            package_spatial_weights=(
                np.asarray(
                    [],
                    dtype=float,
                )
                if sample.package_spatial_loss
                is None
                else sample.package_spatial_loss.weights
            ),
            package_spatial_matrix=(
                np.empty(
                    (
                        0,
                        sample.target_impedance.shape[
                            0
                        ],
                        sample.target_impedance.shape[
                            1
                        ],
                    ),
                    dtype=complex,
                )
                if sample.package_spatial_loss
                is None
                else sample.package_spatial_loss.dissipation_matrix
            ),
            background_spatial_root_local_position=(
                np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=float,
                )
                if sample.background_spatial_loss
                is None
                else sample.background_spatial_loss.root_local_position
            ),
            background_spatial_weights=(
                np.asarray(
                    [],
                    dtype=float,
                )
                if sample.background_spatial_loss
                is None
                else sample.background_spatial_loss.weights
            ),
            background_spatial_matrix=(
                np.empty(
                    (
                        0,
                        sample.target_impedance.shape[
                            0
                        ],
                        sample.target_impedance.shape[
                            1
                        ],
                    ),
                    dtype=complex,
                )
                if sample.background_spatial_loss
                is None
                else sample.background_spatial_loss.dissipation_matrix
            ),
        )

        record = HybridDatasetRecord(
            sample_id=sample_id,
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
            surface_vertical_order=int(
                sample.surface_vertical_order
            ),
            surface_azimuthal_order=int(
                sample.surface_azimuthal_order
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
                HYBRID_REFERENCE_BACKEND
            ),
            package_volume_axial_order=int(
                sample.package_volume_axial_order
            ),
            package_volume_radial_order=int(
                sample.package_volume_radial_order
            ),
            package_volume_azimuthal_order=int(
                sample.package_volume_azimuthal_order
            ),
            background_radial_order=int(
                sample.background_radial_order
            ),
            background_angular_order=int(
                sample.background_angular_order
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
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
        include_spatial_truth: bool = True,
        package_volume_axial_order: int = 8,
        package_volume_radial_order: int = 6,
        package_volume_azimuthal_order: int = 24,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
        source: str = "initial",
        split: str | None = None,
    ):
        sample = (
            HybridTeacherSample.generate(
                scene,
                frequency_hz,
                teacher_config=(
                    teacher_config
                ),
                baseline_segments=(
                    baseline_segments
                ),
                surface_vertical_order=(
                    surface_vertical_order
                ),
                surface_azimuthal_order=(
                    surface_azimuthal_order
                ),
                include_spatial_truth=(
                    include_spatial_truth
                ),
                package_volume_axial_order=(
                    package_volume_axial_order
                ),
                package_volume_radial_order=(
                    package_volume_radial_order
                ),
                package_volume_azimuthal_order=(
                    package_volume_azimuthal_order
                ),
                background_radial_order=(
                    background_radial_order
                ),
                background_angular_order=(
                    background_angular_order
                ),
            )
        )
        return self.add_sample(
            sample,
            teacher_config=(
                teacher_config
            ),
            source=source,
            split=split,
        )

    def _record(
        self,
        sample_id: str,
    ) -> HybridDatasetRecord:
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
    ) -> HybridTeacherSample:
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
            coil = EncodedScene(
                np.asarray(
                    data[
                        "coil_node_features"
                    ],
                    dtype=float,
                ),
                np.asarray(
                    data[
                        "coil_pair_features"
                    ],
                    dtype=float,
                ),
                float(
                    data[
                        "length_scale"
                    ]
                ),
            )
            encoded = EncodedHybridScene(
                coil,
                np.asarray(
                    data[
                        "package_features"
                    ],
                    dtype=float,
                ),
                np.asarray(
                    data[
                        "coil_package_features"
                    ],
                    dtype=float,
                ),
                np.asarray(
                    data[
                        "package_pair_features"
                    ],
                    dtype=float,
                ),
            )
            baseline_resistance = np.asarray(
                data[
                    "baseline_resistance"
                ],
                dtype=float,
            )
            baseline_reactance = np.asarray(
                data[
                    "baseline_reactance"
                ],
                dtype=float,
            )
            target_impedance = np.asarray(
                data[
                    "target_impedance"
                ],
                dtype=complex,
            )
            target_channels = np.asarray(
                data[
                    "target_dissipation_channels"
                ],
                dtype=complex,
            )
            surface_residual = float(
                data[
                    "surface_residual"
                ]
            )
            raw_reciprocity = float(
                data[
                    "raw_potential_reciprocity_defect"
                ]
            )
            closure = float(
                data[
                    "power_closure_error"
                ]
            )
            if (
                "conductor_spatial_coil_index"
                in data.files
                and data[
                    "conductor_spatial_coil_index"
                ].size
                > 0
            ):
                conductor_spatial = SpatialLossSamples(
                    np.asarray(
                        data[
                            "conductor_spatial_coil_index"
                        ],
                        dtype=int,
                    ),
                    np.asarray(
                        data[
                            "conductor_spatial_arc_fraction"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "conductor_spatial_xy"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "conductor_spatial_weights"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "conductor_spatial_matrix"
                        ],
                        dtype=complex,
                    ),
                )
            else:
                conductor_spatial = None

            if (
                "package_spatial_package_index"
                in data.files
                and data[
                    "package_spatial_package_index"
                ].size
                > 0
            ):
                package_spatial = PackageSpatialLossSamples(
                    np.asarray(
                        data[
                            "package_spatial_package_index"
                        ],
                        dtype=int,
                    ),
                    np.asarray(
                        data[
                            "package_spatial_local_position"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "package_spatial_weights"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "package_spatial_matrix"
                        ],
                        dtype=complex,
                    ),
                )
            else:
                package_spatial = None

            if (
                "background_spatial_root_local_position"
                in data.files
                and data[
                    "background_spatial_root_local_position"
                ].size
                > 0
            ):
                background_spatial = BackgroundSpatialLossSamples(
                    np.asarray(
                        data[
                            "background_spatial_root_local_position"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "background_spatial_weights"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        data[
                            "background_spatial_matrix"
                        ],
                        dtype=complex,
                    ),
                )
            else:
                background_spatial = None

        return HybridTeacherSample(
            scene=scene_from_dict(
                record.scene
            ),
            frequency_hz=(
                record.frequency_hz
            ),
            encoded=encoded,
            baseline_resistance=(
                baseline_resistance
            ),
            baseline_reactance=(
                baseline_reactance
            ),
            target_impedance=(
                target_impedance
            ),
            target_dissipation_channels=(
                target_channels
            ),
            baseline_segments=(
                record.baseline_segments
            ),
            surface_vertical_order=(
                record.surface_vertical_order
            ),
            surface_azimuthal_order=(
                record.surface_azimuthal_order
            ),
            surface_residual=(
                surface_residual
            ),
            raw_potential_reciprocity_defect=(
                raw_reciprocity
            ),
            power_closure_error=(
                closure
            ),
            conductor_spatial_loss=(
                conductor_spatial
            ),
            package_spatial_loss=(
                package_spatial
            ),
            background_spatial_loss=(
                background_spatial
            ),
            package_volume_axial_order=int(
                getattr(
                    record,
                    "package_volume_axial_order",
                    0,
                )
            ),
            package_volume_radial_order=int(
                getattr(
                    record,
                    "package_volume_radial_order",
                    0,
                )
            ),
            package_volume_azimuthal_order=int(
                getattr(
                    record,
                    "package_volume_azimuthal_order",
                    0,
                )
            ),
            background_radial_order=int(
                getattr(
                    record,
                    "background_radial_order",
                    0,
                )
            ),
            background_angular_order=int(
                getattr(
                    record,
                    "background_angular_order",
                    0,
                )
            ),
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
