from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from .dataset import DATASET_SCHEMA
from .hybrid_dataset import HYBRID_DATASET_SCHEMA
from .hybrid_training_data import HYBRID_REFERENCE_BACKEND
from .system import (
    MeshfreeVNextSystem,
    mvp_system_capabilities,
)
from .uncertainty import (
    FastErrorCalibrator,
    calibrated_fast_predict,
)


BUNDLE_SCHEMA = 3


def _file_sha256(
    path: Path,
) -> str:
    digest = sha256()
    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024
                * 1024
            )
            if not block:
                break
            digest.update(
                block
            )
    return digest.hexdigest()


def _validate_release_gate(
    release_gate,
):
    if release_gate is None:
        return None
    if not isinstance(
        release_gate,
        dict,
    ):
        raise TypeError(
            "release_gate must be a dictionary"
        )
    normalized = dict(
        release_gate
    )
    required = (
        "port",
        "spatial",
        "certified",
    )
    missing = [
        name
        for name
        in required
        if name not in normalized
    ]
    if missing:
        raise ValueError(
            "release_gate is missing required audits: "
            + ", ".join(
                missing
            )
        )
    failed = [
        name
        for name
        in required
        if not bool(
            normalized[
                name
            ].get(
                "passed",
                False,
            )
        )
    ]
    if failed:
        raise ValueError(
            "release_gate contains failed audits: "
            + ", ".join(
                failed
            )
        )
    return normalized


def _write_manifest(
    path: Path,
    payload,
):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class LoadedVNextBundle:
    root: Path
    system: MeshfreeVNextSystem
    port_artifact: object
    spatial_artifact: object | None
    calibrator: FastErrorCalibrator | None
    manifest: dict

    def calibrated_fast(
        self,
        scene,
        frequency_hz: float,
        *,
        baseline_segments: int | None = None,
    ):
        if self.calibrator is None:
            raise RuntimeError(
                "bundle does not contain an uncertainty calibrator"
            )
        if (
            self.calibrator.ensemble_size
            != 1
        ):
            raise RuntimeError(
                "single-artifact bundle cannot use a multi-model calibrator"
            )
        if baseline_segments is None:
            baseline_segments = int(
                getattr(
                    self.port_artifact,
                    "baseline_segments",
                    64,
                )
            )
        return calibrated_fast_predict(
            (
                self.port_artifact,
            ),
            self.calibrator,
            scene,
            frequency_hz,
            baseline_segments=(
                baseline_segments
            ),
        )


def publish_bundle(
    output,
    port_artifact,
    *,
    spatial_artifact=None,
    calibrator: FastErrorCalibrator | None = None,
    metadata=None,
    release_gate=None,
    overwrite: bool = False,
):
    release_gate = (
        _validate_release_gate(
            release_gate
        )
    )
    output = Path(
        output
    )
    if (
        output.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"bundle already exists: {output}"
        )
    if not hasattr(
        port_artifact,
        "fingerprint",
    ):
        raise TypeError(
            "bundle requires a fingerprinted neural port artifact"
        )
    port_fingerprint = (
        port_artifact.fingerprint()
    )
    artifact_family = (
        "hybrid"
        if bool(
            getattr(
                port_artifact,
                "supports_packages",
                False,
            )
        )
        else "conductor"
    )
    expected_spatial_packages = (
        artifact_family
        == "hybrid"
    )
    if (
        calibrator is not None
        and calibrator.ensemble_size
        != 1
    ):
        raise ValueError(
            "bundle v2 supports only calibrators fitted to one port artifact"
        )
    if (
        spatial_artifact is not None
        and bool(
            getattr(
                spatial_artifact,
                "supports_packages",
                False,
            )
        )
        != expected_spatial_packages
    ):
        raise ValueError(
            "port and spatial artifacts belong to different model families"
        )
    if (
        spatial_artifact is not None
        and getattr(
            spatial_artifact,
            "port_fingerprint",
            None,
        )
        != port_fingerprint
    ):
        raise ValueError(
            "spatial artifact is bound to a different port artifact"
        )
    if (
        calibrator is not None
        and calibrator.artifact_fingerprints
        and calibrator.artifact_fingerprints
        != (
            port_fingerprint,
        )
    ):
        raise ValueError(
            "calibrator is bound to a different port artifact"
        )
    if (
        release_gate is not None
        and calibrator is not None
        and not calibrator.artifact_fingerprints
    ):
        raise ValueError(
            "released bundle requires a calibrator bound to the exact port artifact"
        )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=(
                output.name
                + ".tmp."
            ),
            dir=output.parent,
        )
    )
    try:
        port_path = (
            temporary
            / "port.pt"
        )
        port_artifact.save(
            port_path
        )
        files = {
            "port": {
                "path": "port.pt",
                "sha256": _file_sha256(
                    port_path
                ),
            }
        }

        if spatial_artifact is not None:
            spatial_path = (
                temporary
                / "spatial.pt"
            )
            spatial_artifact.save(
                spatial_path
            )
            files[
                "spatial"
            ] = {
                "path": "spatial.pt",
                "sha256": _file_sha256(
                    spatial_path
                ),
            }

        if calibrator is not None:
            calibrator_path = (
                temporary
                / "calibrator.json"
            )
            calibrator.save(
                calibrator_path
            )
            files[
                "calibrator"
            ] = {
                "path": "calibrator.json",
                "sha256": _file_sha256(
                    calibrator_path
                ),
            }

        manifest = {
            "schema": BUNDLE_SCHEMA,
            "model_family": (
                "sdfmpneo_vnext_meshfree_mvp"
            ),
            "capabilities": asdict(
                mvp_system_capabilities()
            ),
            "artifact_family": (
                artifact_family
            ),
            "reference_backend": (
                HYBRID_REFERENCE_BACKEND
                if artifact_family
                == "hybrid"
                else "mixed"
            ),
            "dataset_schema": (
                HYBRID_DATASET_SCHEMA
                if artifact_family
                == "hybrid"
                else DATASET_SCHEMA
            ),
            "port_fingerprint": (
                port_fingerprint
            ),
            "calibrator_bound": bool(
                calibrator is not None
                and calibrator.artifact_fingerprints
            ),
            "release_status": (
                "released"
                if release_gate is not None
                else "development"
            ),
            "release_gate": (
                release_gate
            ),
            "files": files,
            "metadata": (
                {}
                if metadata is None
                else dict(
                    metadata
                )
            ),
        }
        _write_manifest(
            temporary
            / "manifest.json",
            manifest,
        )

        if output.exists():
            if output.is_dir():
                shutil.rmtree(
                    output
                )
            else:
                output.unlink()
        temporary.replace(
            output
        )
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )
        raise


def _validate_bundle_files(
    root: Path,
    manifest,
):
    files = manifest.get(
        "files",
        {}
    )
    if "port" not in files:
        raise ValueError(
            "bundle manifest has no port artifact"
        )
    resolved = {}
    for name, info in files.items():
        relative = Path(
            info[
                "path"
            ]
        )
        if (
            relative.is_absolute()
            or ".."
            in relative.parts
        ):
            raise ValueError(
                "bundle contains an unsafe relative path"
            )
        path = (
            root
            / relative
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"bundle file is missing: {path}"
            )
        actual = _file_sha256(
            path
        )
        expected = str(
            info.get(
                "sha256",
                "",
            )
        )
        if actual != expected:
            raise ValueError(
                f"bundle checksum mismatch for {name}"
            )
        resolved[
            name
        ] = path
    return resolved


def load_bundle(
    root,
    *,
    device: str = "cpu",
    require_release: bool = False,
) -> LoadedVNextBundle:
    root = Path(
        root
    )
    manifest_path = (
        root
        / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"bundle manifest is missing: {manifest_path}"
        )
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )
    if (
        manifest.get(
            "schema"
        )
        != BUNDLE_SCHEMA
    ):
        raise ValueError(
            "unsupported vNext bundle schema"
        )
    release_status = str(
        manifest.get(
            "release_status",
            "development",
        )
    )
    if release_status not in (
        "development",
        "released",
    ):
        raise ValueError(
            "bundle release_status is invalid"
        )
    if (
        release_status
        == "released"
    ):
        _validate_release_gate(
            manifest.get(
                "release_gate"
            )
        )
    elif require_release:
        raise ValueError(
            "bundle is development-only and has not passed the locked release gate"
        )

    expected_capabilities = asdict(
        mvp_system_capabilities()
    )
    if (
        manifest.get(
            "capabilities"
        )
        != expected_capabilities
    ):
        raise ValueError(
            "bundle capability domain is incompatible with this runtime"
        )

    artifact_family = str(
        manifest.get(
            "artifact_family",
            "conductor",
        )
    )
    if artifact_family not in (
        "conductor",
        "hybrid",
    ):
        raise ValueError(
            "bundle artifact_family is invalid"
        )
    expected_reference_backend = (
        HYBRID_REFERENCE_BACKEND
        if artifact_family
        == "hybrid"
        else "mixed"
    )
    expected_dataset_schema = (
        HYBRID_DATASET_SCHEMA
        if artifact_family
        == "hybrid"
        else DATASET_SCHEMA
    )
    if (
        manifest.get(
            "reference_backend"
        )
        != expected_reference_backend
    ):
        raise ValueError(
            "bundle reference backend is incompatible with artifact family"
        )
    if int(
        manifest.get(
            "dataset_schema",
            -1,
        )
    ) != expected_dataset_schema:
        raise ValueError(
            "bundle dataset schema is incompatible with artifact family"
        )

    files = (
        _validate_bundle_files(
            root,
            manifest,
        )
    )

    if artifact_family == "hybrid":
        from .hybrid_neural import (
            HybridNeuralResidualArtifact,
        )
        port = (
            HybridNeuralResidualArtifact.load(
                files[
                    "port"
                ],
                device=device,
            )
        )
    else:
        from .neural import (
            NeuralResidualArtifact,
        )
        port = (
            NeuralResidualArtifact.load(
                files[
                    "port"
                ],
                device=device,
            )
        )
    actual_port_fingerprint = (
        port.fingerprint()
    )
    expected_port_fingerprint = str(
        manifest.get(
            "port_fingerprint",
            "",
        )
    )
    if (
        not expected_port_fingerprint
        or expected_port_fingerprint
        != actual_port_fingerprint
    ):
        raise ValueError(
            "bundle port fingerprint mismatch"
        )

    spatial = None
    if "spatial" in files:
        if artifact_family == "hybrid":
            from .hybrid_spatial_neural import (
                HybridSpatialLossArtifact,
            )
            spatial = (
                HybridSpatialLossArtifact.load(
                    files[
                        "spatial"
                    ],
                    port,
                    device=device,
                )
            )
        else:
            from .spatial_neural import (
                NeuralSpatialLossArtifact,
            )
            spatial = (
                NeuralSpatialLossArtifact.load(
                    files[
                        "spatial"
                    ],
                    port,
                    device=device,
                )
            )

    calibrator = None
    if "calibrator" in files:
        calibrator = (
            FastErrorCalibrator.load(
                files[
                    "calibrator"
                ]
            )
        )
        if (
            calibrator.ensemble_size
            != 1
        ):
            raise ValueError(
                "bundle contains a calibrator fitted to a different ensemble size"
            )
        if (
            calibrator.artifact_fingerprints
            and calibrator.artifact_fingerprints
            != (
                actual_port_fingerprint,
            )
        ):
            raise ValueError(
                "bundle calibrator fingerprint does not match the port artifact"
            )

    system = MeshfreeVNextSystem(
        port,
        spatial_artifact=(
            spatial
        ),
    )
    return LoadedVNextBundle(
        root,
        system,
        port,
        spatial,
        calibrator,
        manifest,
    )
