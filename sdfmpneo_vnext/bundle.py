from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from .dataset import DATASET_SCHEMA
from .system import MeshfreeVNextSystem
from .uncertainty import (
    FastErrorCalibrator,
    calibrated_fast_predict,
)


BUNDLE_SCHEMA = 1


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
    overwrite: bool = False,
):
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
    if (
        calibrator is not None
        and calibrator.ensemble_size
        != 1
    ):
        raise ValueError(
            "bundle v1 supports only calibrators fitted to one port artifact"
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
            "reference_backend": "mixed",
            "dataset_schema": (
                DATASET_SCHEMA
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
    if (
        manifest.get(
            "reference_backend"
        )
        != "mixed"
    ):
        raise ValueError(
            "bundle reference backend is incompatible with this runtime"
        )
    if int(
        manifest.get(
            "dataset_schema",
            -1,
        )
    ) != DATASET_SCHEMA:
        raise ValueError(
            "bundle dataset schema is incompatible with this runtime"
        )

    files = (
        _validate_bundle_files(
            root,
            manifest,
        )
    )

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

    spatial = None
    if "spatial" in files:
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
