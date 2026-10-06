from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from .bundle import LoadedVNextBundle
from .device import resolve_torch_device
from .system import MeshfreeVNextSystem, mvp_system_capabilities
from .tensor_artifact_io import (
    load_tensor_port_artifact,
    load_tensor_spatial_artifact,
)
from .tensor_neural import TensorHybridNeuralResidualArtifact
from .tensor_spatial_neural import (
    TensorHybridSpatialLossArtifact,
    tensor_port_fingerprint,
)


TENSOR_BUNDLE_SCHEMA = 1


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest):
    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def publish_tensor_bundle(
    output,
    port_artifact: TensorHybridNeuralResidualArtifact,
    *,
    spatial_artifact: TensorHybridSpatialLossArtifact | None = None,
    metadata=None,
    overwrite: bool = False,
):
    """Publish a self-contained tensor-electric FAST artifact directory."""
    if not bool(getattr(port_artifact, "supports_tensor_electric", False)):
        raise TypeError("tensor bundle requires a tensor-aware port artifact")
    fingerprint = tensor_port_fingerprint(port_artifact)
    if spatial_artifact is not None:
        if not bool(getattr(spatial_artifact, "supports_tensor_electric", False)):
            raise TypeError("tensor bundle requires a tensor-aware spatial artifact")
        if spatial_artifact.port_fingerprint != fingerprint:
            raise ValueError("tensor spatial artifact is bound to a different port model")

    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=output.name + ".tmp.",
            dir=output.parent,
        )
    )
    try:
        port_path = temporary / "port.pt"
        port_artifact.save(port_path)
        files = {
            "port": {
                "path": "port.pt",
                "sha256": _file_sha256(port_path),
            }
        }
        if spatial_artifact is not None:
            spatial_path = temporary / "spatial.pt"
            spatial_artifact.save(spatial_path)
            files["spatial"] = {
                "path": "spatial.pt",
                "sha256": _file_sha256(spatial_path),
            }
        manifest = {
            "schema": TENSOR_BUNDLE_SCHEMA,
            "model_family": "sdfmpneo_vnext_tensor_electric",
            "artifact_family": "tensor_hybrid",
            "capabilities": asdict(mvp_system_capabilities()),
            "port_fingerprint": fingerprint,
            "files": files,
            "metadata": {} if metadata is None else dict(metadata),
        }
        _write_manifest(temporary / "manifest.json", manifest)
        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        temporary.replace(output)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_files(root: Path, manifest):
    files = manifest.get("files", {})
    if "port" not in files:
        raise ValueError("tensor bundle manifest has no port artifact")
    resolved = {}
    for name, info in files.items():
        relative = Path(info["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("tensor bundle contains an unsafe relative path")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"tensor bundle file is missing: {path}")
        if _file_sha256(path) != str(info.get("sha256", "")):
            raise ValueError(f"tensor bundle checksum mismatch for {name}")
        resolved[name] = path
    return resolved


def load_tensor_bundle(
    root,
    *,
    device: str = "auto",
) -> LoadedVNextBundle:
    """Load a tensor-electric FAST bundle on the requested/available device."""
    resolved_device = resolve_torch_device(device)
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"tensor bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema", -1)) != TENSOR_BUNDLE_SCHEMA:
        raise ValueError("unsupported tensor bundle schema")
    if manifest.get("model_family") != "sdfmpneo_vnext_tensor_electric":
        raise ValueError("tensor bundle model family is incompatible")
    if manifest.get("artifact_family") != "tensor_hybrid":
        raise ValueError("tensor bundle artifact family is incompatible")
    if manifest.get("capabilities") != asdict(mvp_system_capabilities()):
        raise ValueError("tensor bundle capability domain is incompatible")

    files = _validated_files(root, manifest)
    port = load_tensor_port_artifact(
        files["port"],
        device=resolved_device,
    )
    fingerprint = tensor_port_fingerprint(port)
    if fingerprint != str(manifest.get("port_fingerprint", "")):
        raise ValueError("tensor bundle port fingerprint mismatch")
    spatial = None
    if "spatial" in files:
        spatial = load_tensor_spatial_artifact(
            files["spatial"],
            port,
            device=resolved_device,
        )
    system = MeshfreeVNextSystem(
        port,
        spatial_artifact=spatial,
    )
    return LoadedVNextBundle(
        root,
        system,
        port,
        spatial,
        None,
        manifest,
    )
