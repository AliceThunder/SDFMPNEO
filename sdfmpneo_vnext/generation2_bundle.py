from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from .bundle import LoadedVNextBundle
from .device import resolve_torch_device
from .generation2_port import Generation2PortArtifact, generation2_port_fingerprint
from .generation2_spatial import Generation2SpatialArtifact
from .system import MeshfreeVNextSystem, mvp_system_capabilities


GENERATION2_BUNDLE_SCHEMA = 1
GENERATION2_MODEL_FAMILY = "sdfmpneo_vnext_tensor_generation2"
GENERATION2_ARTIFACT_FAMILY = "tensor_hybrid_generation2"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def publish_generation2_bundle(
    output,
    port_artifact: Generation2PortArtifact,
    *,
    spatial_artifact: Generation2SpatialArtifact | None = None,
    metadata=None,
    overwrite: bool = False,
):
    if not isinstance(port_artifact, Generation2PortArtifact):
        raise TypeError("generation-2 bundle requires Generation2PortArtifact")
    port_fingerprint = generation2_port_fingerprint(port_artifact)
    if spatial_artifact is not None:
        if not isinstance(spatial_artifact, Generation2SpatialArtifact):
            raise TypeError("generation-2 bundle requires Generation2SpatialArtifact")
        if spatial_artifact.port_fingerprint != port_fingerprint:
            raise ValueError("generation-2 Spatial artifact is bound to a different Port model")

    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent)
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
            "schema": GENERATION2_BUNDLE_SCHEMA,
            "model_generation": 2,
            "model_family": GENERATION2_MODEL_FAMILY,
            "artifact_family": GENERATION2_ARTIFACT_FAMILY,
            "capabilities": asdict(mvp_system_capabilities()),
            "port_fingerprint": port_fingerprint,
            "partition_fingerprint": port_artifact.partition_fingerprint,
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
        raise ValueError("generation-2 bundle manifest has no Port artifact")
    resolved = {}
    for name, info in files.items():
        relative = Path(info["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("generation-2 bundle contains an unsafe relative path")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"generation-2 bundle file is missing: {path}")
        if _file_sha256(path) != str(info.get("sha256", "")):
            raise ValueError(f"generation-2 bundle checksum mismatch for {name}")
        resolved[name] = path
    return resolved


def load_generation2_bundle(root, *, device: str = "auto") -> LoadedVNextBundle:
    resolved_device = resolve_torch_device(device)
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"generation-2 bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema", -1)) != GENERATION2_BUNDLE_SCHEMA:
        raise ValueError("unsupported generation-2 bundle schema")
    if int(manifest.get("model_generation", -1)) != 2:
        raise ValueError("bundle is not generation-2")
    if manifest.get("model_family") != GENERATION2_MODEL_FAMILY:
        raise ValueError("generation-2 bundle model family is incompatible")
    if manifest.get("artifact_family") != GENERATION2_ARTIFACT_FAMILY:
        raise ValueError("generation-2 bundle artifact family is incompatible")
    if manifest.get("capabilities") != asdict(mvp_system_capabilities()):
        raise ValueError("generation-2 bundle capability domain is incompatible")

    files = _validated_files(root, manifest)
    port = Generation2PortArtifact.load(files["port"], device=resolved_device)
    fingerprint = generation2_port_fingerprint(port)
    if fingerprint != str(manifest.get("port_fingerprint", "")):
        raise ValueError("generation-2 bundle Port fingerprint mismatch")
    if port.partition_fingerprint != manifest.get("partition_fingerprint"):
        raise ValueError("generation-2 bundle partition fingerprint mismatch")
    spatial = None
    if "spatial" in files:
        spatial = Generation2SpatialArtifact.load(
            files["spatial"],
            port,
            device=resolved_device,
        )
    system = MeshfreeVNextSystem(port, spatial_artifact=spatial)
    return LoadedVNextBundle(root, system, port, spatial, None, manifest)


__all__ = [
    "GENERATION2_BUNDLE_SCHEMA",
    "GENERATION2_MODEL_FAMILY",
    "GENERATION2_ARTIFACT_FAMILY",
    "publish_generation2_bundle",
    "load_generation2_bundle",
]
