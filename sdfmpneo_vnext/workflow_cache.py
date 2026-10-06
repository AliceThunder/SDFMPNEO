from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
from typing import Callable, Iterable

import numpy as np

from .em import MQSConfig
from .sampling import HybridSceneSamplerConfig, MVPSceneSamplerConfig
from .tensor_sampling import TensorHybridSceneSamplerConfig, sample_tensor_hybrid_scene
from .tensor_teacher_pipeline import generate_tensor_teacher_once


CACHE_SCHEMA = 1


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def build_sampler_config(mapping) -> TensorHybridSceneSamplerConfig:
    cfg = dict(mapping or {})
    base_cfg = dict(cfg.pop("base", {}) or {})
    conductor_cfg = dict(base_cfg.pop("conductor", {}) or {})
    conductor = MVPSceneSamplerConfig(**conductor_cfg)
    base = HybridSceneSamplerConfig(conductor=conductor, **base_cfg)
    return TensorHybridSceneSamplerConfig(base=base, **cfg)


def build_teacher_config(mapping) -> MQSConfig:
    return MQSConfig(**dict(mapping or {}))


def teacher_cache_payload(config) -> dict:
    data = dict(config["DATA"])
    return {
        "schema": CACHE_SCHEMA,
        "seed": int(data["seed"]),
        "sampler": _jsonable(config["SAMPLER"]),
        "teacher": _jsonable(config["TEACHER"]),
        "truth": _jsonable(config["TRUTH"]),
    }


def teacher_cache_key(config) -> str:
    return sha256(canonical_json(teacher_cache_payload(config)).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_init(native_threads: int):
    if native_threads <= 0:
        return
    from threadpoolctl import threadpool_limits

    # Keep the limiter object alive for the lifetime of the worker process.
    global _THREADPOOL_LIMITER
    _THREADPOOL_LIMITER = threadpool_limits(limits=int(native_threads))


def _generate_job(payload):
    index, seed, sampler, teacher, truth = payload
    rng = np.random.default_rng([int(seed), int(index)])
    scene, frequency_hz = sample_tensor_hybrid_scene(rng, sampler)
    sample = generate_tensor_teacher_once(
        scene,
        frequency_hz,
        teacher_config=teacher,
        **truth,
    )
    return int(index), sample


class TensorTeacherCache:
    """Content-addressed, append-only cache of deterministic teacher samples.

    The cache directory identity intentionally excludes requested sample count,
    worker count, accelerator selection and optimizer settings. Increasing the
    sample budget therefore only generates missing deterministic index shards.
    """

    def __init__(self, root, config, *, verify_checksums: bool = True):
        self.root = Path(root).expanduser().resolve(strict=False)
        self.payload = teacher_cache_payload(config)
        self.key = teacher_cache_key(config)
        self.path = self.root / self.key
        self.samples_dir = self.path / "samples"
        self.manifest_path = self.path / "manifest.json"
        self.verify_checksums = bool(verify_checksums)
        self.path.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self):
        if self.manifest_path.is_file():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if int(payload.get("schema", -1)) != CACHE_SCHEMA:
                raise ValueError("unsupported tensor teacher cache schema")
            if payload.get("cache_key") != self.key:
                raise ValueError("tensor teacher cache key mismatch")
            if canonical_json(payload.get("identity")) != canonical_json(self.payload):
                raise ValueError("tensor teacher cache identity mismatch")
            return payload
        payload = {
            "schema": CACHE_SCHEMA,
            "cache_key": self.key,
            "identity": self.payload,
            "samples": {},
        }
        self._write_manifest(payload)
        return payload

    def _write_manifest(self, manifest=None):
        if manifest is not None:
            self._manifest = manifest
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                self._manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def _sample_path(self, index: int) -> Path:
        return self.samples_dir / f"{int(index):08d}.pkl"

    def _valid_entry(self, index: int) -> bool:
        entry = self._manifest.get("samples", {}).get(str(int(index)))
        if not entry:
            return False
        path = self.path / entry["path"]
        if not path.is_file():
            return False
        if self.verify_checksums and _sha256_file(path) != str(entry.get("sha256", "")):
            return False
        return True

    def existing_indices(self) -> tuple[int, ...]:
        result = []
        for key in self._manifest.get("samples", {}):
            try:
                index = int(key)
            except ValueError:
                continue
            if self._valid_entry(index):
                result.append(index)
        return tuple(sorted(result))

    def missing_indices(self, count: int) -> tuple[int, ...]:
        if int(count) < 1:
            raise ValueError("teacher sample count must be positive")
        return tuple(index for index in range(int(count)) if not self._valid_entry(index))

    def load(self, index: int):
        if not self._valid_entry(index):
            raise FileNotFoundError(f"teacher cache sample {index} is unavailable or corrupt")
        entry = self._manifest["samples"][str(int(index))]
        with (self.path / entry["path"]).open("rb") as handle:
            return pickle.load(handle)

    def load_many(self, count: int):
        return tuple(self.load(index) for index in range(int(count)))

    def store(self, index: int, sample):
        index = int(index)
        path = self._sample_path(index)
        temporary = path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        digest = _sha256_file(path)
        self._manifest.setdefault("samples", {})[str(index)] = {
            "path": str(path.relative_to(self.path)),
            "sha256": digest,
            "size": int(path.stat().st_size),
        }
        self._write_manifest()

    def ensure(
        self,
        count: int,
        *,
        config,
        checkpoint: Callable[[], None] | None = None,
        progress: Callable[[dict], None] | None = None,
    ):
        """Generate only missing shards, returning samples [0, count)."""
        data = dict(config["DATA"])
        missing = list(self.missing_indices(count))
        if progress is not None:
            progress(
                {
                    "phase": "teacher_cache",
                    "cache_key": self.key,
                    "requested": int(count),
                    "cached": int(count) - len(missing),
                    "missing": len(missing),
                    "cache_path": str(self.path),
                }
            )
        if not missing:
            return self.load_many(count)

        sampler = build_sampler_config(config["SAMPLER"])
        teacher = build_teacher_config(config["TEACHER"])
        truth = dict(config["TRUTH"])
        seed = int(data["seed"])
        workers = max(1, int(data["workers"]))
        native_threads = max(0, int(data["native_threads_per_worker"]))
        chunk_size = max(1, int(data.get("generation_chunk_size", workers)))

        completed = int(count) - len(missing)
        for start in range(0, len(missing), chunk_size):
            if checkpoint is not None:
                checkpoint()
            indices = missing[start : start + chunk_size]
            jobs = [
                (index, seed, sampler, teacher, truth)
                for index in indices
            ]
            if workers == 1:
                if native_threads > 0:
                    from threadpoolctl import threadpool_limits

                    with threadpool_limits(limits=native_threads):
                        results = list(map(_generate_job, jobs))
                else:
                    results = list(map(_generate_job, jobs))
            else:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(jobs)),
                    initializer=_worker_init,
                    initargs=(native_threads,),
                ) as executor:
                    results = list(executor.map(_generate_job, jobs, chunksize=1))
            for index, sample in results:
                self.store(index, sample)
                completed += 1
                if progress is not None:
                    progress(
                        {
                            "phase": "teacher_cache",
                            "cache_key": self.key,
                            "requested": int(count),
                            "cached": completed,
                            "missing": int(count) - completed,
                            "sample_index": int(index),
                            "cache_path": str(self.path),
                        }
                    )
            if checkpoint is not None:
                checkpoint()
        return self.load_many(count)
