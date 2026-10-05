from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .em import MQSConfig
from .performance import (
    resolve_torch_device,
    train_tensor_hybrid_residual_surrogate_accelerated,
)
from .sampling import HybridSceneSamplerConfig
from .spatial_performance import (
    train_tensor_hybrid_spatial_loss_surrogate_accelerated,
)
from .tensor_bundle import publish_tensor_bundle
from .tensor_sampling import TensorHybridSceneSamplerConfig
from .tensor_teacher_pipeline import iter_tensor_teacher_samples_parallel_once


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tensor-electric teacher data in local worker processes, "
            "train vectorized CUDA/CPU FAST models, and publish one bundle."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--native-threads-per-worker",
        type=int,
        default=1,
        help=(
            "BLAS/OpenMP threads per teacher worker. Use 1 with several worker "
            "processes to avoid CPU oversubscription; 0 leaves native pools unmanaged."
        ),
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--spatial-epochs", type=int, default=120)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float64"),
        default="auto",
        help="auto uses float32 on CUDA/MPS and float64 on CPU",
    )
    parser.add_argument("--with-spatial", action="store_true")
    parser.add_argument("--tensor-background-probability", type=float, default=0.25)
    parser.add_argument("--baseline-segments", type=int, default=64)
    parser.add_argument("--surface-vertical-order", type=int, default=12)
    parser.add_argument("--surface-azimuthal-order", type=int, default=24)
    parser.add_argument("--package-volume-axial-order", type=int, default=6)
    parser.add_argument("--package-volume-radial-order", type=int, default=4)
    parser.add_argument("--package-volume-azimuthal-order", type=int, default=16)
    parser.add_argument("--background-radial-order", type=int, default=10)
    parser.add_argument("--background-angular-order", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _split(samples, fraction: float, seed: int):
    samples = tuple(samples)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0,1)")
    if not samples:
        raise ValueError("no teacher samples were generated")
    if fraction == 0.0 or len(samples) == 1:
        return samples, ()
    count = max(1, int(round(len(samples) * fraction)))
    count = min(count, len(samples) - 1)
    order = np.random.default_rng(int(seed) + 991).permutation(len(samples))
    validation_indices = set(int(index) for index in order[:count])
    train = tuple(
        sample
        for index, sample in enumerate(samples)
        if index not in validation_indices
    )
    validation = tuple(
        sample
        for index, sample in enumerate(samples)
        if index in validation_indices
    )
    return train, validation


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.count < 1 or args.workers < 1 or args.batch_size < 1:
        raise SystemExit("count, workers, and batch-size must be positive")
    if args.native_threads_per_worker < 0:
        raise SystemExit("native-threads-per-worker must be nonnegative")

    resolved_device = resolve_torch_device(args.device)
    sampler = TensorHybridSceneSamplerConfig(
        base=HybridSceneSamplerConfig(dc_probability=0.0),
        tensor_package_probability=1.0,
        tensor_background_probability=float(args.tensor_background_probability),
    )
    teacher = MQSConfig()
    teacher_options = {
        "baseline_segments": int(args.baseline_segments),
        "surface_vertical_order": int(args.surface_vertical_order),
        "surface_azimuthal_order": int(args.surface_azimuthal_order),
        "include_spatial": bool(args.with_spatial),
        "magnetic_volume_axial_order": int(args.package_volume_axial_order),
        "magnetic_volume_radial_order": int(args.package_volume_radial_order),
        "magnetic_volume_azimuthal_order": int(args.package_volume_azimuthal_order),
        "package_volume_axial_order": int(args.package_volume_axial_order),
        "package_volume_radial_order": int(args.package_volume_radial_order),
        "package_volume_azimuthal_order": int(args.package_volume_azimuthal_order),
        "background_radial_order": int(args.background_radial_order),
        "background_angular_order": int(args.background_angular_order),
    }

    generated = tuple(
        iter_tensor_teacher_samples_parallel_once(
            args.count,
            sampler_config=sampler,
            teacher_config=teacher,
            workers=args.workers,
            native_threads_per_worker=args.native_threads_per_worker,
            seed=args.seed,
            **teacher_options,
        )
    )
    if args.with_spatial:
        port_samples = tuple(sample.port for sample in generated)
        spatial_samples = generated
    else:
        port_samples = generated
        spatial_samples = ()

    train_port, validation_port = _split(
        port_samples,
        args.validation_fraction,
        args.seed,
    )
    port, port_report = train_tensor_hybrid_residual_surrogate_accelerated(
        train_port,
        validation_samples=validation_port,
        epochs=args.epochs,
        batch_size=args.batch_size,
        precision=args.precision,
        device=resolved_device,
        seed=args.seed,
    )

    spatial = None
    spatial_report = None
    if args.with_spatial:
        validation_ids = {id(sample) for sample in validation_port}
        train_spatial = tuple(
            sample
            for sample in spatial_samples
            if id(sample.port) not in validation_ids
        )
        validation_spatial = tuple(
            sample
            for sample in spatial_samples
            if id(sample.port) in validation_ids
        )
        spatial, spatial_report = train_tensor_hybrid_spatial_loss_surrogate_accelerated(
            port,
            train_spatial,
            validation_samples=validation_spatial,
            epochs=args.spatial_epochs,
            batch_size=args.batch_size,
            device=resolved_device,
            seed=args.seed + 10,
        )

    manifest = publish_tensor_bundle(
        args.output,
        port,
        spatial_artifact=spatial,
        metadata={
            "training_device": resolved_device,
            "training_precision": args.precision,
            "teacher_workers": int(args.workers),
            "native_threads_per_worker": int(args.native_threads_per_worker),
            "teacher_samples": int(args.count),
            "batch_size": int(args.batch_size),
            "single_solve_spatial_truth": bool(args.with_spatial),
        },
        overwrite=bool(args.overwrite),
    )
    result = {
        "device": resolved_device,
        "count": len(generated),
        "port": {
            "epochs": port_report.epochs,
            "best_epoch": port_report.best_epoch,
            "final_loss": port_report.final_loss,
            "best_validation_score": port_report.best_validation_score,
            "stopped_early": port_report.stopped_early,
        },
        "spatial": None
        if spatial_report is None
        else {
            "epochs": spatial_report.epochs,
            "best_epoch": spatial_report.best_epoch,
            "final_loss": spatial_report.final_loss,
            "best_validation_error": spatial_report.best_validation_error,
            "best_validation_shape_error": spatial_report.best_validation_shape_error,
            "stopped_early": spatial_report.stopped_early,
        },
        "bundle": str(args.output),
        "manifest": manifest,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
