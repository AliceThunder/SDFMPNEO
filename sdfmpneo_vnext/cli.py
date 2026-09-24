from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from .active_learning import run_active_learning_round
from .analytic_baseline import AnalyticBaselineArtifact
from .bundle import publish_bundle
from .certified import certify_mixed_ports
from .certified_evaluation import audit_certified_release
from .dataset import (
    ImmutableTeacherDataset,
    migrate_dataset_v3_to_v4,
)
from .electrothermal import (
    CoilThermalProperties,
    build_lumped_coil_thermal_model,
)
from .em import MQSConfig
from .evaluation import audit_surrogate
from .geometry import RigidPose, SuperellipseSpiral
from .sampling import sample_two_coil_mvp_scene
from .scene import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    Scene,
)
from .spatial_evaluation import audit_spatial_surrogate
from .system import MeshfreeVNextSystem
from .uncertainty import (
    FastErrorCalibrator,
    fit_fast_error_calibrator,
)


def _demo_scene() -> Scene:
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=0.00393,
    )
    tx = CoilObject(
        SuperellipseSpiral(
            0.035,
            0.030,
            0.9,
            0.002,
            0.002,
            exponent=4.0,
            conductor_width=1.5e-3,
            conductor_thickness=0.9e-3,
        ),
        copper,
        "tx",
    )
    rx = CoilObject(
        SuperellipseSpiral(
            0.028,
            0.024,
            0.8,
            0.0015,
            0.0015,
            exponent=3.5,
            conductor_width=1.4e-3,
            conductor_thickness=0.8e-3,
            pose=RigidPose.from_axis_angle(
                (1.0, 0.0, 0.0),
                np.deg2rad(15.0),
                translation=(
                    0.006,
                    0.0,
                    0.024,
                ),
            ),
        ),
        copper,
        "rx",
    )
    return Scene(
        (
            tx,
            rx,
        ),
        HomogeneousMedium(),
    )


def _reference_config() -> MQSConfig:
    return MQSConfig(
        segments_per_turn=10,
        min_segments=12,
        section_degree=1,
        radial_order=3,
        angular_order=16,
        line_order=2,
    )


def command_self_check(_args) -> int:
    scene = _demo_scene()
    config = _reference_config()
    artifact = (
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        )
    )
    system = MeshfreeVNextSystem(
        artifact,
        reference_config=config,
    )

    fast = system.fast_ports(
        scene,
        85_000.0,
    )
    reference = system.reference_ports(
        scene,
        85_000.0,
    )
    discrete = certify_mixed_ports(
        scene,
        20_000.0,
        artifact,
        config=config,
        algebraic_tolerance=1e-8,
        correction_rtol=1e-9,
        allow_reference_fallback=False,
        operator_backend="dense",
    )
    matrix_free = certify_mixed_ports(
        scene,
        20_000.0,
        artifact,
        config=config,
        algebraic_tolerance=1e-7,
        correction_rtol=1e-9,
        correction_restart=30,
        correction_maxiter=160,
        allow_reference_fallback=False,
        operator_backend="matrix_free",
        matrix_free_chunk_size=64,
    )
    backend_relative_difference = float(
        np.linalg.norm(
            matrix_free.impedance
            - discrete.impedance
        )
        / max(
            np.linalg.norm(
                discrete.impedance
            ),
            1e-30,
        )
    )

    thermal = (
        build_lumped_coil_thermal_model(
            (
                CoilThermalProperties(
                    10.0,
                    0.14,
                ),
                CoilThermalProperties(
                    8.0,
                    0.12,
                ),
            ),
            mutual_conductance=np.asarray(
                [
                    [
                        0.0,
                        0.025,
                    ],
                    [
                        0.025,
                        0.0,
                    ],
                ]
            ),
        )
    )
    thermal_step = (
        system.fast_current_envelope(
            scene,
            85_000.0,
            thermal,
            coupling_tolerance=1e-8,
            max_coupling_iterations=20,
        ).step(
            np.zeros(
                2
            ),
            np.asarray(
                [
                    3.0
                    + 0j,
                    -1.0
                    + 0.2j,
                ]
            ),
            10.0,
        )
    )

    payload = {
        "fast_power_closure": (
            fast.power_closure_error()
        ),
        "reference_power_closure": (
            reference.power_closure_error()
        ),
        "certified_status": (
            discrete.status
        ),
        "certified_final_residual": (
            discrete.final_residual
        ),
        "matrix_free_status": (
            matrix_free.status
        ),
        "matrix_free_final_residual": (
            matrix_free.final_residual
        ),
        "matrix_free_dense_relative_difference": (
            backend_relative_difference
        ),
        "thermal_converged": (
            thermal_step.converged
        ),
        "temperatures_K": (
            thermal_step.temperatures.tolist()
        ),
    }
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )
    ok = bool(
        fast.power_closure_error()
        < 1e-10
        and reference.power_closure_error()
        < 1e-7
        and discrete.algebraic_certified
        and matrix_free.algebraic_certified
        and backend_relative_difference
        < 1e-5
        and thermal_step.converged
    )
    return (
        0
        if ok
        else 2
    )


def command_dataset_generate(
    args,
) -> int:
    output = Path(
        args.output
    )
    if (
        output
        / "manifest.json"
    ).exists():
        dataset = (
            ImmutableTeacherDataset(
                output
            )
        )
    else:
        dataset = (
            ImmutableTeacherDataset.create(
                output,
                split_seed=args.seed,
            )
        )
    rng = np.random.default_rng(
        args.seed
    )
    config = _reference_config()
    for index in range(
        args.count
    ):
        scene, frequency = (
            sample_two_coil_mvp_scene(
                rng
            )
        )
        record = (
            dataset.generate_and_add(
                scene,
                frequency,
                teacher_config=config,
                baseline_segments=(
                    args.baseline_segments
                ),
                reference_backend="mixed",
                source="initial",
            )
        )
        print(
            f"[{index + 1}/{args.count}] "
            f"{record.sample_id[:12]} "
            f"{record.split} "
            f"{frequency / 1e3:.2f} kHz"
        )
    print(
        json.dumps(
            dataset.counts(),
            sort_keys=True,
        )
    )
    return 0


def command_dataset_migrate(
    args,
) -> int:
    print(
        json.dumps(
            migrate_dataset_v3_to_v4(
                args.dataset
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_neural_artifact(
    path,
    device,
):
    from .neural import (
        NeuralResidualArtifact,
    )

    return NeuralResidualArtifact.load(
        path,
        device=device,
    )


def command_train_port(
    args,
) -> int:
    from .neural import (
        train_residual_surrogate,
    )

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    train = tuple(
        dataset.iter_samples(
            "train"
        )
    )
    validation = tuple(
        dataset.iter_samples(
            "validation"
        )
    )
    if (
        not train
        or not validation
    ):
        raise SystemExit(
            "train-port requires non-empty train and validation splits"
        )
    artifact, report = (
        train_residual_surrogate(
            train,
            validation_samples=validation,
            hidden_dim=args.hidden,
            factor_rank=(
                args.factor_rank
            ),
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )
    )
    artifact.save(
        args.output
    )
    print(
        json.dumps(
            {
                "epochs_run": (
                    report.epochs
                ),
                "best_epoch": (
                    report.best_epoch
                ),
                "best_validation_score": (
                    report.best_validation_score
                ),
                "best_validation_z_error": (
                    report.best_validation_z_error
                ),
                "best_validation_channel_error": (
                    report.best_validation_channel_error
                ),
                "stopped_early": (
                    report.stopped_early
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_train_spatial(
    args,
) -> int:
    from .spatial_neural import (
        train_spatial_loss_surrogate,
    )

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    train = tuple(
        dataset.iter_samples(
            "train"
        )
    )
    validation = tuple(
        dataset.iter_samples(
            "validation"
        )
    )
    if (
        not train
        or not validation
    ):
        raise SystemExit(
            "train-spatial requires non-empty train and validation splits"
        )
    port = _load_neural_artifact(
        args.port_artifact,
        args.device,
    )
    spatial, report = (
        train_spatial_loss_surrogate(
            port,
            train,
            validation_samples=(
                validation
            ),
            field_hidden_dim=(
                args.hidden
            ),
            factor_rank=(
                args.factor_rank
            ),
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )
    )
    spatial.save(
        args.output
    )
    print(
        json.dumps(
            {
                "epochs_run": (
                    report.epochs
                ),
                "best_epoch": (
                    report.best_epoch
                ),
                "best_validation_error": (
                    report.best_validation_error
                ),
                "stopped_early": (
                    report.stopped_early
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_audit_port(
    args,
) -> int:
    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    samples = tuple(
        dataset.iter_samples(
            args.split
        )
    )
    if not samples:
        raise SystemExit(
            f"dataset contains no {args.split} samples"
        )
    artifact = _load_neural_artifact(
        args.artifact,
        args.device,
    )
    report = audit_surrogate(
        artifact,
        samples,
        mean_relative_error_limit=(
            args.mean_limit
        ),
        maximum_relative_error_limit=(
            args.max_limit
        ),
    )
    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if report.passed
        else 2
    )


def command_audit_spatial(
    args,
) -> int:
    from .spatial_neural import (
        NeuralSpatialLossArtifact,
    )

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    samples = tuple(
        dataset.iter_samples(
            args.split
        )
    )
    if not samples:
        raise SystemExit(
            f"dataset contains no {args.split} samples"
        )
    port = _load_neural_artifact(
        args.port_artifact,
        args.device,
    )
    spatial = (
        NeuralSpatialLossArtifact.load(
            args.spatial_artifact,
            port,
            device=args.device,
        )
    )
    report = audit_spatial_surrogate(
        spatial,
        samples,
        mean_relative_error_limit=(
            args.mean_limit
        ),
        maximum_relative_error_limit=(
            args.max_limit
        ),
        maximum_probe_joule_error_limit=(
            args.joule_limit
        ),
    )
    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if report.passed
        else 2
    )


def command_active_learn(
    args,
) -> int:
    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    artifacts = tuple(
        _load_neural_artifact(
            path,
            args.device,
        )
        for path in args.artifacts
    )
    result = run_active_learning_round(
        dataset,
        artifacts,
        candidate_count=(
            args.candidates
        ),
        select_count=(
            args.select
        ),
        seed=args.seed,
        teacher_config=(
            _reference_config()
        ),
        baseline_segments=(
            args.baseline_segments
        ),
    )
    print(
        json.dumps(
            {
                "candidates_evaluated": (
                    result.candidates_evaluated
                ),
                "selected": len(
                    result.selected
                ),
                "dataset_counts": (
                    dataset.counts()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_calibrate(
    args,
) -> int:
    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    validation = tuple(
        dataset.iter_samples(
            "validation"
        )
    )
    if not validation:
        raise SystemExit(
            "dataset contains no validation samples"
        )
    artifacts = tuple(
        _load_neural_artifact(
            path,
            args.device,
        )
        for path in args.artifacts
    )
    calibrator = (
        fit_fast_error_calibrator(
            artifacts,
            validation,
            quantile=args.quantile,
            baseline_segments=(
                args.baseline_segments
            ),
        )
    )
    calibrator.save(
        args.output
    )
    print(
        json.dumps(
            {
                "quantile": (
                    calibrator.quantile
                ),
                "scale": (
                    calibrator.scale
                ),
                "indicator_floor": (
                    calibrator.indicator_floor
                ),
                "ensemble_size": (
                    calibrator.ensemble_size
                ),
                "validation_samples": (
                    calibrator.validation_samples
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_bundle_publish(
    args,
) -> int:
    from .spatial_neural import (
        NeuralSpatialLossArtifact,
    )

    port = _load_neural_artifact(
        args.port_artifact,
        args.device,
    )
    spatial = None
    if (
        args.spatial_artifact
        is not None
    ):
        spatial = (
            NeuralSpatialLossArtifact.load(
                args.spatial_artifact,
                port,
                device=args.device,
            )
        )
    calibrator = None
    if (
        args.calibrator
        is not None
    ):
        calibrator = (
            FastErrorCalibrator.load(
                args.calibrator
            )
        )
    manifest = publish_bundle(
        args.output,
        port,
        spatial_artifact=(
            spatial
        ),
        calibrator=(
            calibrator
        ),
        overwrite=(
            args.overwrite
        ),
    )
    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_release(
    args,
) -> int:
    from .spatial_neural import (
        NeuralSpatialLossArtifact,
    )

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    release_samples = tuple(
        dataset.iter_samples(
            "release"
        )
    )
    if not release_samples:
        raise SystemExit(
            "release requires a non-empty locked release split"
        )

    port = _load_neural_artifact(
        args.port_artifact,
        args.device,
    )
    spatial = (
        NeuralSpatialLossArtifact.load(
            args.spatial_artifact,
            port,
            device=args.device,
        )
    )

    port_report = audit_surrogate(
        port,
        release_samples,
        mean_relative_error_limit=(
            args.port_mean_limit
        ),
        maximum_relative_error_limit=(
            args.port_max_limit
        ),
    )
    spatial_report = (
        audit_spatial_surrogate(
            spatial,
            release_samples,
            mean_relative_error_limit=(
                args.spatial_mean_limit
            ),
            maximum_relative_error_limit=(
                args.spatial_max_limit
            ),
            maximum_probe_joule_error_limit=(
                args.joule_limit
            ),
        )
    )

    coarse_config = MQSConfig(
        segments_per_turn=(
            args.certified_coarse_segments
        ),
        min_segments=(
            args.certified_coarse_segments
        ),
        section_degree=1,
        radial_order=3,
        angular_order=16,
        line_order=2,
    )
    fine_config = MQSConfig(
        segments_per_turn=(
            args.certified_fine_segments
        ),
        min_segments=(
            args.certified_fine_segments
        ),
        section_degree=1,
        radial_order=3,
        angular_order=16,
        line_order=2,
    )
    certified_report = audit_certified_release(
        port,
        release_samples,
        coarse_config=coarse_config,
        fine_config=fine_config,
        convergence_tolerance=(
            args.certified_convergence_limit
        ),
        algebraic_tolerance=(
            args.certified_algebraic_limit
        ),
        correction_rtol=(
            args.certified_correction_rtol
        ),
        correction_maxiter=(
            args.certified_correction_maxiter
        ),
        fast_domain_correction_limit=(
            args.certified_fast_correction_limit
        ),
        operator_backend=(
            args.certified_backend
        ),
        matrix_free_chunk_size=(
            args.certified_chunk_size
        ),
    )

    gate = {
        "port": (
            port_report.to_dict()
        ),
        "spatial": (
            spatial_report.to_dict()
        ),
        "certified": (
            certified_report.to_dict()
        ),
    }
    if (
        not port_report.passed
        or not spatial_report.passed
        or not certified_report.passed
    ):
        print(
            json.dumps(
                {
                    "released": False,
                    "gate": gate,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    calibrator = None
    if (
        args.calibrator
        is not None
    ):
        calibrator = (
            FastErrorCalibrator.load(
                args.calibrator
            )
        )

    manifest_bytes = (
        dataset.manifest_path.read_bytes()
    )
    dataset_manifest_sha256 = (
        sha256(
            manifest_bytes
        ).hexdigest()
    )
    manifest = publish_bundle(
        args.output,
        port,
        spatial_artifact=spatial,
        calibrator=calibrator,
        metadata={
            "release_gate": gate,
            "release_split": "release",
            "release_samples": len(
                release_samples
            ),
            "dataset_manifest_sha256": (
                dataset_manifest_sha256
            ),
        },
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "released": True,
                "bundle": manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sdfmpneo-vnext",
        description="SDF-MPNEO vNext mesh-free electrothermal runtime",
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    self_check = sub.add_parser(
        "self-check"
    )
    self_check.set_defaults(
        handler=command_self_check
    )

    generate = sub.add_parser(
        "dataset-generate"
    )
    generate.add_argument(
        "output",
        type=Path,
    )
    generate.add_argument(
        "--count",
        type=int,
        default=32,
    )
    generate.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    generate.add_argument(
        "--baseline-segments",
        type=int,
        default=64,
    )
    generate.set_defaults(
        handler=command_dataset_generate
    )

    migrate = sub.add_parser(
        "dataset-migrate"
    )
    migrate.add_argument(
        "dataset",
        type=Path,
    )
    migrate.set_defaults(
        handler=command_dataset_migrate
    )

    train_port = sub.add_parser(
        "train-port"
    )
    train_port.add_argument(
        "dataset",
        type=Path,
    )
    train_port.add_argument(
        "output",
        type=Path,
    )
    train_port.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    train_port.add_argument(
        "--hidden",
        type=int,
        default=64,
    )
    train_port.add_argument(
        "--factor-rank",
        type=int,
        default=4,
    )
    train_port.add_argument(
        "--patience",
        type=int,
        default=30,
    )
    train_port.add_argument(
        "--device",
        default="cpu",
    )
    train_port.set_defaults(
        handler=command_train_port
    )

    train_spatial = sub.add_parser(
        "train-spatial"
    )
    train_spatial.add_argument(
        "dataset",
        type=Path,
    )
    train_spatial.add_argument(
        "port_artifact",
        type=Path,
    )
    train_spatial.add_argument(
        "output",
        type=Path,
    )
    train_spatial.add_argument(
        "--epochs",
        type=int,
        default=120,
    )
    train_spatial.add_argument(
        "--hidden",
        type=int,
        default=64,
    )
    train_spatial.add_argument(
        "--factor-rank",
        type=int,
        default=4,
    )
    train_spatial.add_argument(
        "--patience",
        type=int,
        default=20,
    )
    train_spatial.add_argument(
        "--device",
        default="cpu",
    )
    train_spatial.set_defaults(
        handler=command_train_spatial
    )

    for name, handler in (
        (
            "audit-port",
            command_audit_port,
        ),
        (
            "audit-spatial",
            command_audit_spatial,
        ),
    ):
        audit = sub.add_parser(
            name
        )
        audit.add_argument(
            "dataset",
            type=Path,
        )
        if (
            name
            == "audit-port"
        ):
            audit.add_argument(
                "artifact",
                type=Path,
            )
        else:
            audit.add_argument(
                "port_artifact",
                type=Path,
            )
            audit.add_argument(
                "spatial_artifact",
                type=Path,
            )
        audit.add_argument(
            "--split",
            choices=(
                "test",
                "release",
            ),
            default="release",
        )
        audit.add_argument(
            "--mean-limit",
            type=float,
            default=(
                0.02
                if name
                == "audit-port"
                else 0.10
            ),
        )
        audit.add_argument(
            "--max-limit",
            type=float,
            default=(
                0.05
                if name
                == "audit-port"
                else 0.20
            ),
        )
        if (
            name
            == "audit-spatial"
        ):
            audit.add_argument(
                "--joule-limit",
                type=float,
                default=0.20,
            )
        audit.add_argument(
            "--device",
            default="cpu",
        )
        audit.set_defaults(
            handler=handler
        )

    active = sub.add_parser(
        "active-learn"
    )
    active.add_argument(
        "dataset",
        type=Path,
    )
    active.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
    )
    active.add_argument(
        "--candidates",
        type=int,
        default=128,
    )
    active.add_argument(
        "--select",
        type=int,
        default=8,
    )
    active.add_argument(
        "--seed",
        type=int,
        default=101,
    )
    active.add_argument(
        "--baseline-segments",
        type=int,
        default=64,
    )
    active.add_argument(
        "--device",
        default="cpu",
    )
    active.set_defaults(
        handler=command_active_learn
    )

    calibrate = sub.add_parser(
        "calibrate"
    )
    calibrate.add_argument(
        "dataset",
        type=Path,
    )
    calibrate.add_argument(
        "output",
        type=Path,
    )
    calibrate.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
    )
    calibrate.add_argument(
        "--quantile",
        type=float,
        default=0.95,
    )
    calibrate.add_argument(
        "--baseline-segments",
        type=int,
        default=64,
    )
    calibrate.add_argument(
        "--device",
        default="cpu",
    )
    calibrate.set_defaults(
        handler=command_calibrate
    )

    publish = sub.add_parser(
        "bundle-publish"
    )
    publish.add_argument(
        "port_artifact",
        type=Path,
    )
    publish.add_argument(
        "output",
        type=Path,
    )
    publish.add_argument(
        "--spatial-artifact",
        type=Path,
    )
    publish.add_argument(
        "--calibrator",
        type=Path,
    )
    publish.add_argument(
        "--device",
        default="cpu",
    )
    publish.add_argument(
        "--overwrite",
        action="store_true",
    )
    publish.set_defaults(
        handler=command_bundle_publish
    )

    release = sub.add_parser(
        "release"
    )
    release.add_argument(
        "dataset",
        type=Path,
    )
    release.add_argument(
        "port_artifact",
        type=Path,
    )
    release.add_argument(
        "spatial_artifact",
        type=Path,
    )
    release.add_argument(
        "output",
        type=Path,
    )
    release.add_argument(
        "--calibrator",
        type=Path,
    )
    release.add_argument(
        "--port-mean-limit",
        type=float,
        default=0.02,
    )
    release.add_argument(
        "--port-max-limit",
        type=float,
        default=0.05,
    )
    release.add_argument(
        "--spatial-mean-limit",
        type=float,
        default=0.05,
    )
    release.add_argument(
        "--spatial-max-limit",
        type=float,
        default=0.10,
    )
    release.add_argument(
        "--joule-limit",
        type=float,
        default=0.10,
    )
    release.add_argument(
        "--certified-coarse-segments",
        type=int,
        default=8,
    )
    release.add_argument(
        "--certified-fine-segments",
        type=int,
        default=12,
    )
    release.add_argument(
        "--certified-convergence-limit",
        type=float,
        default=0.02,
    )
    release.add_argument(
        "--certified-algebraic-limit",
        type=float,
        default=1e-7,
    )
    release.add_argument(
        "--certified-correction-rtol",
        type=float,
        default=1e-9,
    )
    release.add_argument(
        "--certified-correction-maxiter",
        type=int,
        default=160,
    )
    release.add_argument(
        "--certified-fast-correction-limit",
        type=float,
        default=0.20,
    )
    release.add_argument(
        "--certified-backend",
        choices=("dense", "matrix_free"),
        default="matrix_free",
    )
    release.add_argument(
        "--certified-chunk-size",
        type=int,
        default=256,
    )
    release.add_argument(
        "--device",
        default="cpu",
    )
    release.add_argument(
        "--overwrite",
        action="store_true",
    )
    release.set_defaults(
        handler=command_release
    )
    return parser


def main(
    argv=None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(
        argv
    )
    return int(
        args.handler(
            args
        )
    )
