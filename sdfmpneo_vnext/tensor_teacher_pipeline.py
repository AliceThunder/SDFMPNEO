from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from .analytic_baseline import analytic_port_baseline
from .em import MQSConfig
from .exterior_quadrature import conductor_volume_mask
from .hybrid_dielectric import DielectricCoupledMixedTeacher
from .hybrid_domain import package_domain_topology
from .hybrid_training_data import (
    BackgroundSpatialLossSamples,
    PackageSpatialLossSamples,
)
from .scene import Scene
from .tensor_features import encode_tensor_hybrid_scene_invariant
from .tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from .tensor_spatial_reference import prepare_tensor_spatial_reference_adaptive
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample
from .tensor_training_data import TensorHybridTeacherSample
from .training_data import SpatialLossSamples


_WORKER_THREAD_LIMIT = None


def _worker_init(native_threads: int):
    global _WORKER_THREAD_LIMIT
    if native_threads > 0:
        from threadpoolctl import threadpool_limits

        _WORKER_THREAD_LIMIT = threadpool_limits(limits=int(native_threads))


def generate_tensor_teacher_once(
    scene: Scene,
    frequency_hz: float,
    *,
    teacher_config: MQSConfig | None = None,
    baseline_segments: int = 96,
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    magnetic_volume_axial_order: int = 8,
    magnetic_volume_radial_order: int = 6,
    magnetic_volume_azimuthal_order: int = 24,
    maximum_raw_magnetic_reciprocity_defect: float = 0.08,
    include_spatial: bool = False,
    package_volume_axial_order: int = 6,
    package_volume_radial_order: int = 4,
    package_volume_azimuthal_order: int = 16,
    background_radial_order: int = 10,
    background_angular_order: int = 32,
    maximum_raw_spatial_closure_error: float = 0.35,
    maximum_spatial_quadrature_refinements: int = 4,
):
    """Produce one energy-consistent tensor port/spatial teacher sample."""
    if not scene.packages:
        raise ValueError("tensor hybrid teacher samples require at least one package")
    if (
        package_volume_axial_order < 2
        or package_volume_radial_order < 2
        or package_volume_azimuthal_order < 8
        or background_radial_order < 3
        or background_angular_order < 8
    ):
        raise ValueError("invalid tensor spatial truth quadrature resolution")
    if int(maximum_spatial_quadrature_refinements) < 0:
        raise ValueError(
            "maximum_spatial_quadrature_refinements must be nonnegative"
        )

    frequency_hz = float(frequency_hz)
    encoded = encode_tensor_hybrid_scene_invariant(scene, frequency_hz)
    baseline = analytic_port_baseline(
        Scene(scene.coils, scene.medium, ()),
        frequency_hz,
        segments_per_coil=int(baseline_segments),
    )
    config = teacher_config or MQSConfig()
    teacher = DielectricCoupledMixedTeacher(
        scene,
        frequency_hz,
        config,
        surface_vertical_order=int(surface_vertical_order),
        surface_azimuthal_order=int(surface_azimuthal_order),
        magnetic_volume_axial_order=int(magnetic_volume_axial_order),
        magnetic_volume_radial_order=int(magnetic_volume_radial_order),
        magnetic_volume_azimuthal_order=int(magnetic_volume_azimuthal_order),
        maximum_raw_magnetic_reciprocity_defect=float(
            maximum_raw_magnetic_reciprocity_defect
        ),
    )
    result = teacher.solve()
    calibration = prepare_tensor_spatial_reference_adaptive(
        teacher,
        result,
        volume_axial_order=int(package_volume_axial_order),
        volume_radial_order=int(package_volume_radial_order),
        volume_azimuthal_order=int(package_volume_azimuthal_order),
        background_radial_order=int(background_radial_order),
        background_angular_order=int(background_angular_order),
        maximum_raw_closure_error=float(maximum_raw_spatial_closure_error),
        maximum_quadrature_refinements=int(
            maximum_spatial_quadrature_refinements
        ),
    )
    prepared = calibration.prepared

    port = TensorHybridTeacherSample(
        scene=scene,
        frequency_hz=frequency_hz,
        encoded=encoded,
        baseline_resistance=baseline.resistance,
        baseline_reactance=(
            2.0 * np.pi * frequency_hz * baseline.inductance
        ),
        target_impedance=calibration.target_impedance,
        target_dissipation_channels=calibration.target_dissipation_channels,
        baseline_segments=int(baseline_segments),
        surface_vertical_order=int(surface_vertical_order),
        surface_azimuthal_order=int(surface_azimuthal_order),
        surface_residual=float(result.surface_residual),
        raw_potential_reciprocity_defect=float(
            result.raw_potential_reciprocity_defect
        ),
        power_closure_error=float(calibration.power_closure_error),
        magnetic_surface_residual=float(result.magnetic_surface_residual),
        raw_magnetic_reciprocity_defect=float(
            result.raw_magnetic_reciprocity_defect
        ),
        teacher_config=config,
        magnetic_volume_axial_order=int(magnetic_volume_axial_order),
        magnetic_volume_radial_order=int(magnetic_volume_radial_order),
        magnetic_volume_azimuthal_order=int(magnetic_volume_azimuthal_order),
        maximum_raw_magnetic_reciprocity_defect=float(
            maximum_raw_magnetic_reciprocity_defect
        ),
        package_volume_axial_order=int(package_volume_axial_order),
        package_volume_radial_order=int(package_volume_radial_order),
        package_volume_azimuthal_order=int(package_volume_azimuthal_order),
        background_radial_order=int(background_radial_order),
        background_angular_order=int(background_angular_order),
        maximum_raw_spatial_closure_error=float(
            maximum_raw_spatial_closure_error
        ),
    )
    if not include_spatial:
        return port

    coil_segments = {}
    segments = teacher.conductor_teacher._mqs._segments
    for index, segment in enumerate(segments):
        coil_segments.setdefault(int(segment.coil), []).append(index)
    local_position = {
        coil: {
            segment_index: position
            for position, segment_index in enumerate(indices)
        }
        for coil, indices in coil_segments.items()
    }
    conductor_coil = []
    conductor_arc = []
    conductor_xy = []
    conductor_weights = []
    conductor_matrix = []
    for segment_index, segment in enumerate(segments):
        coil = int(segment.coil)
        position = local_position[coil][segment_index]
        n_segments = len(coil_segments[coil])
        arc = (position + 0.5) / n_segments
        quadrature = segment.basis.quadrature
        transfer = (
            segment.basis.values
            @ result.mixed_result.current_coefficients[segment.mode_slice]
        )
        conductivity = scene.coils[coil].material.conductivity
        matrices = np.einsum(
            "qi,qj->qij",
            transfer.conj(),
            transfer,
        ) / conductivity
        matrices = prepared.transform_dissipation_matrices(matrices)
        count = len(quadrature.weights)
        conductor_coil.append(np.full(count, coil, dtype=int))
        conductor_arc.append(np.full(count, arc, dtype=float))
        conductor_xy.append(quadrature.xy)
        conductor_weights.append(quadrature.weights * segment.length)
        conductor_matrix.append(matrices)
    conductor_spatial = SpatialLossSamples(
        np.concatenate(conductor_coil),
        np.concatenate(conductor_arc),
        np.concatenate(conductor_xy, axis=0),
        np.concatenate(conductor_weights),
        np.concatenate(conductor_matrix, axis=0),
    )

    topology = package_domain_topology(scene.packages)
    package_index = []
    package_local = []
    package_weights = []
    package_matrices = []
    for index, package in enumerate(scene.packages):
        quadrature = package.geometry.volume_quadrature(
            axial_order=int(package_volume_axial_order),
            radial_order=int(package_volume_radial_order),
            azimuthal_order=int(package_volume_azimuthal_order),
        )
        region = np.asarray(
            topology.deepest_containing(
                scene.packages,
                quadrature.positions,
                tolerance=2e-12,
            ),
            dtype=int,
        )
        keep = region == index
        if np.any(keep):
            keep &= ~np.asarray(
                conductor_volume_mask(
                    scene,
                    quadrature.positions,
                    segments=segments,
                ),
                dtype=bool,
            )
        count = int(np.count_nonzero(keep))
        if count == 0:
            continue
        package_index.append(np.full(count, index, dtype=int))
        package_local.append(quadrature.local_positions[keep])
        package_weights.append(quadrature.weights[keep])
        package_matrices.append(
            prepared.package_dissipation_matrices(
                index,
                quadrature.positions[keep],
            )
        )

    n_ports = len(scene.coils)
    if package_index:
        package_spatial = PackageSpatialLossSamples(
            np.concatenate(package_index),
            np.concatenate(package_local, axis=0),
            np.concatenate(package_weights),
            np.concatenate(package_matrices, axis=0),
        )
    else:
        package_spatial = PackageSpatialLossSamples(
            np.zeros(0, dtype=int),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros((0, n_ports, n_ports), dtype=complex),
        )

    background_spatial = None
    if scene.medium.loss_conductivity(frequency_hz) > 0.0:
        points, weights = prepared.background_quadrature(
            radial_order=int(background_radial_order),
            angular_order=int(background_angular_order),
        )
        root_pose = scene.coils[0].geometry.pose
        root_local = (
            points - root_pose.translation[None, :]
        ) @ root_pose.rotation
        background_spatial = BackgroundSpatialLossSamples(
            root_local,
            weights,
            prepared.background_dissipation_matrices(points),
        )

    return TensorHybridSpatialTeacherSample(
        port=port,
        conductor_spatial_loss=conductor_spatial,
        package_spatial_loss=package_spatial,
        background_spatial_loss=background_spatial,
        package_volume_axial_order=int(package_volume_axial_order),
        package_volume_radial_order=int(package_volume_radial_order),
        package_volume_azimuthal_order=int(package_volume_azimuthal_order),
        background_radial_order=int(background_radial_order),
        background_angular_order=int(background_angular_order),
    )


def _teacher_job(payload):
    index, seed, sampler, teacher_config, options = payload
    rng = np.random.default_rng([int(seed), int(index)])
    scene, frequency_hz = sample_tensor_hybrid_scene(rng, sampler)
    return index, generate_tensor_teacher_once(
        scene,
        frequency_hz,
        teacher_config=teacher_config,
        **options,
    )


def iter_tensor_teacher_samples_parallel_once(
    count: int,
    *,
    sampler_config: TensorHybridSceneSamplerConfig | None = None,
    teacher_config: MQSConfig | None = None,
    workers: int = 1,
    native_threads_per_worker: int = 1,
    seed: int = 37,
    **options,
):
    """Deterministic local-process tensor teacher generation without duplicate solves."""
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    if native_threads_per_worker < 0:
        raise ValueError("native_threads_per_worker must be nonnegative")
    sampler = sampler_config or TensorHybridSceneSamplerConfig()
    config = teacher_config or MQSConfig()
    jobs = (
        (index, int(seed), sampler, config, dict(options))
        for index in range(int(count))
    )
    if workers == 1:
        if native_threads_per_worker > 0:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=int(native_threads_per_worker)):
                for _, sample in map(_teacher_job, jobs):
                    yield sample
        else:
            for _, sample in map(_teacher_job, jobs):
                yield sample
        return

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(int(native_threads_per_worker),),
    ) as executor:
        for _, sample in executor.map(_teacher_job, jobs, chunksize=1):
            yield sample


def recommended_teacher_workers(*, native_threads_per_worker: int = 1) -> int:
    """Conservative CPU worker count for independent correctness solves."""
    cpus = max(int(os.cpu_count() or 1), 1)
    threads = max(int(native_threads_per_worker), 1)
    return max(1, cpus // threads)
