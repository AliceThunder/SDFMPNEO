from __future__ import annotations

import numpy as np

from .convergence import mixed_reference_convergence
from .em import MQSConfig
from .hybrid_convergence import hybrid_reference_convergence
from .prediction import StructuredPortPrediction
from .scene import TensorElectricMaterial
from .serialization import scene_from_dict
from .thermal_field import (
    AnisotropicThermalMedium,
    HomogeneousThermalMedium,
)


def _complex_vector(value, *, length: int):
    array = np.asarray(value, dtype=float)
    if array.shape == (length,):
        return array.astype(complex)
    if array.shape != (length, 2):
        raise ValueError("complex vector must be [real,imag] pairs")
    return array[:, 0] + 1j * array[:, 1]


def _complex_matrix(value, *, columns: int):
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape[1] == columns:
        return array.astype(complex)
    if array.ndim != 3 or array.shape[1:] != (columns, 2):
        raise ValueError(
            "complex matrix must be rows of real values or [real,imag] pairs"
        )
    return array[:, :, 0] + 1j * array[:, :, 1]


def _complex_json(value):
    array = np.asarray(value, dtype=complex)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def _mqs_config(segments: int):
    if segments < 2:
        raise ValueError("segments must be >= 2")
    return MQSConfig(
        segments_per_turn=int(segments),
        min_segments=int(segments),
        section_degree=1,
        radial_order=3,
        angular_order=16,
        line_order=2,
    )


def _thermal_medium(data):
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError("thermal medium must be a dictionary")
    density = float(data["density"])
    heat_capacity = float(data["heat_capacity"])
    ambient = float(data.get("ambient_temperature", 293.15))
    has_scalar = "conductivity" in data
    has_tensor = "conductivity_tensor" in data
    if has_scalar == has_tensor:
        raise ValueError(
            "thermal medium requires exactly one of conductivity or "
            "conductivity_tensor"
        )
    if has_tensor:
        return AnisotropicThermalMedium(
            conductivity_tensor=np.asarray(
                data["conductivity_tensor"],
                dtype=float,
            ),
            density=density,
            heat_capacity=heat_capacity,
            ambient_temperature=ambient,
        )
    return HomogeneousThermalMedium(
        conductivity=float(data["conductivity"]),
        density=density,
        heat_capacity=heat_capacity,
        ambient_temperature=ambient,
    )


def _uses_heterogeneous_reference(scene) -> bool:
    return bool(
        scene.packages
        or isinstance(scene.medium, TensorElectricMaterial)
    )


def _certified_prediction(
    system,
    scene,
    frequency_hz: float,
    options,
):
    """Certify through the same scene-aware backend used by the system.

    Conductor-only scenes retain matrix-free correction options. Package and
    tensor-electric scenes use the dense electric/magnetic interface truth and
    hybrid discretization refinement. Tensor-background-only scenes can still
    receive an algebraic/physical certificate; until a background-only tensor
    refinement report exists they are explicitly marked discrete-uncertified.
    """
    options = {} if options is None else dict(options)
    coarse_segments = int(options.get("coarse_segments", 8))
    fine_segments = int(options.get("fine_segments", 12))
    if fine_segments < coarse_segments:
        raise ValueError("fine_segments must be >= coarse_segments")
    convergence_tolerance = float(
        options.get("convergence_tolerance", 0.02)
    )
    fine = _mqs_config(fine_segments)

    heterogeneous = _uses_heterogeneous_reference(scene)
    convergence = None
    if heterogeneous and scene.packages:
        convergence = hybrid_reference_convergence(
            scene,
            frequency_hz,
            fine,
            surface_vertical_order=int(
                getattr(system, "dielectric_surface_vertical_order", 16)
            ),
            surface_azimuthal_order=int(
                getattr(system, "dielectric_surface_azimuthal_order", 32)
            ),
            magnetic_volume_axial_order=int(
                options.get("magnetic_volume_axial_order", 8)
            ),
            magnetic_volume_radial_order=int(
                options.get("magnetic_volume_radial_order", 6)
            ),
            magnetic_volume_azimuthal_order=int(
                options.get("magnetic_volume_azimuthal_order", 24)
            ),
            tolerance=convergence_tolerance,
            surface_residual_tolerance=float(
                options.get("surface_tolerance", 1e-9)
            ),
            magnetic_surface_residual_tolerance=float(
                options.get("magnetic_surface_tolerance", 1e-9)
            ),
        )
    elif not heterogeneous:
        convergence = mixed_reference_convergence(
            scene,
            frequency_hz,
            fine,
            tolerance=convergence_tolerance,
        )

    if heterogeneous:
        certified = system.certified_ports(
            scene,
            frequency_hz,
            convergence_report=convergence,
            config=fine,
            algebraic_tolerance=float(
                options.get("algebraic_tolerance", 1e-7)
            ),
            surface_tolerance=float(
                options.get("surface_tolerance", 1e-9)
            ),
            magnetic_reciprocity_tolerance=float(
                options.get("magnetic_reciprocity_tolerance", 0.08)
            ),
            power_tolerance=float(
                options.get("power_tolerance", 1e-6)
            ),
            fast_domain_correction_limit=float(
                options.get("fast_domain_correction_limit", 0.20)
            ),
        )
        reference_prediction = certified.result.prediction
        prediction = StructuredPortPrediction(
            certified.impedance,
            reference_prediction.dissipation_channels,
            reference_prediction.channel_names,
        )
    else:
        certified = system.certified_ports(
            scene,
            frequency_hz,
            convergence_report=convergence,
            config=fine,
            algebraic_tolerance=float(
                options.get("algebraic_tolerance", 1e-7)
            ),
            correction_rtol=float(
                options.get("correction_rtol", 1e-9)
            ),
            correction_restart=int(
                options.get("correction_restart", 40)
            ),
            correction_maxiter=int(
                options.get("correction_maxiter", 160)
            ),
            allow_reference_fallback=bool(
                options.get("allow_reference_fallback", False)
            ),
            operator_backend=str(
                options.get("operator_backend", "matrix_free")
            ),
            matrix_free_chunk_size=int(
                options.get("matrix_free_chunk_size", 256)
            ),
            fast_domain_correction_limit=float(
                options.get("fast_domain_correction_limit", 0.20)
            ),
        )
        prediction = StructuredPortPrediction(
            certified.impedance,
            certified.result.coil_dissipation_matrices(),
        )
    return prediction, None, certified, convergence


def _spatial_query(spatial, query, currents):
    matrix = spatial.local_dissipation_matrix(
        int(query["coil_index"]),
        float(query["arc_fraction"]),
        np.asarray(query.get("xy", (0.0, 0.0)), dtype=float),
    )
    item = {
        "coil_index": int(query["coil_index"]),
        "arc_fraction": float(query["arc_fraction"]),
        "xy": list(query.get("xy", (0.0, 0.0))),
        "dissipation_matrix": _complex_json(matrix),
    }
    if currents is not None:
        item["joule_density"] = float(
            0.5
            * np.real(
                np.vdot(currents, matrix @ currents)
            )
        )
    return item


def _certification_payload(certified, convergence):
    payload = {
        "status": certified.status,
        "certified": bool(certified.certified),
        "algebraic_certified": bool(certified.algebraic_certified),
        "discretization_certified": bool(
            certified.discretization_certified
        ),
        "fast_domain_valid": bool(certified.fast_domain_valid),
        "initial_residual": float(certified.initial_residual),
        "final_residual": float(certified.final_residual),
        "relative_observable_correction": float(
            certified.relative_observable_correction
        ),
        "operator_backend": certified.operator_backend,
        "correction_iterations": list(certified.correction_iterations),
    }
    if convergence is None:
        payload["discretization_change"] = None
        payload["discretization_directions"] = {}
        return payload
    payload["discretization_change"] = float(
        convergence.maximum_relative_change
    )
    directions = {}
    for direction in convergence.directions:
        item = {
            "maximum_relative_change": float(
                direction.maximum_relative_change
            ),
            "impedance_relative_change": float(
                direction.impedance_relative_change
            ),
            "channel_relative_change": float(
                direction.channel_relative_change
            ),
        }
        if hasattr(direction, "local_loss_relative_change"):
            item["local_loss_relative_change"] = float(
                direction.local_loss_relative_change
            )
        if hasattr(direction, "surface_residual"):
            item["surface_residual"] = float(direction.surface_residual)
        if hasattr(direction, "magnetic_surface_residual"):
            item["magnetic_surface_residual"] = float(
                direction.magnetic_surface_residual
            )
        directions[direction.name] = item
    payload["discretization_directions"] = directions
    return payload


def run_system_inference(system, request):
    if not isinstance(request, dict):
        raise TypeError("inference request must be a dictionary")
    if "scene" not in request:
        raise ValueError("inference request is missing scene")
    scene = scene_from_dict(request["scene"])
    frequency_hz = float(request["frequency_hz"])
    mode = str(request.get("mode", "fast")).lower()
    if mode not in ("fast", "reference", "certified"):
        raise ValueError("mode must be fast, reference, or certified")

    certified = None
    convergence = None
    spatial = None
    if mode == "fast":
        prediction = system.fast_ports(scene, frequency_hz)
    elif mode == "reference":
        prediction = system.reference_ports(scene, frequency_hz)
    else:
        (
            prediction,
            spatial,
            certified,
            convergence,
        ) = _certified_prediction(
            system,
            scene,
            frequency_hz,
            request.get("certified"),
        )

    currents = None
    if "currents" in request:
        currents = _complex_vector(
            request["currents"],
            length=len(scene.coils),
        )

    output = {
        "mode": mode,
        "frequency_hz": frequency_hz,
        "n_ports": len(scene.coils),
        "impedance": _complex_json(prediction.impedance),
        "power_closure_error": float(prediction.power_closure_error()),
        "reciprocity_defect": float(prediction.reciprocity_defect()),
    }
    if currents is not None:
        output["currents"] = _complex_json(currents)
        output["coil_power"] = prediction.coil_power(currents).tolist()
        output["port_power"] = float(
            0.5
            * np.real(
                np.vdot(currents, prediction.impedance @ currents)
            )
        )

    queries = request.get("spatial_queries", ())
    thermal_request = request.get("thermal")
    if queries or thermal_request is not None:
        if spatial is None:
            spatial = (
                system.fast_spatial(scene, frequency_hz)
                if mode == "fast"
                else system.reference_spatial(scene, frequency_hz)
            )

    if queries:
        output["spatial_queries"] = [
            _spatial_query(spatial, query, currents)
            for query in queries
        ]

    if thermal_request is not None:
        history_request = thermal_request.get("history")
        if history_request is None and currents is None:
            raise ValueError(
                "thermal step query requires top-level currents"
            )
        medium = _thermal_medium(thermal_request.get("medium"))
        thermal_options = dict(thermal_request.get("quadrature", {}))
        thermal = (
            system.fast_continuous_thermal_field(
                scene,
                frequency_hz,
                medium,
                **thermal_options,
            )
            if mode == "fast"
            else system.reference_continuous_thermal_field(
                scene,
                frequency_hz,
                medium,
                **thermal_options,
            )
        )
        points = np.asarray(thermal_request["points"], dtype=float)
        thermal_output = {
            "points": points.tolist(),
            "ambient_temperature": float(
                thermal.medium.ambient_temperature
            ),
            "source_normalization_closure_error": float(
                thermal.source.normalization_closure_error
            ),
        }
        if history_request is None:
            if "time" not in thermal_request:
                raise ValueError("thermal query requires time or history")
            time = float(thermal_request["time"])
            thermal_output["time"] = time
            thermal_output["temperature"] = np.asarray(
                thermal.temperature_step(points, time, currents),
                dtype=float,
            ).tolist()
        else:
            if not isinstance(history_request, dict):
                raise TypeError("thermal history must be a dictionary")
            interval_edges = np.asarray(
                history_request["interval_edges"],
                dtype=float,
            )
            interval_currents = _complex_matrix(
                history_request["interval_currents"],
                columns=len(scene.coils),
            )
            observation_times = np.asarray(
                history_request["observation_times"],
                dtype=float,
            )
            temperature = thermal.temperature_history(
                points,
                interval_edges,
                interval_currents,
                observation_times,
            )
            thermal_output["history"] = {
                "interval_edges": interval_edges.tolist(),
                "interval_currents": _complex_json(interval_currents),
                "observation_times": observation_times.tolist(),
                "temperature": np.asarray(
                    temperature,
                    dtype=float,
                ).tolist(),
            }
        output["thermal"] = thermal_output

    if certified is not None:
        output["certification"] = _certification_payload(
            certified,
            convergence,
        )
    return output
