from __future__ import annotations

import numpy as np

from .convergence import mixed_reference_convergence
from .em import MQSConfig
from .mixed import DenseMixedConductorTeacher
from .prediction import StructuredPortPrediction
from .reference import PreparedReferenceLossField
from .serialization import scene_from_dict
from .thermal_field import (
    HomogeneousThermalMedium,
    PreparedThermalGreenField,
    build_thermal_source_quadrature,
)


def _complex_vector(
    value,
    *,
    length: int,
):
    array = np.asarray(
        value,
        dtype=float,
    )
    if array.shape == (
        length,
    ):
        return array.astype(
            complex
        )
    if array.shape != (
        length,
        2,
    ):
        raise ValueError(
            "complex vector must be [real,imag] pairs"
        )
    return (
        array[:, 0]
        + 1j
        * array[:, 1]
    )


def _complex_json(
    value,
):
    array = np.asarray(
        value,
        dtype=complex,
    )
    stacked = np.stack(
        (
            array.real,
            array.imag,
        ),
        axis=-1,
    )
    return stacked.tolist()


def _mqs_config(
    segments: int,
):
    if segments < 2:
        raise ValueError(
            "segments must be >= 2"
        )
    return MQSConfig(
        segments_per_turn=int(
            segments
        ),
        min_segments=int(
            segments
        ),
        section_degree=1,
        radial_order=3,
        angular_order=16,
        line_order=2,
    )


def _thermal_medium(
    data,
):
    if data is None:
        raise ValueError(
            "thermal query requires a medium"
        )
    return HomogeneousThermalMedium(
        conductivity=float(
            data[
                "conductivity"
            ]
        ),
        density=float(
            data[
                "density"
            ]
        ),
        heat_capacity=float(
            data[
                "heat_capacity"
            ]
        ),
        ambient_temperature=float(
            data.get(
                "ambient_temperature",
                293.15,
            )
        ),
    )


def _certified_prediction(
    system,
    scene,
    frequency_hz: float,
    options,
):
    options = (
        {}
        if options is None
        else dict(
            options
        )
    )
    coarse_segments = int(
        options.get(
            "coarse_segments",
            8,
        )
    )
    fine_segments = int(
        options.get(
            "fine_segments",
            12,
        )
    )
    convergence_tolerance = float(
        options.get(
            "convergence_tolerance",
            0.02,
        )
    )
    fine = _mqs_config(
        fine_segments
    )
    if (
        fine_segments
        < coarse_segments
    ):
        raise ValueError(
            "fine_segments must be >= coarse_segments"
        )
    convergence = (
        mixed_reference_convergence(
            scene,
            frequency_hz,
            fine,
            tolerance=(
                convergence_tolerance
            ),
        )
    )
    result = system.certified_ports(
        scene,
        frequency_hz,
        convergence_report=(
            convergence
        ),
        config=fine,
        algebraic_tolerance=float(
            options.get(
                "algebraic_tolerance",
                1e-7,
            )
        ),
        correction_rtol=float(
            options.get(
                "correction_rtol",
                1e-9,
            )
        ),
        correction_restart=int(
            options.get(
                "correction_restart",
                40,
            )
        ),
        correction_maxiter=int(
            options.get(
                "correction_maxiter",
                160,
            )
        ),
        allow_reference_fallback=bool(
            options.get(
                "allow_reference_fallback",
                False,
            )
        ),
        operator_backend=str(
            options.get(
                "operator_backend",
                "matrix_free",
            )
        ),
        matrix_free_chunk_size=int(
            options.get(
                "matrix_free_chunk_size",
                256,
            )
        ),
        fast_domain_correction_limit=float(
            options.get(
                "fast_domain_correction_limit",
                0.20,
            )
        ),
    )
    prediction = StructuredPortPrediction(
        result.impedance,
        result.result.coil_dissipation_matrices(),
    )
    teacher = DenseMixedConductorTeacher(
        scene,
        frequency_hz,
        fine,
    )
    spatial = PreparedReferenceLossField(
        scene,
        float(
            frequency_hz
        ),
        teacher,
        result.result,
        prediction,
    )
    return (
        prediction,
        spatial,
        result,
        convergence,
    )


def run_system_inference(
    system,
    request,
):
    if not isinstance(
        request,
        dict,
    ):
        raise TypeError(
            "inference request must be a dictionary"
        )
    if "scene" not in request:
        raise ValueError(
            "inference request is missing scene"
        )
    scene = scene_from_dict(
        request[
            "scene"
        ]
    )
    frequency_hz = float(
        request[
            "frequency_hz"
        ]
    )
    mode = str(
        request.get(
            "mode",
            "fast",
        )
    ).lower()
    if mode not in (
        "fast",
        "reference",
        "certified",
    ):
        raise ValueError(
            "mode must be fast, reference, or certified"
        )

    certified = None
    convergence = None
    if mode == "fast":
        prediction = system.fast_ports(
            scene,
            frequency_hz,
        )
        spatial = None
    elif mode == "reference":
        prediction = (
            system.reference_ports(
                scene,
                frequency_hz,
            )
        )
        spatial = None
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
            request.get(
                "certified"
            ),
        )

    currents = None
    if "currents" in request:
        currents = _complex_vector(
            request[
                "currents"
            ],
            length=len(
                scene.coils
            ),
        )

    output = {
        "mode": mode,
        "frequency_hz": (
            frequency_hz
        ),
        "n_ports": len(
            scene.coils
        ),
        "impedance": _complex_json(
            prediction.impedance
        ),
        "power_closure_error": float(
            prediction.power_closure_error()
        ),
        "reciprocity_defect": float(
            prediction.reciprocity_defect()
        ),
    }
    if currents is not None:
        output[
            "currents"
        ] = _complex_json(
            currents
        )
        output[
            "coil_power"
        ] = (
            prediction.coil_power(
                currents
            ).tolist()
        )
        output[
            "port_power"
        ] = float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    prediction.impedance
                    @ currents,
                )
            )
        )

    queries = request.get(
        "spatial_queries",
        ()
    )
    thermal_request = request.get(
        "thermal"
    )
    if (
        queries
        or thermal_request
        is not None
    ):
        if spatial is None:
            spatial = (
                system.fast_spatial(
                    scene,
                    frequency_hz,
                )
                if mode == "fast"
                else system.reference_spatial(
                    scene,
                    frequency_hz,
                )
            )

    if queries:
        spatial_output = []
        for query in queries:
            matrix = (
                spatial.local_dissipation_matrix(
                    int(
                        query[
                            "coil_index"
                        ]
                    ),
                    float(
                        query[
                            "arc_fraction"
                        ]
                    ),
                    np.asarray(
                        query.get(
                            "xy",
                            (
                                0.0,
                                0.0,
                            ),
                        ),
                        dtype=float,
                    ),
                )
            )
            item = {
                "coil_index": int(
                    query[
                        "coil_index"
                    ]
                ),
                "arc_fraction": float(
                    query[
                        "arc_fraction"
                    ]
                ),
                "xy": list(
                    query.get(
                        "xy",
                        (
                            0.0,
                            0.0,
                        ),
                    )
                ),
                "dissipation_matrix": (
                    _complex_json(
                        matrix
                    )
                ),
            }
            if currents is not None:
                item[
                    "joule_density"
                ] = float(
                    0.5
                    * np.real(
                        np.vdot(
                            currents,
                            matrix
                            @ currents,
                        )
                    )
                )
            spatial_output.append(
                item
            )
        output[
            "spatial_queries"
        ] = spatial_output

    if thermal_request is not None:
        if currents is None:
            raise ValueError(
                "thermal query requires currents"
            )
        medium = _thermal_medium(
            thermal_request.get(
                "medium"
            )
        )
        thermal_options = dict(
            thermal_request.get(
                "quadrature",
                {}
            )
        )
        if mode == "fast":
            thermal = (
                system.fast_continuous_thermal_field(
                    scene,
                    frequency_hz,
                    medium,
                    **thermal_options,
                )
            )
        elif mode == "reference":
            thermal = (
                system.reference_continuous_thermal_field(
                    scene,
                    frequency_hz,
                    medium,
                    **thermal_options,
                )
            )
        else:
            source = (
                build_thermal_source_quadrature(
                    scene,
                    spatial,
                    **thermal_options,
                )
            )
            thermal = (
                PreparedThermalGreenField(
                    source,
                    medium,
                )
            )
        points = np.asarray(
            thermal_request[
                "points"
            ],
            dtype=float,
        )
        time = float(
            thermal_request[
                "time"
            ]
        )
        temperature = (
            thermal.temperature_step(
                points,
                time,
                currents,
            )
        )
        output[
            "thermal"
        ] = {
            "time": time,
            "points": points.tolist(),
            "temperature": (
                np.asarray(
                    temperature,
                    dtype=float,
                ).tolist()
            ),
            "ambient_temperature": (
                medium.ambient_temperature
            ),
            "source_normalization_closure_error": float(
                thermal.source.normalization_closure_error
            ),
        }

    if certified is not None:
        output[
            "certification"
        ] = {
            "status": (
                certified.status
            ),
            "certified": bool(
                certified.certified
            ),
            "algebraic_certified": bool(
                certified.algebraic_certified
            ),
            "discretization_certified": bool(
                certified.discretization_certified
            ),
            "fast_domain_valid": bool(
                certified.fast_domain_valid
            ),
            "initial_residual": float(
                certified.initial_residual
            ),
            "final_residual": float(
                certified.final_residual
            ),
            "relative_observable_correction": float(
                certified.relative_observable_correction
            ),
            "operator_backend": (
                certified.operator_backend
            ),
            "correction_iterations": list(
                certified.correction_iterations
            ),
            "discretization_change": float(
                convergence.maximum_relative_change
            ),
            "discretization_directions": {
                direction.name: {
                    "maximum_relative_change": float(
                        direction.maximum_relative_change
                    ),
                    "impedance_relative_change": float(
                        direction.impedance_relative_change
                    ),
                    "channel_relative_change": float(
                        direction.channel_relative_change
                    ),
                    "local_loss_relative_change": float(
                        direction.local_loss_relative_change
                    ),
                }
                for direction
                in convergence.directions
            },
        }
    return output
