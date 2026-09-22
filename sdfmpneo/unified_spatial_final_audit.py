"""Completely-held-out audit for spatial-Joule + online thermal ROM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .unified_corrected_truth import solve_spatial_truth_tensors
from .unified_online_thermal import (
    audit_online_thermal_trajectories,
    build_online_thermal_context,
)
from .unified_tensor_surrogate import SpatialDecodedTensors, encode_geometry
from .unified_thermal import _port_current_vectors


def _relative(value, reference, natural_scale=None):
    a = np.asarray(value)
    b = np.asarray(reference)
    floor = np.finfo(float).tiny
    if natural_scale is not None:
        floor = max(floor, 1e-10 * float(natural_scale))
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(b)), floor)
    )



def _jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {
            "real": float(value.real),
            "imag": float(value.imag),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_truth_cache(
    path,
    *,
    cache_key,
    geometries,
    n_ports,
    n_cells,
):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    expected_geometry = np.asarray(
        [encode_geometry(g) for g in geometries],
        float,
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"]) != 1:
                return None
            if str(data["cache_key"]) != str(cache_key):
                return None
            cached_geometry = np.asarray(
                data["geometry_encoding"],
                float,
            )
            if (
                cached_geometry.shape
                != expected_geometry.shape
                or not np.array_equal(
                    cached_geometry,
                    expected_geometry,
                )
            ):
                return None
            z = np.asarray(data["z_field"], complex)
            d = np.asarray(data["d_vol"], complex)
            cells = np.asarray(data["cell_h"], complex)
            d_out = np.asarray(data["d_out"], complex)
            audits_json = data["audit_json"].astype(str)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):
        return None

    count = len(geometries)
    n = int(n_ports)
    m = int(n_cells)
    if (
        z.shape != (count, n, n)
        or d.shape != (count, n, n)
        or d_out.shape != (count, n, n)
        or cells.shape != (count, m, n, n)
        or audits_json.shape != (count,)
    ):
        return None

    bundles = []
    try:
        for index in range(count):
            audit = json.loads(str(audits_json[index]))
            bundles.append(
                (
                    SpatialDecodedTensors(
                        z[index],
                        d[index],
                        cells[index],
                        d_out[index],
                        0.0,
                        0.0,
                    ),
                    audit,
                )
            )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return bundles


def _save_truth_cache(
    path,
    *,
    cache_key,
    geometries,
    bundles,
):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry_encoding = np.asarray(
        [encode_geometry(g) for g in geometries],
        float,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(1),
            cache_key=np.asarray(str(cache_key)),
            geometry_encoding=geometry_encoding,
            z_field=np.asarray(
                [bundle[0].z_field for bundle in bundles],
                complex,
            ),
            d_vol=np.asarray(
                [bundle[0].d_vol for bundle in bundles],
                complex,
            ),
            cell_h=np.asarray(
                [bundle[0].cell_h for bundle in bundles],
                complex,
            ),
            d_out=np.asarray(
                [
                    bundle[0].implied_d_out
                    for bundle in bundles
                ],
                complex,
            ),
            audit_json=np.asarray(
                [
                    json.dumps(
                        _jsonable(bundle[1]),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=True,
                    )
                    for bundle in bundles
                ]
            ),
        )
    temporary.replace(path)


def _truth_tensors(background, geometry):
    z, d, cells, d_out, audit = solve_spatial_truth_tensors(
        background,
        geometry,
        return_outward=True,
    )
    return (
        SpatialDecodedTensors(
            np.asarray(z, complex),
            np.asarray(d, complex),
            np.asarray(cells, complex),
            np.asarray(d_out, complex),
            0.0,
            0.0,
        ),
        audit,
    )


def _tensor_case(
    model,
    geometry,
    *,
    truth_projection_limit=2e-1,
    truth_bundle=None,
):
    if truth_bundle is None:
        truth, truth_audit = _truth_tensors(
            model.background,
            geometry,
        )
    else:
        truth, truth_audit = truth_bundle
    predicted = model.surrogate.predict(
        geometry,
        background=model.background,
    )
    z_scale = max(
        float(np.linalg.norm(truth.z_field)),
        np.finfo(float).tiny,
    )
    d_scale = max(
        float(np.linalg.norm(truth.d_vol)),
        z_scale * 1e-12,
    )
    cell_scale = max(
        float(np.linalg.norm(truth.cell_h)),
        d_scale * 1e-12,
    )
    row = {
        "z_relative_error": _relative(
            predicted.z_field,
            truth.z_field,
            z_scale,
        ),
        "d_relative_error": _relative(
            predicted.d_vol,
            truth.d_vol,
            d_scale,
        ),
        "spatial_joule_relative_error": _relative(
            predicted.cell_h,
            truth.cell_h,
            cell_scale,
        ),
        "outward_relative_error": _relative(
            predicted.implied_d_out,
            truth.implied_d_out,
            z_scale,
        ),
        "zd_projection_correction": float(
            predicted.zd_projection_correction
        ),
        "spatial_projection_correction": float(
            predicted.spatial_projection_correction
        ),
    }

    current_error = 0.0
    heat_error = 0.0
    power_error = 0.0
    for current in _port_current_vectors(
        truth.z_field.shape[0]
    ):
        c = np.asarray(current, complex)
        c2 = float(np.linalg.norm(c) ** 2)
        current_error = max(
            current_error,
            _relative(
                predicted.z_field @ c,
                truth.z_field @ c,
                z_scale * np.sqrt(c2),
            ),
        )
        q_truth = truth.cell_heat(c)
        q_pred = predicted.cell_heat(c)
        heat_error = max(
            heat_error,
            _relative(
                q_pred,
                q_truth,
                max(
                    float(np.linalg.norm(q_truth)),
                    d_scale * c2 * 1e-12,
                ),
            ),
        )
        p_truth = truth.volume_power(c)
        p_pred = predicted.volume_power(c)
        power_error = max(
            power_error,
            abs(p_pred - p_truth)
            / max(
                abs(p_truth),
                1e-10 * d_scale * c2,
                np.finfo(float).tiny,
            ),
        )
    row["maximum_current_space_relative_error"] = float(
        max(current_error, heat_error, power_error)
    )
    row["maximum_current_space_cell_heat_error"] = float(
        heat_error
    )
    row["maximum_current_space_power_error"] = float(
        power_error
    )
    truth_checks = {
        "linear_solve_ok": (
            float(truth_audit["max_linear_relative_residual"]) <= 1e-8
        ),
        "reciprocity_ok": (
            float(truth_audit["reciprocity_relative_error"]) <= 1e-8
        ),
        "volume_passivity_ok": (
            float(truth_audit["minimum_d_vol_eigenvalue"]) >= -1e-9
        ),
        "physical_outward_passivity_ok": (
            float(truth_audit["minimum_physical_outward_eigenvalue"])
            >= -1e-9
        ),
        "implied_outward_passivity_ok": (
            float(truth_audit["minimum_implied_outward_eigenvalue"])
            >= -1e-9
        ),
        "poynting_balance_ok": (
            float(
                truth_audit[
                    "open_boundary_power_balance_relative_error"
                ]
            )
            <= 1e-7
        ),
        "local_self_ok": (
            float(truth_audit["local_self_correction_enabled"]) >= 0.5
            and float(
                truth_audit[
                    "local_self_correction_power_balance_relative_error"
                ]
            )
            <= 1e-7
        ),
        "cell_psd_ok": (
            float(
                truth_audit["minimum_cell_joule_tensor_eigenvalue"]
            )
            >= -1e-10
        ),
        "cell_sum_to_d_ok": (
            float(
                truth_audit["maximum_spatial_joule_total_mismatch"]
            )
            <= 1e-10
        ),
        "truth_projection_ok": (
            float(
                truth_audit[
                    "spatial_truth_projection_correction"
                ]
            )
            <= float(truth_projection_limit)
        ),
    }
    row["truth_physics"] = {
        **{k: bool(v) for k, v in truth_checks.items()},
        "certified": bool(all(truth_checks.values())),
    }
    row["truth_audit"] = truth_audit
    return truth, predicted, row


def _rhs(model, context, tensors, operating, state):
    a = np.asarray(state, float).reshape(-1)
    q = model._heat_source_with_tensors(
        a,
        context,
        tensors,
        operating,
    )[0]
    return np.linalg.solve(
        context.thermal_mass_reduced,
        -context.thermal_stiffness_reduced @ a + q,
    )


def _integrate(model, context, tensors, operating, times):
    times = np.asarray(times, float)
    rank = int(context.thermal_basis.shape[1])
    result = solve_ivp(
        lambda _t, a: _rhs(
            model,
            context,
            tensors,
            operating,
            a,
        ),
        (0.0, float(times[-1])),
        np.zeros(rank, float),
        t_eval=times,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
    )
    if (
        not result.success
        or result.y.shape != (rank, len(times))
    ):
        raise RuntimeError(
            "final audit reduced reference integration failed: "
            + str(result.message)
        )
    return np.asarray(result.y.T, float)


def _mass_field_relative(diff, reference, mass):
    d = np.asarray(diff, float).reshape(-1)
    r = np.asarray(reference, float).reshape(-1)
    numerator = max(
        float(d @ (mass @ d)),
        0.0,
    )
    denominator = max(
        float(r @ (mass @ r)),
        np.finfo(float).tiny,
    )
    return float(np.sqrt(numerator / denominator))


def _numerical_jacobian(function, state, value=None):
    a = np.asarray(state, float).reshape(-1)
    base = np.asarray(
        function(a) if value is None else value,
        float,
    ).reshape(-1)
    jacobian = np.empty((base.size, a.size), float)
    epsilon = np.sqrt(np.finfo(float).eps) * (
        1.0 + np.abs(a)
    )
    for k in range(a.size):
        trial = a.copy()
        trial[k] += epsilon[k]
        jacobian[:, k] = (
            np.asarray(function(trial), float).reshape(-1)
            - base
        ) / epsilon[k]
    return jacobian


def _steady(model, context, tensors, operating, initial):
    function = lambda a: _rhs(
        model,
        context,
        tensors,
        operating,
        a,
    )
    state = np.asarray(initial, float).reshape(-1).copy()
    for _ in range(40):
        value = function(state)
        if np.linalg.norm(value) <= 1e-10:
            break
        jacobian = _numerical_jacobian(
            function,
            state,
            value,
        )
        try:
            step = np.linalg.solve(jacobian, -value)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(
                jacobian,
                -value,
                rcond=None,
            )[0]
        factor = 1.0
        norm = float(np.linalg.norm(value))
        for _ in range(12):
            trial = state + factor * step
            if np.linalg.norm(function(trial)) < norm:
                state = trial
                break
            factor *= 0.5
        else:
            break
    residual = function(state)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1e-10:
        spectral_abscissa = float(
            np.max(
                np.real(
                    np.linalg.eigvals(
                        _numerical_jacobian(
                            function,
                            state,
                            residual,
                        )
                    )
                )
            )
        )
        stable = bool(
            np.isfinite(spectral_abscissa)
            and spectral_abscissa < 0.0
        )
    else:
        spectral_abscissa = float("nan")
        stable = False
    return (
        state,
        residual_norm,
        stable,
        spectral_abscissa,
    )


def _circuit_condition(
    model,
    context,
    tensors,
    operating,
    state,
):
    if not isinstance(operating, dict):
        return 1.0
    resistance = model.background.wire_resistances(
        context,
        state,
    )
    total = (
        tensors.z_field
        + np.diag(resistance)
        + model._series_impedance(
            operating.get("series_impedance"),
            len(resistance),
        )
    )
    return float(np.linalg.cond(total))


def _trajectory_case(
    model,
    geometry,
    truth,
    predicted,
    operating,
    times,
    *,
    integrator_rtol,
    integrator_atol,
    integrator_max_step,
):
    truth_context = build_online_thermal_context(
        model.background,
        geometry,
        truth.cell_h,
        time_scales=model.thermal_time_scales,
        conditioning_limit=model.thermal_conditioning_limit,
        target_relative_error=model.thermal_target_relative_error,
    )
    predicted_context = model.geometry_context(geometry)
    truth_states = _integrate(
        model,
        truth_context,
        truth,
        operating,
        times,
    )
    predicted_states = _integrate(
        model,
        predicted_context,
        predicted,
        operating,
        times,
    )
    phi_truth = np.asarray(
        truth_context.thermal_basis,
        float,
    )
    phi_pred = np.asarray(
        predicted_context.thermal_basis,
        float,
    )
    mass = truth_context.thermal_mass_full
    worst = 0.0
    maximum_circuit_condition = 1.0
    maximum_integrator_error = 0.0
    rows = []
    zero_pred = np.zeros(phi_pred.shape[1], float)

    for time, a_truth, a_pred in zip(
        times,
        truth_states,
        predicted_states,
    ):
        theta_truth = phi_truth @ a_truth
        theta_pred = phi_pred @ a_pred
        field_error = _mass_field_relative(
            theta_pred - theta_truth,
            theta_truth,
            mass,
        )
        scale = max(
            float(np.max(np.abs(theta_truth))),
            np.finfo(float).tiny,
        )
        tmin = (
            abs(
                float(
                    np.min(theta_pred)
                    - np.min(theta_truth)
                )
            )
            / scale
        )
        tmax = (
            abs(
                float(
                    np.max(theta_pred)
                    - np.max(theta_truth)
                )
            )
            / scale
        )
        wire_error = 0.0
        for truth_weights, pred_weights in zip(
            truth_context.line_heat_weights,
            predicted_context.line_heat_weights,
        ):
            wt = np.asarray(truth_weights, float)
            wp = np.asarray(pred_weights, float)
            ref = float(wt @ theta_truth)
            val = float(wp @ theta_pred)
            wire_error = max(
                wire_error,
                abs(val - ref)
                / max(
                    abs(ref),
                    1e-10 * scale,
                    np.finfo(float).tiny,
                ),
            )

        z_truth = (
            truth.z_field
            + np.diag(
                model.background.wire_resistances(
                    truth_context,
                    a_truth,
                )
            )
        )
        z_pred = (
            predicted.z_field
            + np.diag(
                model.background.wire_resistances(
                    predicted_context,
                    a_pred,
                )
            )
        )
        z_error = _relative(
            z_pred,
            z_truth,
            np.linalg.norm(z_truth),
        )
        currents_truth = model._currents(
            a_truth,
            truth_context,
            truth,
            operating,
        )
        currents_pred = model._currents(
            a_pred,
            predicted_context,
            predicted,
            operating,
        )
        current_error = _relative(
            currents_pred,
            currents_truth,
            np.linalg.norm(currents_truth),
        )
        condition = max(
            _circuit_condition(
                model,
                truth_context,
                truth,
                operating,
                a_truth,
            ),
            _circuit_condition(
                model,
                predicted_context,
                predicted,
                operating,
                a_pred,
            ),
        )
        maximum_circuit_condition = max(
            maximum_circuit_condition,
            condition,
        )

        production = model.predict(
            float(time),
            initial_state=zero_pred,
            geometry=geometry,
            operating=operating,
            max_step=min(
                float(integrator_max_step),
                float(time),
            ),
            method="etd2_adaptive",
            rtol=float(integrator_rtol),
            atol=float(integrator_atol),
        )
        production_theta = (
            phi_pred @ np.asarray(production.state, float)
        )
        integrator_error = _mass_field_relative(
            production_theta - theta_pred,
            theta_pred,
            mass,
        )
        maximum_integrator_error = max(
            maximum_integrator_error,
            integrator_error,
        )
        composite = max(
            field_error,
            tmin,
            tmax,
            wire_error,
            z_error,
            current_error,
        )
        worst = max(worst, composite)
        rows.append(
            {
                "time": float(time),
                "field_mass_relative_error": float(field_error),
                "minimum_temperature_relative_error": float(tmin),
                "maximum_temperature_relative_error": float(tmax),
                "maximum_wire_temperature_relative_error": float(
                    wire_error
                ),
                "impedance_relative_error": float(z_error),
                "current_relative_error": float(current_error),
                "integrator_reference_relative_error": float(
                    integrator_error
                ),
                "circuit_condition": float(condition),
                "composite_relative_error": float(composite),
            }
        )

    (
        steady_truth,
        r_truth,
        stable_truth,
        alpha_truth,
    ) = _steady(
        model,
        truth_context,
        truth,
        operating,
        truth_states[-1],
    )
    (
        steady_pred,
        r_pred,
        stable_pred,
        alpha_pred,
    ) = _steady(
        model,
        predicted_context,
        predicted,
        operating,
        predicted_states[-1],
    )
    theta_truth = phi_truth @ steady_truth
    theta_pred = phi_pred @ steady_pred
    scale = max(
        float(np.max(np.abs(theta_truth))),
        np.finfo(float).tiny,
    )
    steady_error = max(
        _mass_field_relative(
            theta_pred - theta_truth,
            theta_truth,
            mass,
        ),
        abs(
            float(
                np.min(theta_pred)
                - np.min(theta_truth)
            )
        )
        / scale,
        abs(
            float(
                np.max(theta_pred)
                - np.max(theta_truth)
            )
        )
        / scale,
    )
    for truth_weights, pred_weights in zip(
        truth_context.line_heat_weights,
        predicted_context.line_heat_weights,
    ):
        ref = float(
            np.asarray(truth_weights, float) @ theta_truth
        )
        val = float(
            np.asarray(pred_weights, float) @ theta_pred
        )
        steady_error = max(
            steady_error,
            abs(val - ref)
            / max(
                abs(ref),
                1e-10 * scale,
                np.finfo(float).tiny,
            ),
        )
    z_truth = (
        truth.z_field
        + np.diag(
            model.background.wire_resistances(
                truth_context,
                steady_truth,
            )
        )
    )
    z_pred = (
        predicted.z_field
        + np.diag(
            model.background.wire_resistances(
                predicted_context,
                steady_pred,
            )
        )
    )
    steady_error = max(
        steady_error,
        _relative(
            z_pred,
            z_truth,
            np.linalg.norm(z_truth),
        ),
    )
    maximum_circuit_condition = max(
        maximum_circuit_condition,
        _circuit_condition(
            model,
            truth_context,
            truth,
            operating,
            steady_truth,
        ),
        _circuit_condition(
            model,
            predicted_context,
            predicted,
            operating,
            steady_pred,
        ),
    )
    worst = max(worst, steady_error)
    return {
        "maximum_relative_error": float(worst),
        "maximum_integrator_relative_error": float(
            maximum_integrator_error
        ),
        "maximum_circuit_condition": float(
            maximum_circuit_condition
        ),
        "truth_steady_residual": float(r_truth),
        "predicted_steady_residual": float(r_pred),
        "truth_steady_stable": bool(stable_truth),
        "predicted_steady_stable": bool(stable_pred),
        "truth_steady_spectral_abscissa": float(alpha_truth),
        "predicted_steady_spectral_abscissa": float(alpha_pred),
        "steady_relative_error": float(steady_error),
        "truth_online_rank": int(phi_truth.shape[1]),
        "predicted_online_rank": int(phi_pred.shape[1]),
        "times": rows,
    }


def _audit_operating_cases(settings, cfg):
    configured = cfg.get("operating_cases")
    if configured is not None:
        cases = []
        for index, item in enumerate(list(configured)):
            if not isinstance(item, dict):
                raise ValueError(
                    "final_audit.operating_cases entries must be mappings"
                )
            name = str(item.get("name", f"case_{index}"))
            if "drive" in item:
                value = item["drive"]
            elif "operating" in item:
                value = item["operating"]
            else:
                raise ValueError(
                    "each final audit operating case needs 'operating' or 'drive'"
                )
            cases.append((name, value))
        if not cases:
            raise ValueError(
                "final_audit.operating_cases cannot be empty"
            )
        return cases
    if "drive" in cfg:
        return [("circuit", cfg["drive"])]
    if "operating" in cfg:
        return [("current", cfg["operating"])]
    prediction = settings["PREDICTION"]
    value = prediction.get(
        "drive",
        prediction.get("operating"),
    )
    if value is None:
        raise ValueError(
            "final audit requires an explicit operating or drive setting"
        )
    return [("prediction_default", value)]


def run_spatial_final_held_out_audit(
    settings,
    model,
    geometries,
    monitor=None,
    *,
    truth_cache_path=None,
    truth_cache_key=None,
):
    geometries = list(geometries)
    if not geometries:
        raise ValueError(
            "final held-out audit needs at least one geometry"
        )
    cfg = dict(
        settings["TRAINING"].get("final_audit", {})
    )
    times = np.asarray(
        cfg.get(
            "times",
            settings["TRAINING"].get(
                "thermal_trajectory_times",
                [0.1, 1.0, 10.0, 100.0],
            ),
        ),
        float,
    )
    if (
        times.ndim != 1
        or times.size == 0
        or np.any(times <= 0.0)
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            "final audit times must be strictly increasing and positive"
        )
    tensor_tol = float(
        cfg.get("tensor_relative_tolerance", 2e-1)
    )
    current_tol = float(
        cfg.get("current_space_relative_tolerance", 2e-1)
    )
    outward_tol = float(
        cfg.get("outward_relative_tolerance", 2e-1)
    )
    projection_limit = float(
        cfg.get("projection_correction_limit", 2e-1)
    )
    dynamic_tol = float(
        cfg.get("reduced_dynamic_relative_tolerance", 1e-1)
    )
    thermal_tol = float(
        cfg.get(
            "full_vs_rom_thermal_tolerance",
            settings["TRAINING"].get(
                "thermal_basis_energy_tolerance",
                5e-2,
            ),
        )
    )
    circuit_limit = float(
        cfg.get("circuit_condition_limit", 1e8)
    )
    integrator_tol = float(
        cfg.get("integrator_relative_tolerance", 1e-4)
    )
    integrator_rtol = float(
        cfg.get("integrator_rtol", 1e-7)
    )
    integrator_atol = float(
        cfg.get("integrator_atol", 1e-9)
    )
    integrator_max_step = float(
        cfg.get("integrator_max_step", 10.0)
    )
    operating_cases = _audit_operating_cases(
        settings,
        cfg,
    )

    truth_bundles = _load_truth_cache(
        truth_cache_path,
        cache_key=truth_cache_key,
        geometries=geometries,
        n_ports=model.surrogate.n_ports,
        n_cells=model.background.n_cells,
    )
    if truth_bundles is None:
        truth_bundles = []
        for index, geometry in enumerate(geometries):
            if monitor is not None:
                monitor.checkpoint()
            print(
                "生成 completely-held-out spatial truth……"
                f"{index + 1}/{len(geometries)}",
                flush=True,
            )
            truth_bundles.append(
                _truth_tensors(
                    model.background,
                    geometry,
                )
            )
        _save_truth_cache(
            truth_cache_path,
            cache_key=truth_cache_key,
            geometries=geometries,
            bundles=truth_bundles,
        )
    elif truth_cache_path is not None:
        print(
            "复用 completely-held-out spatial truth cache。",
            flush=True,
        )

    rows = []
    thermal_rows = []
    for index, (geometry, truth_bundle) in enumerate(
        zip(geometries, truth_bundles)
    ):
        if monitor is not None:
            monitor.checkpoint()
        truth, predicted, tensor = _tensor_case(
            model,
            geometry,
            truth_projection_limit=projection_limit,
            truth_bundle=truth_bundle,
        )
        thermal = audit_online_thermal_trajectories(
            model.background,
            geometry,
            truth.cell_h,
            times=times,
            time_scales=model.thermal_time_scales,
            conditioning_limit=model.thermal_conditioning_limit,
            target_relative_error=thermal_tol,
        )
        thermal_rows.append(thermal)
        trajectories = []
        for name, operating in operating_cases:
            trajectory = _trajectory_case(
                model,
                geometry,
                truth,
                predicted,
                operating,
                times,
                integrator_rtol=integrator_rtol,
                integrator_atol=integrator_atol,
                integrator_max_step=integrator_max_step,
            )
            trajectories.append(
                {
                    "name": name,
                    "operating": operating,
                    **trajectory,
                }
            )
        dynamic = max(
            row["maximum_relative_error"]
            for row in trajectories
        )
        integrator = max(
            row["maximum_integrator_relative_error"]
            for row in trajectories
        )
        rows.append(
            {
                "index": int(index),
                "tensor": tensor,
                "online_thermal": thermal,
                "operating_cases": trajectories,
            }
        )
        print(
            "completely-held-out spatial final audit……"
            f"{index + 1}/{len(geometries)} "
            f"tensor={max(tensor['z_relative_error'], tensor['d_relative_error'], tensor['spatial_joule_relative_error']):.3e} "
            f"thermal={thermal['maximum_mass_relative_error']:.3e} "
            f"dynamic={dynamic:.3e} integrator={integrator:.3e}",
            flush=True,
        )

    trajectory_rows = [
        case
        for row in rows
        for case in row["operating_cases"]
    ]
    maximum_tensor = max(
        max(
            row["tensor"]["z_relative_error"],
            row["tensor"]["d_relative_error"],
            row["tensor"]["spatial_joule_relative_error"],
        )
        for row in rows
    )
    maximum_current = max(
        row["tensor"]["maximum_current_space_relative_error"]
        for row in rows
    )
    maximum_outward = max(
        row["tensor"]["outward_relative_error"]
        for row in rows
    )
    maximum_projection = max(
        max(
            row["tensor"]["zd_projection_correction"],
            row["tensor"]["spatial_projection_correction"],
        )
        for row in rows
    )
    maximum_thermal = max(
        row["maximum_mass_relative_error"]
        for row in thermal_rows
    )
    maximum_dynamic = max(
        row["maximum_relative_error"]
        for row in trajectory_rows
    )
    maximum_integrator = max(
        row["maximum_integrator_relative_error"]
        for row in trajectory_rows
    )
    maximum_circuit = max(
        row["maximum_circuit_condition"]
        for row in trajectory_rows
    )
    steady_ok = all(
        row["truth_steady_residual"] <= 1e-10
        and row["predicted_steady_residual"] <= 1e-10
        and row["truth_steady_stable"]
        and row["predicted_steady_stable"]
        for row in trajectory_rows
    )
    heldout_truth_ok = all(
        bool(row["tensor"]["truth_physics"]["certified"])
        for row in rows
    )
    checks = {
        "heldout_truth_physics_ok": heldout_truth_ok,
        "full_vs_rom_thermal_ok": maximum_thermal <= thermal_tol,
        "tensor_surrogate_ok": maximum_tensor <= tensor_tol,
        "current_space_contractions_ok": maximum_current <= current_tol,
        "physical_outward_loss_ok": maximum_outward <= outward_tol,
        "decoder_projection_correction_ok": (
            maximum_projection <= projection_limit
        ),
        "reduced_end_to_end_dynamics_ok": (
            maximum_dynamic <= dynamic_tol
        ),
        "production_integrator_ok": (
            maximum_integrator <= integrator_tol
        ),
        "steady_state_ok": steady_ok,
        "circuit_conditioning_ok": (
            maximum_circuit <= circuit_limit
        ),
    }
    certified = all(bool(value) for value in checks.values())
    return {
        **{k: bool(v) for k, v in checks.items()},
        "certified": bool(certified),
        "certificate_level": (
            "frozen_held_out_numerical_validation"
        ),
        "tensor_representation": "cellwise_joule_tensor_v1",
        "thermal_representation": (
            "geometry_local_rational_krylov_v1"
        ),
        "maximum_tensor_relative_error": float(maximum_tensor),
        "maximum_current_space_relative_error": float(
            maximum_current
        ),
        "maximum_outward_relative_error": float(
            maximum_outward
        ),
        "maximum_projection_correction": float(
            maximum_projection
        ),
        "maximum_full_vs_rom_thermal_relative_error": float(
            maximum_thermal
        ),
        "maximum_dynamic_relative_error": float(
            maximum_dynamic
        ),
        "maximum_integrator_relative_error": float(
            maximum_integrator
        ),
        "maximum_circuit_condition": float(
            maximum_circuit
        ),
        "samples": rows,
    }


__all__ = ["run_spatial_final_held_out_audit"]
