from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
import os

import numpy as np
import scipy.linalg

from .parallel_runtime import _ordered_map

os.environ.setdefault("SDFMPNEO_POINT_WORKERS", str(max(1, min(8, os.cpu_count() or 1))))
os.environ.setdefault("SDFMPNEO_BLAS_THREADS", "1")


def _prepare_working_set(field, *point_sets) -> None:
    prepare = getattr(field, "prepare_training_contexts", None)
    if prepare is not None:
        prepare(*point_sets)


def _thermal_factor(context):
    factor = getattr(context, "_sdfmpneo_thermal_factor", None)
    if factor is None:
        factor = scipy.linalg.cho_factor(
            np.asarray(context.M, dtype=float), lower=True, check_finite=False
        )
        setattr(context, "_sdfmpneo_thermal_factor", factor)
    return factor


def _thermal_diffusion_jacobian(context):
    cached = getattr(context, "_sdfmpneo_diffusion_jacobian", None)
    if cached is None:
        cached = scipy.linalg.cho_solve(
            _thermal_factor(context),
            -np.asarray(context.K, dtype=float),
            check_finite=False,
        )
        setattr(context, "_sdfmpneo_diffusion_jacobian", cached)
    return cached


def _prepare_operating_contexts(field, operating_rows) -> None:
    if not (hasattr(field, "split") and hasattr(field, "thermal_model")):
        return
    from sdfmpneo.em import modal_heat as mh

    seen = set()
    for operating in np.asarray(operating_rows, dtype=float):
        key = tuple(float(v) for v in operating)
        if key in seen:
            continue
        seen.add(key)
        context, _ = field.split(operating)
        _thermal_factor(context)
        _thermal_diffusion_jacobian(context)
        if mh._supported(context.em.problem):
            mh._geometry_cache(context.em.problem)


def physics_vector_field(field, state, operating):
    from sdfmpneo.em.modal_heat import heat_source_for_reduced_model

    a = np.asarray(state, dtype=float)
    u = np.asarray(operating, dtype=float)
    if hasattr(field, "split") and hasattr(field, "thermal_model"):
        context, current = field.split(u)
        rhs = context.rhs.evaluate(current)
        heat = heat_source_for_reduced_model(context.em, a, rhs)
        return scipy.linalg.cho_solve(
            _thermal_factor(context),
            -np.asarray(context.K, dtype=float) @ a + heat,
            check_finite=False,
        )
    if hasattr(field, "em_model") and hasattr(field, "thermal_model") and hasattr(field, "rhs"):
        rhs = field.rhs(u if getattr(field, "rhs_map", None) is not None else None)
        heat = heat_source_for_reduced_model(field.em_model, a, rhs)
        forcing = np.asarray(getattr(field, "thermal_forcing", np.zeros_like(a)), dtype=float)
        return -np.asarray(field.thermal_model.lambdas, dtype=float) * a + heat + forcing
    if hasattr(field, "vector_field"):
        return np.asarray(field.vector_field(a, u), dtype=float)
    return np.asarray(field.evaluate(a, u).vector_field, dtype=float)


def gn_field_jacobian(field, state, operating):
    u = np.asarray(operating, dtype=float)
    if hasattr(field, "split") and hasattr(field, "thermal_model"):
        context, _ = field.split(u)
        return _thermal_diffusion_jacobian(context)
    if hasattr(field, "em_model") and hasattr(field, "thermal_model"):
        return -np.diag(np.asarray(field.thermal_model.lambdas, dtype=float))
    return np.asarray(field.evaluate(state, u).vector_field_jacobian, dtype=float)


@lru_cache(maxsize=4096)
def _normalized_third_moment_kernel(powers: tuple[int, int, int, int]) -> np.ndarray:
    from math import factorial

    tensor = np.empty((4, 4, 4), dtype=float)
    for l in range(4):
        for a in range(4):
            for b in range(4):
                exponents = list(powers)
                exponents[l] += 1
                exponents[a] += 1
                exponents[b] += 1
                numerator = 6
                for value in exponents:
                    numerator *= factorial(value)
                tensor[l, a, b] = numerator / factorial(sum(exponents) + 3)
    tensor.setflags(write=False)
    return tensor


def fast_third_moment_tensor(volume: float, poly) -> np.ndarray:
    tensor = np.zeros((4, 4, 4), dtype=float)
    for powers, coefficient in poly.items():
        tensor += float(volume) * float(coefficient) * _normalized_third_moment_kernel(
            tuple(int(v) for v in powers)
        )
    return tensor


def fast_exact_modal_heat_source(problem, electromagnetic_state, thermal_state) -> np.ndarray:
    from sdfmpneo.em import modal_heat as mh

    if not mh._supported(problem):
        raise TypeError("problem does not expose the nonlinear tetrahedral modal-heat interface")
    state = np.asarray(thermal_state, dtype=float)
    x = np.asarray(electromagnetic_state, dtype=complex)
    if state.shape != (problem.n_thermal,) or x.shape != (problem.n_em,):
        raise ValueError("electromagnetic or thermal state dimension mismatch")
    coefficients, edge_ids, volumes, extraction = mh._geometry_cache(problem)
    conductivity_polynomials, _ = problem._weighted_polynomials(state)
    edge_field = np.asarray(extraction @ x, dtype=complex).reshape(-1)
    local_edge = edge_field[edge_ids]
    C = np.einsum("qp,qpic->qic", local_edge, coefficients, optimize=True)
    gram = np.real(np.einsum("qic,qjc->qij", np.conj(C), C, optimize=True))
    moments = np.zeros((problem.mesh.n_tetrahedra, 4), dtype=float)
    for start in range(0, problem.mesh.n_tetrahedra, 256):
        stop = min(start + 256, problem.mesh.n_tetrahedra)
        tensors = np.stack([
            fast_third_moment_tensor(volumes[q], conductivity_polynomials[q])
            for q in range(start, stop)
        ])
        moments[start:stop] = np.einsum(
            "qab,qlab->ql", gram[start:stop], tensors, optimize=True
        )
    tests = np.asarray(problem.thermal_test_local, dtype=float)
    if tests.shape != (problem.n_thermal, problem.mesh.n_tetrahedra, 4):
        raise ValueError("thermal test-mode shape mismatch")
    return 0.5 * np.einsum("rqi,qi->r", tests, moments, optimize=True)


def _progress(rh, monitor, label):
    if monitor is None:
        return None
    return lambda completed, total: rh._work(monitor, label, completed, total)


def evaluate_physics_batch(network, field, points, *, jacobian=False, monitor=None, work_label=None):
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    _prepare_working_set(field, points)
    n = network.n_modes
    if len(points):
        _prepare_operating_contexts(field, points[:, n:-1])
    label = work_label or ("physics_jacobian" if jacobian else "physics_residual")

    def one(point):
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        if jacobian:
            a, da, ja, jda = network.evaluate_parameter_jacobian(
                time, a0=initial, operating=operating
            )
            F = physics_vector_field(field, a, operating)
            JFJa = rh._gn_apply_field_jacobian(field, a, operating, ja)
            return SimpleNamespace(
                residual=np.asarray(da - F, dtype=float),
                parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
            )
        a, da = network.evaluate(time, a0=initial, operating=operating)
        F = physics_vector_field(field, a, operating)
        return SimpleNamespace(residual=np.asarray(da - F, dtype=float))

    return _ordered_map(one, points, monitor=monitor, progress=_progress(rh, monitor, label))


def evaluate_semigroup_batch(network, rows, *, jacobian=False, monitor=None, work_label=None):
    from . import research_helpers as rh

    rows = np.asarray(rows, dtype=float)
    n = network.n_modes
    horizon = float(network.max_response_time)
    label = work_label or ("restart_jacobian" if jacobian else "restart_residual")

    def one(row):
        initial, operating = row[:n], row[n:-2]
        t1, t2 = float(row[-2]), float(row[-1])
        total = t1 + t2
        if total > horizon + 64.0 * np.finfo(float).eps * horizon:
            raise ValueError("semigroup sample exceeds max_response_time")
        total = min(total, horizon)
        if jacobian:
            direct, _, Jdirect, _ = network.evaluate_parameter_jacobian(total, a0=initial, operating=operating)
            first, _, Jfirst, _ = network.evaluate_parameter_jacobian(t1, a0=initial, operating=operating)
            restarted, _, Jrestart, _ = network.evaluate_parameter_jacobian(t2, a0=first, operating=operating)
            _, _, Jinitial, _ = network.evaluate_initial_jacobian(t2, a0=first, operating=operating)
            defect = np.asarray(direct - restarted, dtype=float)
            return SimpleNamespace(
                residual=defect / horizon,
                raw_defect=defect,
                parameter_jacobian=np.asarray(Jdirect - (Jrestart + Jinitial @ Jfirst), dtype=float) / horizon,
            )
        direct, _ = network.evaluate(total, a0=initial, operating=operating)
        first, _ = network.evaluate(t1, a0=initial, operating=operating)
        restarted, _ = network.evaluate(t2, a0=first, operating=operating)
        defect = np.asarray(direct - restarted, dtype=float)
        return SimpleNamespace(residual=defect / horizon, raw_defect=defect)

    return _ordered_map(one, rows, monitor=monitor, progress=_progress(rh, monitor, label))


def source_prefit(network, field, config, monitor=None):
    from .source_prefit_factorization import fit_source_factors

    samples = np.asarray(config.source_points(), dtype=float)
    _prepare_working_set(field, samples)
    n = network.n_modes
    if len(samples):
        _prepare_operating_contexts(field, samples[:, n:])

    def one(row):
        a0, operating = row[:n], row[n:]
        return np.asarray(physics_vector_field(field, a0, operating), float) + np.asarray(network.lambdas) * a0

    desired = _ordered_map(one, samples, monitor=monitor)
    fitted, residual = fit_source_factors(network, samples, np.vstack(desired))
    norms = np.linalg.norm(residual, axis=1)
    return fitted, SimpleNamespace(
        rms=float(np.sqrt(np.mean(norms * norms))) if len(norms) else 0.0,
        maximum=float(np.max(norms, initial=0.0)),
        sample_count=int(len(samples)),
    )


def evaluate_layer_physics(network, field, points, layer, *, monitor=None, work_label="physics_layer_jacobian"):
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    _prepare_working_set(field, points)
    n = network.n_modes
    if len(points):
        _prepare_operating_contexts(field, points[:, n:-1])

    def one(point):
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        a, da, ja, jda, ids = network.evaluate_layer_amplitude_jacobian(
            time, a0=initial, operating=operating, layer=layer
        )
        F = physics_vector_field(field, a, operating)
        JFJa = rh._gn_apply_field_jacobian(field, a, operating, ja)
        return SimpleNamespace(
            residual=np.asarray(da - F, dtype=float),
            parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
            parameter_indices=np.asarray(ids, dtype=int),
        )

    records = _ordered_map(one, points, monitor=monitor, progress=_progress(rh, monitor, work_label))
    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    for record in records:
        if not np.array_equal(ids, record.parameter_indices):
            raise RuntimeError("layer amplitude parameter block changed across collocation points")
    return records, ids


def evaluate_layer_semigroup(network, rows, layer, *, monitor=None, work_label="restart_layer_jacobian"):
    from . import research_helpers as rh

    rows = np.asarray(rows, dtype=float)
    n = network.n_modes
    horizon = float(network.max_response_time)

    def one(row):
        initial, operating = row[:n], row[n:-2]
        t1, t2 = float(row[-2]), float(row[-1])
        direct, _, Jdirect, _, ids = network.evaluate_layer_amplitude_jacobian(
            min(t1 + t2, horizon), a0=initial, operating=operating, layer=layer
        )
        first, _ = network.evaluate(t1, a0=initial, operating=operating)
        restarted, _, Jrestart, _, restart_ids = network.evaluate_layer_amplitude_jacobian(
            t2, a0=first, operating=operating, layer=layer
        )
        if not np.array_equal(ids, restart_ids):
            raise RuntimeError("restart layer parameter block changed")
        defect = np.asarray(direct - restarted, dtype=float)
        return SimpleNamespace(
            residual=defect / horizon,
            raw_defect=defect,
            parameter_jacobian=np.asarray(Jdirect - Jrestart, dtype=float) / horizon,
            parameter_indices=np.asarray(ids, dtype=int),
        )

    records = _ordered_map(one, rows, monitor=monitor, progress=_progress(rh, monitor, work_label))
    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    for record in records:
        if not np.array_equal(ids, record.parameter_indices):
            raise RuntimeError("restart layer parameter block changed")
    return records, ids


def install_physics_acceleration() -> None:
    from sdfmpneo.em import modal_heat as mh
    from . import research_helpers as rh, research_multilayer as rm, research_trainer as rt

    if getattr(rh, "_sdfmpneo_fixed_acceleration", False):
        return
    mh._third_moment_tensor = fast_third_moment_tensor
    mh.exact_modal_heat_source = fast_exact_modal_heat_source
    rh._physics_vector_field = physics_vector_field
    rh._gn_field_jacobian = gn_field_jacobian
    rh._evaluate_network = evaluate_physics_batch
    rh._evaluate_semigroup = evaluate_semigroup_batch
    rm._physics_vector_field = physics_vector_field
    rm.evaluate_layer_physics = evaluate_layer_physics
    rm.evaluate_layer_semigroup = evaluate_layer_semigroup
    rm.source_prefit = source_prefit
    rt._evaluate_network = evaluate_physics_batch
    rt._evaluate_semigroup = evaluate_semigroup_batch
    rt._evaluate_all = rh._evaluate_all
    rt.source_prefit = source_prefit
    rh._sdfmpneo_fixed_acceleration = True
