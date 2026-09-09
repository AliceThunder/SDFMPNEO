from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from threading import RLock

import numpy as np

from . import cpp_training_backend as _base

_LIB = None
_ERROR: str | None = None
_ATTEMPTED = False
_LOCK = RLock()


def _source() -> Path:
    return Path(__file__).resolve().with_name("cpp_dag_combined.cpp")


def _build_dir() -> Path:
    return Path(__file__).resolve().parent / "_cpp_dag_build"


def _signature() -> str:
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "cpp_dag_combined.cpp",
        "cpp_training_backend.cpp",
        "cpp_dag_values.inc",
        "cpp_dag_backend.inc",
        "cpp_thread_probe.inc",
    ):
        path = root / name
        h.update(name.encode("utf-8"))
        h.update(_base._sha(path).encode("ascii"))
    h.update(b"sdfmpneo_dag_cpp_v1")
    return h.hexdigest()


def native_threads() -> int:
    raw = os.environ.get("SDFMPNEO_NATIVE_THREADS")
    if raw is None:
        return max(1, int(os.cpu_count() or 1))
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError("SDFMPNEO_NATIVE_THREADS must be a positive integer") from exc


def _configure(lib) -> None:
    pd = ctypes.POINTER(ctypes.c_double)
    pi32 = ctypes.POINTER(ctypes.c_int32)
    pi64 = ctypes.POINTER(ctypes.c_int64)
    lib.sdfmpneo_dag_batch_values_f64.argtypes = [
        pd, pd, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        pd, pi32, pi32, pi32, pi64,
        pi64, pi32, pi64, pi32,
        pd, pd, pd, pd, pd, pd, ctypes.c_int,
    ]
    lib.sdfmpneo_dag_batch_values_f64.restype = ctypes.c_int
    lib.sdfmpneo_dag_batch_sparse_jacobian_f64.argtypes = [
        pd, pd, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        pd, pi32, pi32, pi32, pi64, pi64, pi64, pi32,
        pi64, pi32, pi64, pi32,
        pd, pd, pd, pd, pd, pd, pd, pd, ctypes.c_int,
    ]
    lib.sdfmpneo_dag_batch_sparse_jacobian_f64.restype = ctypes.c_int
    lib.sdfmpneo_gn_linearize_f64.argtypes = [
        pd, pd, pd, pd, pd,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        pd, pd, ctypes.c_int,
    ]
    lib.sdfmpneo_gn_linearize_f64.restype = ctypes.c_int
    lib.sdfmpneo_native_thread_probe.argtypes = [ctypes.c_int]
    lib.sdfmpneo_native_thread_probe.restype = ctypes.c_int


def _load(*, auto_build: bool = True):
    global _LIB, _ERROR, _ATTEMPTED
    if str(os.environ.get("SDFMPNEO_CPP_DISABLE", "0")).lower() in {"1", "true", "yes", "on"}:
        _ERROR = "disabled by SDFMPNEO_CPP_DISABLE"
        return None
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        if _ATTEMPTED and auto_build:
            return None
        old_source = _base._source
        old_build_dir = _base._build_dir
        old_signature = _base._signature
        try:
            _base._source = _source
            _base._build_dir = _build_dir
            _base._signature = _signature
            path = _base._valid_manifest_library()
            if path is None:
                if not auto_build:
                    _ERROR = "compatible native DAG backend is not built"
                    return None
                _ATTEMPTED = True
                path = _base.build_inplace(force=False)
            manifest = _base._read_manifest()
            _base._apply_runtime_dirs(manifest.get("runtime_dirs") or [])
            lib = ctypes.CDLL(str(path))
            _base._configure(lib)
            _configure(lib)
            # Touch every new symbol now, so a stale/partial library fails before training.
            for name in (
                "sdfmpneo_dag_batch_values_f64",
                "sdfmpneo_dag_batch_sparse_jacobian_f64",
                "sdfmpneo_gn_linearize_f64",
                "sdfmpneo_native_thread_probe",
            ):
                getattr(lib, name)
            _LIB = lib
            _ERROR = None
            return lib
        except Exception as exc:
            if auto_build:
                _ATTEMPTED = True
            _ERROR = repr(exc)
            return None
        finally:
            _base._source = old_source
            _base._build_dir = old_build_dir
            _base._signature = old_signature


def backend_info(*, auto_build: bool = True) -> dict:
    lib = _load(auto_build=auto_build)
    return {
        "available": lib is not None,
        "native_threads": native_threads(),
        "openmp": False if lib is None else bool(lib.sdfmpneo_training_backend_has_openmp()),
        "error": _ERROR,
    }


def _f64(value):
    return np.ascontiguousarray(value, dtype=np.float64)


def _i32(value):
    return np.ascontiguousarray(value, dtype=np.int32)


def _i64(value):
    return np.ascontiguousarray(value, dtype=np.int64)


def dag_batch_values(plan, initial, operating, weights, rows, initial_values, initial_slopes):
    lib = _load(auto_build=True)
    if lib is None:
        return None
    initial = _f64(initial); operating = _f64(operating); weights = _f64(weights)
    value_rows = _f64(rows[0]); slope_rows = _f64(rows[1])
    initial_values = _f64(initial_values); initial_slopes = _f64(initial_slopes)
    n_points, n_modes = initial.shape
    n_operating = operating.shape[1]
    n_nodes = weights.size
    a = np.empty((n_points, n_modes), dtype=np.float64)
    da = np.empty_like(a)
    pd = ctypes.POINTER(ctypes.c_double); pi32 = ctypes.POINTER(ctypes.c_int32); pi64 = ctypes.POINTER(ctypes.c_int64)
    arrays = plan._native_arrays
    code = lib.sdfmpneo_dag_batch_values_f64(
        initial.ctypes.data_as(pd), operating.ctypes.data_as(pd), n_points, n_modes, n_operating, n_nodes,
        weights.ctypes.data_as(pd), arrays["targets"].ctypes.data_as(pi32),
        arrays["response_parent"].ctypes.data_as(pi32), arrays["dimensions"].ctypes.data_as(pi32),
        arrays["state_offsets"].ctypes.data_as(pi64),
        arrays["initial_factor_offsets"].ctypes.data_as(pi64), arrays["initial_factor_indices"].ctypes.data_as(pi32),
        arrays["operating_factor_offsets"].ctypes.data_as(pi64), arrays["operating_factor_indices"].ctypes.data_as(pi32),
        value_rows.ctypes.data_as(pd), slope_rows.ctypes.data_as(pd),
        initial_values.ctypes.data_as(pd), initial_slopes.ctypes.data_as(pd),
        a.ctypes.data_as(pd), da.ctypes.data_as(pd), native_threads(),
    )
    if code:
        raise RuntimeError(f"native DAG value kernel failed with code {code}")
    return a, da


def dag_batch_sparse_jacobian(plan, initial, operating, weights, rows, initial_values, initial_slopes):
    lib = _load(auto_build=True)
    if lib is None:
        return None
    initial = _f64(initial); operating = _f64(operating); weights = _f64(weights)
    value_rows = _f64(rows[0]); slope_rows = _f64(rows[1])
    initial_values = _f64(initial_values); initial_slopes = _f64(initial_slopes)
    n_points, n_modes = initial.shape
    n_operating = operating.shape[1]
    n_nodes = weights.size
    a = np.empty((n_points, n_modes), dtype=np.float64)
    da = np.empty_like(a)
    ja = np.empty((n_points, n_modes, n_nodes), dtype=np.float64)
    jda = np.empty_like(ja)
    pd = ctypes.POINTER(ctypes.c_double); pi32 = ctypes.POINTER(ctypes.c_int32); pi64 = ctypes.POINTER(ctypes.c_int64)
    arrays = plan._native_arrays
    code = lib.sdfmpneo_dag_batch_sparse_jacobian_f64(
        initial.ctypes.data_as(pd), operating.ctypes.data_as(pd), n_points, n_modes, n_operating, n_nodes,
        weights.ctypes.data_as(pd), arrays["targets"].ctypes.data_as(pi32),
        arrays["response_parent"].ctypes.data_as(pi32), arrays["dimensions"].ctypes.data_as(pi32),
        arrays["state_offsets"].ctypes.data_as(pi64), arrays["derivative_offsets"].ctypes.data_as(pi64),
        arrays["ancestor_offsets"].ctypes.data_as(pi64), arrays["ancestors"].ctypes.data_as(pi32),
        arrays["initial_factor_offsets"].ctypes.data_as(pi64), arrays["initial_factor_indices"].ctypes.data_as(pi32),
        arrays["operating_factor_offsets"].ctypes.data_as(pi64), arrays["operating_factor_indices"].ctypes.data_as(pi32),
        value_rows.ctypes.data_as(pd), slope_rows.ctypes.data_as(pd),
        initial_values.ctypes.data_as(pd), initial_slopes.ctypes.data_as(pd),
        a.ctypes.data_as(pd), da.ctypes.data_as(pd), ja.ctypes.data_as(pd), jda.ctypes.data_as(pd), native_threads(),
    )
    if code:
        raise RuntimeError(f"native DAG Jacobian kernel failed with code {code}")
    return a, da, ja, jda


def gn_linearize(da, physical_f, ja, jda, physical_jacobian):
    lib = _load(auto_build=True)
    if lib is None:
        return None
    da = _f64(da); physical_f = _f64(physical_f); ja = _f64(ja); jda = _f64(jda)
    physical_jacobian = _f64(physical_jacobian)
    n_points, n_modes = da.shape
    n_weights = ja.shape[2]
    residual = np.empty_like(da)
    jacobian = np.empty_like(ja)
    pd = ctypes.POINTER(ctypes.c_double)
    code = lib.sdfmpneo_gn_linearize_f64(
        da.ctypes.data_as(pd), physical_f.ctypes.data_as(pd), ja.ctypes.data_as(pd), jda.ctypes.data_as(pd),
        physical_jacobian.ctypes.data_as(pd), n_points, n_modes, n_weights,
        residual.ctypes.data_as(pd), jacobian.ctypes.data_as(pd), native_threads(),
    )
    if code:
        raise RuntimeError(f"native Gauss-Newton linearization kernel failed with code {code}")
    return residual, jacobian
