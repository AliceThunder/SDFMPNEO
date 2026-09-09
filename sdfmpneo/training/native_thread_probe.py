from __future__ import annotations

import ctypes

from .. import cpp_dag_backend


def observed_native_threads(requested: int | None = None) -> int:
    """Return actual OpenMP team size created by the native DAG backend."""
    lib = cpp_dag_backend._load(auto_build=True)
    if lib is None:
        return 0
    function = lib.sdfmpneo_native_thread_probe
    function.argtypes = [ctypes.c_int]
    function.restype = ctypes.c_int
    target = cpp_dag_backend.native_threads() if requested is None else max(1, int(requested))
    return int(function(target))
