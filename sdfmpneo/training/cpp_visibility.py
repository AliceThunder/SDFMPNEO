from __future__ import annotations

import time

_INSTALLED = False


def install_cpp_training_visibility() -> None:
    """Expose the formerly silent geometry-working-set stage in console logs."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import late_stage_runtime as late
    from .parallel_runtime import training_parallelism
    from ..cpp_training_backend import backend_info

    original = late._prepare_working_set

    def prepare(field, *point_sets):
        if not hasattr(field, "prepare_training_contexts"):
            return original(field, *point_sets)
        if not getattr(field, "_sdfmpneo_cpp_status_reported", False):
            info = backend_info(auto_build=True)
            parallel = training_parallelism()
            if info["available"]:
                print(
                    "C++训练后端已启用："
                    f"{info['version']}，OpenMP={'是' if info['openmp'] else '否'}，"
                    f"point workers={parallel['point_workers']}，"
                    f"C++内部线程={info['threads']}",
                    flush=True,
                )
            else:
                print(
                    "C++训练后端不可用，回退Python实现："
                    f"{info['error']}；point workers={parallel['point_workers']}",
                    flush=True,
                )
            try:
                setattr(field, "_sdfmpneo_cpp_status_reported", True)
            except Exception:
                pass
        count = 0
        for values in point_sets:
            try:
                count += len(values)
            except TypeError:
                pass
        print(f"准备几何物理工作集（输入配点约 {count}）……", flush=True)
        start = time.perf_counter()
        value = original(field, *point_sets)
        elapsed = time.perf_counter() - start
        print(f"几何物理工作集准备完成：{elapsed:.3f} s", flush=True)
        return value

    late._prepare_working_set = prepare
    _INSTALLED = True
