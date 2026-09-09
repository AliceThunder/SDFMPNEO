from __future__ import annotations

import os
import time

import numpy as np

_INSTALLED = False


def install_cpp_training_visibility() -> None:
    """Expose formerly silent expensive stages in console logs."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import late_stage_runtime as late
    from . import research as training_research
    from .parallel_runtime import training_parallelism
    from ..cpp_training_backend import backend_info
    from ..cpp_dag_backend import backend_info as dag_backend_info

    original_prepare = late._prepare_working_set
    original_evaluate = training_research._evaluate
    original_refine = training_research._refine_weights

    def prepare(field, *point_sets):
        if not hasattr(field, "prepare_training_contexts"):
            return original_prepare(field, *point_sets)
        if not getattr(field, "_sdfmpneo_cpp_status_reported", False):
            # Build/load both native libraries exactly once before worker fan-out.
            info = backend_info(auto_build=True)
            dag = dag_backend_info(auto_build=True)
            parallel = training_parallelism()
            if info["available"]:
                print(
                    "C++物理训练后端已启用："
                    f"{info['version']}，OpenMP={'是' if info['openmp'] else '否'}，"
                    f"point workers={parallel['point_workers']}，"
                    f"单点C++线程={info['threads']}",
                    flush=True,
                )
            else:
                print(
                    "C++物理训练后端不可用，回退Python实现："
                    f"{info['error']}；point workers={parallel['point_workers']}",
                    flush=True,
                )
            if dag["available"]:
                print(
                    "C++解析DAG/Gauss-Newton后端已启用："
                    f"OpenMP={'是' if dag['openmp'] else '否'}，"
                    f"native threads={dag['native_threads']}；"
                    "批量DAG/Jacobian阶段直接使用原生OS线程",
                    flush=True,
                )
            else:
                print(
                    "C++解析DAG/Gauss-Newton后端不可用，回退Python实现："
                    f"{dag['error']}",
                    flush=True,
                )
            chart = getattr(field, "chart", None)
            if chart is not None and getattr(chart, "_sdfmpneo_topology_mesh", None) is None:
                try:
                    chart.mesh(np.zeros(chart.n_parameters, dtype=float))
                except Exception:
                    pass
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
        value = original_prepare(field, *point_sets)
        elapsed = time.perf_counter() - start
        print(f"几何物理工作集准备完成：{elapsed:.3f} s", flush=True)
        return value

    def evaluate(graph, field, points, *args, **kwargs):
        start = time.perf_counter()
        value = original_evaluate(graph, field, points, *args, **kwargs)
        elapsed = time.perf_counter() - start
        threshold = float(os.environ.get("SDFMPNEO_TIMING_LOG_S", "3"))
        if elapsed >= threshold:
            print(
                "残差批量评估完成："
                f"points={len(points)}，jacobian={bool(kwargs.get('jacobian', False))}，"
                f"nodes={len(graph.response_nodes)}，耗时={elapsed:.3f} s",
                flush=True,
            )
        return value

    def refine(graph, field, points, *args, **kwargs):
        print(
            f"开始联合权重优化：nodes={len(graph.response_nodes)}，points={len(points)}……",
            flush=True,
        )
        start = time.perf_counter()
        value = original_refine(graph, field, points, *args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"联合权重优化完成：{elapsed:.3f} s", flush=True)
        return value

    late._prepare_working_set = prepare
    training_research._evaluate = evaluate
    training_research._refine_weights = refine
    _INSTALLED = True
