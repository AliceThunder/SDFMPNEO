"""Bridge the existing file-based training monitor to the neural ROM pipeline.

The PyQt window controls a worker through ``control.json``.  This module keeps
that UI independent of the neural mathematics while making snapshot generation
and AdamW training cooperatively pausable/stoppable.  Continuation checkpoints
are intentionally backward compatible with the earlier v1/v2/v3 formats.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import threading

import numpy as np


class NeuralTrainingRuntime:
    """Cooperative pause/stop plus tolerant warm continuation."""

    def __init__(
        self,
        monitor,
        checkpoint_path,
        *,
        initial_network_state=None,
        initial_pod=None,
        total_snapshots: int | None = None,
        snapshot_checkpoint_path: str | Path | None = None,
    ):
        self.monitor = monitor
        self.checkpoint_path = Path(checkpoint_path)
        self.initial_network_state = initial_network_state
        self.initial_pod = initial_pod
        self.model = None
        self.optimizer = None
        self.pod = None
        self.dataset_hash = None
        self._resume_payload = None
        self._snapshot_count = 0
        self._counter_lock = threading.Lock()
        self._last_snapshot_percent = -1
        self._optimizer_restore_allowed = True
        self.total_snapshots = None if total_snapshots is None else max(1, int(total_snapshots))
        if snapshot_checkpoint_path is not None:
            try:
                with np.load(snapshot_checkpoint_path, allow_pickle=False) as data:
                    completed = np.asarray(data["completed"], dtype=bool)
                    self._snapshot_count = int(np.count_nonzero(completed))
            except (OSError, ValueError, KeyError):
                pass

    @staticmethod
    def _warn(message: str) -> None:
        print(f"[续训兼容] {message}", flush=True)

    def _load_payload(self):
        if self._resume_payload is not None:
            return self._resume_payload
        if not self.checkpoint_path.is_file():
            self._resume_payload = {}
            return self._resume_payload
        try:
            import torch
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            import torch
            payload = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            self._warn("旧 training checkpoint 不是字典格式；忽略并从冻结数据继续。")
            self._resume_payload = {}
            return self._resume_payload
        version = int(payload.get("format_version", 1))
        if version not in {1, 2, 3}:
            self._warn(f"未知 training checkpoint v{version}；忽略权重状态，但保留冻结数据。")
            self._resume_payload = {}
            return self._resume_payload
        if version < 3:
            self._warn(
                f"检测到旧 training checkpoint v{version}；将尽可能恢复，"
                "缺失的 POD/dataset 身份信息会采用安全降级。"
            )
        self._resume_payload = payload
        return payload

    def _monitor_phase(self, name: str) -> None:
        if self.monitor is not None:
            self.monitor.phase(name, check=False)

    def _checkpoint_control(self) -> None:
        if self.monitor is not None:
            self.monitor.checkpoint()

    def _snapshot_completed(self) -> None:
        with self._counter_lock:
            self._snapshot_count += 1
            value = self._snapshot_count
        if self.monitor is not None:
            with self.monitor._lock:
                self.monitor.data["phase"] = "neural_snapshots"
                self.monitor.data["training_points"] = value
        if self.total_snapshots is not None:
            percent = min(100, int(100 * value / self.total_snapshots))
            if percent != self._last_snapshot_percent:
                self._last_snapshot_percent = percent
                print(
                    f"生成 Joule tensor 标签……{percent}%  "
                    f"({min(value, self.total_snapshots)}/{self.total_snapshots})",
                    flush=True,
                )

    @staticmethod
    def _pod_payload(pod):
        if pod is None:
            return None
        return {
            "mean": np.asarray(pod.mean, dtype=float),
            "basis": np.asarray(pod.basis, dtype=float),
            "singular_values": np.asarray(pod.singular_values, dtype=float),
            "thermal_rank": int(pod.thermal_rank),
            "current_dimension": int(pod.current_dimension),
            "total_centered_energy": float(pod.total_centered_energy),
        }

    @staticmethod
    def _pod_from_payload(value):
        if value is None:
            return None
        from .pod import TensorPOD
        return TensorPOD(
            np.asarray(value["mean"], dtype=float),
            np.asarray(value["basis"], dtype=float),
            np.asarray(value["singular_values"], dtype=float),
            int(value["thermal_rank"]),
            int(value["current_dimension"]),
            total_centered_energy=float(value.get("total_centered_energy", np.sum(np.asarray(value["singular_values"], dtype=float) ** 2))),
        )

    def _saved_pod(self):
        payload = self._load_payload()
        try:
            pod = self._pod_from_payload(payload.get("pod"))
        except (KeyError, TypeError, ValueError) as exc:
            self._warn(f"旧 checkpoint 的 POD 无法读取（{exc}）；重新拟合 POD。")
            pod = None
        return self.initial_pod if pod is None else pod

    def _save_training_state(self, model) -> None:
        """Persist dataset identity, POD, network weights and AdamW momentum."""
        try:
            import torch
        except ImportError:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 3,
            "dataset_hash": self.dataset_hash,
            "network_config": model.config.to_dict(),
            "pod": self._pod_payload(self.pod),
            "network_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state": None if self.optimizer is None else self.optimizer.state_dict(),
        }
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.checkpoint_path)
        print(f"已保存续训检查点：{self.checkpoint_path}", flush=True)

    def clear_checkpoint(self) -> None:
        self.checkpoint_path.unlink(missing_ok=True)
        self._resume_payload = {}

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_path.is_file()

    @contextmanager
    def installed(self):
        """Install worker-local hooks and restore all patched callables afterwards."""
        import torch
        from sdfmpneo.training.monitor import TrainingStopped
        from . import pipeline, trainer

        original_fixed_factory = pipeline.fixed_research_tensor_factory
        original_geometry_factory = pipeline.geometry_research_tensor_factory
        original_fit_pod = pipeline._fit_pod
        original_build_network = trainer.build_residual_mlp
        original_adamw = torch.optim.AdamW

        def controlled_factory(factory):
            def make(*args, **kwargs):
                base = factory(*args, **kwargs)

                def evaluate(state, geometry):
                    self._monitor_phase("neural_snapshots")
                    self._checkpoint_control()
                    value = base(state, geometry)
                    self._snapshot_completed()
                    return value

                return evaluate
            return make

        def controlled_fit_pod(dataset, *, rank, tolerance, pod_config=None):
            self._monitor_phase("neural_pod")
            self._checkpoint_control()
            current_hash = dataset.manifest().dataset_hash
            payload = self._load_payload()
            saved_hash = payload.get("dataset_hash")
            if saved_hash is not None and str(saved_hash) != str(current_hash):
                self._warn(
                    "training checkpoint 属于另一份冻结数据；为避免错误续训，"
                    "忽略旧网络/optimizer/POD，从当前数据重新训练。"
                )
                self._resume_payload = {}
                payload = self._resume_payload
                self._optimizer_restore_allowed = False
            elif saved_hash is None and payload:
                self._warn("旧 checkpoint 没有 dataset hash；按维度兼容方式继续。")
            self.dataset_hash = str(current_hash)
            saved = self._saved_pod()
            if saved is not None:
                if saved.thermal_rank != dataset.thermal_rank or saved.current_dimension != dataset.current_dimension:
                    self._warn("旧 POD 维度与当前数据不一致；重新拟合 POD 并放弃旧网络状态。")
                    self._resume_payload = {}
                    self._optimizer_restore_allowed = False
                else:
                    if rank is not None and int(rank) != saved.rank:
                        self._warn(
                            f"当前 pod_rank={rank} 与续训 POD rank={saved.rank} 不同；"
                            "续训优先保留原 POD 坐标。"
                        )
                    self.pod = saved
                    print(f"复用续训 POD：rank={saved.rank}", flush=True)
                    return saved
            print("拟合 Joule tensor POD……0%", flush=True)
            pod = original_fit_pod(
                dataset,
                rank=rank,
                tolerance=tolerance,
                pod_config=pod_config,
            )
            print(f"拟合 Joule tensor POD……100%  rank={pod.rank}", flush=True)
            self.pod = pod
            return pod

        def controlled_build(config, normalizer):
            self._monitor_phase("neural_training")
            self._checkpoint_control()
            model = original_build_network(config, normalizer)
            payload = self._load_payload()
            state = payload.get("network_state")
            if state is None:
                state = self.initial_network_state
            if state is not None:
                saved_config = payload.get("network_config")
                if saved_config is not None and dict(saved_config) != config.to_dict():
                    self._warn(
                        "网络结构已改变；保留 dataset/POD，但从新网络重新训练，"
                        "不加载旧权重和 AdamW 状态。"
                    )
                    state = None
                    self._optimizer_restore_allowed = False
                if state is not None:
                    try:
                        model.load_state_dict(state, strict=True)
                        print("已恢复神经网络权重。", flush=True)
                    except Exception as exc:
                        self._warn(f"旧网络权重不兼容（{exc}）；改为重新训练网络。")
                        self._optimizer_restore_allowed = False

            original_forward = model.forward

            def forward(this, inputs):
                self._monitor_phase("neural_training")
                try:
                    self._checkpoint_control()
                except TrainingStopped:
                    self._save_training_state(this)
                    raise
                return original_forward(inputs)

            model.forward = MethodType(forward, model)
            self.model = model
            return model

        def controlled_adamw(params, *args, **kwargs):
            optimizer = original_adamw(params, *args, **kwargs)
            payload = self._load_payload()
            optimizer_state = payload.get("optimizer_state") if self._optimizer_restore_allowed else None
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                    print("已恢复 AdamW optimizer 状态。", flush=True)
                except Exception as exc:
                    self._warn(f"旧 AdamW 状态不兼容（{exc}）；继续使用新 optimizer。")
            self.optimizer = optimizer
            return optimizer

        pipeline.fixed_research_tensor_factory = controlled_factory(original_fixed_factory)
        pipeline.geometry_research_tensor_factory = controlled_factory(original_geometry_factory)
        pipeline._fit_pod = controlled_fit_pod
        trainer.build_residual_mlp = controlled_build
        torch.optim.AdamW = controlled_adamw
        try:
            yield self
        finally:
            pipeline.fixed_research_tensor_factory = original_fixed_factory
            pipeline.geometry_research_tensor_factory = original_geometry_factory
            pipeline._fit_pod = original_fit_pod
            trainer.build_residual_mlp = original_build_network
            torch.optim.AdamW = original_adamw


__all__ = ["NeuralTrainingRuntime"]
