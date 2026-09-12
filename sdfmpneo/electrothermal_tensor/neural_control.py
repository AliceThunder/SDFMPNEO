"""Bridge the existing file-based training monitor to the neural ROM pipeline.

The PyQt window already controls the worker through ``control.json``.  This
module keeps that UI/control layer independent of the neural mathematics while
making snapshot generation and AdamW training cooperatively pausable/stoppable.
A stop during neural training writes a continuation checkpoint containing the
POD coordinates, network weights and AdamW state so a later run resumes in the
same output coordinates.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import threading

import numpy as np


class NeuralTrainingRuntime:
    """Cooperative pause/stop plus warm continuation for one worker process."""

    def __init__(
        self,
        monitor,
        checkpoint_path,
        *,
        initial_network_state=None,
        initial_pod=None,
    ):
        self.monitor = monitor
        self.checkpoint_path = Path(checkpoint_path)
        self.initial_network_state = initial_network_state
        self.initial_pod = initial_pod
        self.model = None
        self.optimizer = None
        self.pod = None
        self._resume_payload = None
        self._snapshot_count = 0
        self._counter_lock = threading.Lock()

    def _load_payload(self):
        if self._resume_payload is not None:
            return self._resume_payload
        if not self.checkpoint_path.is_file():
            self._resume_payload = {}
            return self._resume_payload
        try:
            import torch
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch without weights_only
            import torch
            payload = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or int(payload.get("format_version", -1)) != 2:
            raise ValueError(
                f"unsupported neural training checkpoint: {self.checkpoint_path}"
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
        if self.monitor is None:
            return
        with self._counter_lock:
            self._snapshot_count += 1
            value = self._snapshot_count
        with self.monitor._lock:
            self.monitor.data["phase"] = "neural_snapshots"
            self.monitor.data["training_points"] = value

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
            total_centered_energy=float(value["total_centered_energy"]),
        )

    def _saved_pod(self):
        payload = self._load_payload()
        pod = self._pod_from_payload(payload.get("pod"))
        return self.initial_pod if pod is None else pod

    def _save_training_state(self, model) -> None:
        """Persist POD coordinates, network weights and AdamW momentum."""
        try:
            import torch
        except ImportError:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
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
            saved = self._saved_pod()
            if saved is not None:
                if saved.thermal_rank != dataset.thermal_rank or saved.current_dimension != dataset.current_dimension:
                    raise ValueError("saved POD dimensions do not match the frozen dataset")
                if rank is not None and int(rank) != saved.rank:
                    raise ValueError(
                        "continuation checkpoint uses a different POD rank; restore the previous "
                        "TRAINING['pod_rank'] or delete the training checkpoint"
                    )
                self.pod = saved
                return saved
            pod = original_fit_pod(
                dataset,
                rank=rank,
                tolerance=tolerance,
                pod_config=pod_config,
            )
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
                    raise ValueError(
                        "neural continuation checkpoint uses a different network architecture; "
                        "restore the previous TRAINING['network'] or delete the checkpoint"
                    )
                try:
                    model.load_state_dict(state, strict=True)
                except Exception as exc:
                    raise ValueError(
                        "resume model/checkpoint network architecture does not match current training"
                    ) from exc

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
            optimizer_state = payload.get("optimizer_state")
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except Exception as exc:
                    raise ValueError(
                        "saved AdamW state is incompatible with the current network"
                    ) from exc
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
