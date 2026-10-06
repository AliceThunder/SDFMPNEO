from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time


TERMINAL_STATES = {"completed", "stopped", "failed"}
COMMANDS = {"run", "pause", "stop"}
_CONTEXT_KEYS = {"phase", "message"}


class TrainingStopRequested(Exception):
    pass


def write_control_command(path, command: str):
    command = str(command).strip().lower()
    if command not in COMMANDS:
        raise ValueError(f"unsupported training command: {command}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"command": command}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    control: Path
    metrics: Path
    worker_log: Path
    snapshot: Path

    @classmethod
    def from_root(cls, root):
        root = Path(root)
        return cls(
            root=root,
            control=root / "control.json",
            metrics=root / "metrics.jsonl",
            worker_log=root / "worker.log",
            snapshot=root / "worker.config.json",
        )


class TrainingControl:
    """Cooperative file control plus append-only JSONL telemetry.

    Only stable UI context (phase/message/state) survives across rows. Epoch
    losses and cache counters are event payloads, so heartbeat rows cannot
    duplicate the previous training metric on plots.
    """

    def __init__(
        self,
        session_dir,
        *,
        heartbeat_interval_s: float = 0.5,
        fsync_metrics: bool = False,
    ):
        if float(heartbeat_interval_s) <= 0.0:
            raise ValueError("heartbeat_interval_s must be positive")
        self.paths = SessionPaths.from_root(session_dir)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.fsync_metrics = bool(fsync_metrics)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._file = None
        self._sequence = 0
        self._start = time.monotonic()
        self._state = "running"
        self._context = {
            "phase": "starting",
            "message": "",
        }

    def __enter__(self):
        if not self.paths.control.exists():
            write_control_command(self.paths.control, "run")
        self._file = self.paths.metrics.open("a", encoding="utf-8", buffering=1)
        self.emit(phase="starting", state="running", event="state")
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="vnext-training-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and not isinstance(exc, TrainingStopRequested):
            self.emit(state="failed", message=str(exc), event="state", force=True)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.heartbeat_interval_s))
        if self._file is not None:
            self._file.flush()
            if self.fsync_metrics:
                os.fsync(self._file.fileno())
            self._file.close()
        return False

    def _read_command(self) -> str:
        try:
            payload = json.loads(self.paths.control.read_text(encoding="utf-8"))
            command = str(payload.get("command", "run")).strip().lower()
        except (OSError, ValueError, TypeError):
            return "run"
        return command if command in COMMANDS else "run"

    def _heartbeat(self):
        while not self._stop_event.wait(self.heartbeat_interval_s):
            try:
                command = self._read_command()
                if command == "stop":
                    state = "stopping"
                elif command == "pause":
                    state = "pausing" if self._state != "paused" else "paused"
                else:
                    state = self._state
                self.emit(state=state, event="heartbeat")
            except Exception:
                # Telemetry must never terminate the numerical worker.
                pass

    def emit(self, force=False, **values):
        with self._lock:
            if values.get("state") is not None:
                self._state = str(values["state"])
            for key in _CONTEXT_KEYS:
                if key in values and values[key] is not None:
                    self._context[key] = values[key]
            self._sequence += 1
            row = {
                **self._context,
                **values,
                "state": self._state,
                "sequence": self._sequence,
                "elapsed_s": time.monotonic() - self._start,
                "timestamp": time.time(),
            }
            if self._file is None:
                return row
            self._file.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
            if force or self.fsync_metrics:
                self._file.flush()
            if self.fsync_metrics:
                os.fsync(self._file.fileno())
            return row

    def checkpoint(self, *, phase=None, message=None, **values):
        if phase is not None:
            values["phase"] = phase
        if message is not None:
            values["message"] = message
        if values:
            self.emit(event="progress", **values)
        while True:
            command = self._read_command()
            if command == "stop":
                self.emit(state="stopping", event="state", force=True)
                raise TrainingStopRequested("training stop requested")
            if command == "run":
                if self._state in {"paused", "pausing", "resuming"}:
                    self.emit(state="resuming", event="state", force=True)
                self._state = "running"
                return
            if self._state != "paused":
                self.emit(state="paused", event="state", force=True)
            time.sleep(0.1)

    def finish(self, state="completed", **values):
        state = str(state)
        if state not in TERMINAL_STATES:
            raise ValueError(f"unsupported terminal state: {state}")
        self.emit(state=state, event="state", force=True, **values)