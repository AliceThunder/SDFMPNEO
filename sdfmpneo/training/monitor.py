"""File-based training telemetry and cooperative control, independent of Qt."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path


class TrainingStopped(Exception):
    """A user stop requested at a safe numerical boundary."""


def write_command(path, command):
    if command not in {"run", "pause", "stop"}:
        raise ValueError("unknown training command")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"command": command}), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    """Read complete dictionary rows from a JSONL journal, ignoring bad tail lines."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeError):
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _resolved_path(value, root):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve(strict=False)


def _session_matches_model(session_dir, target, root):
    rows = read_jsonl(Path(session_dir) / "metrics.jsonl")
    for row in reversed(rows):
        for key in ("checkpoint", "model"):
            value = row.get(key)
            if value is None:
                continue
            try:
                if _resolved_path(value, root) == target:
                    return True
            except (OSError, TypeError, ValueError):
                continue
    return False


def find_resume_sessions(log_root, resume_model, *, root=None):
    """Find the newest journal that produced ``resume_model`` and its lineage.

    Old sessions did not store lineage; those are still recognized from their
    final ``checkpoint``/``model`` journal fields. New resumed sessions record
    ``history_sessions`` in worker.settings.json so repeated stop/resume cycles
    recover the complete plotting/log history rather than only the last segment.
    """
    if resume_model is None:
        return []
    root = Path(root or ".").resolve(strict=False)
    target = _resolved_path(resume_model, root)
    log_root = Path(log_root)
    try:
        directories = [path for path in log_root.iterdir() if path.is_dir()]
    except FileNotFoundError:
        return []
    matches = []
    for directory in directories:
        if _session_matches_model(directory, target, root):
            try:
                stamp = directory.stat().st_mtime_ns
            except OSError:
                stamp = 0
            matches.append((stamp, directory))
    if not matches:
        return []
    previous = max(matches, key=lambda item: item[0])[1]
    lineage = []
    try:
        payload = json.loads((previous / "worker.settings.json").read_text(encoding="utf-8"))
        stored = payload.get("history_sessions", [])
        if isinstance(stored, list):
            for value in stored:
                path = Path(value)
                if path.is_dir():
                    lineage.append(path)
    except (OSError, ValueError, TypeError):
        pass
    lineage.append(previous)
    result, seen = [], set()
    for path in lineage:
        key = str(path.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def build_resume_history(log_root, resume_model, *, root=None):
    """Load prior sessions and make their local counters globally continuous."""
    sessions = find_resume_sessions(log_root, resume_model, root=root)
    revision_offset = 0
    elapsed_offset = 0.0
    collocation_offset = 0
    combined = []
    for session in sessions:
        rows = read_jsonl(session / "metrics.jsonl")
        local_revision = 0
        local_elapsed = 0.0
        local_collocation = 0
        for row in rows:
            revision = int(row.get("revision") or 0)
            elapsed = float(row.get("elapsed_s") or 0.0)
            collocation = int(row.get("collocation_epoch") or 0)
            local_revision = max(local_revision, revision)
            local_elapsed = max(local_elapsed, elapsed)
            local_collocation = max(local_collocation, collocation)
            adjusted = dict(row)
            adjusted["revision"] = revision_offset + revision
            adjusted["elapsed_s"] = elapsed_offset + elapsed
            adjusted["collocation_epoch"] = collocation_offset + collocation
            combined.append(adjusted)
        revision_offset += local_revision
        elapsed_offset += local_elapsed
        collocation_offset += local_collocation
    return {
        "sessions": sessions,
        "rows": combined,
        "revision_offset": revision_offset,
        "elapsed_offset_s": elapsed_offset,
        "collocation_offset": collocation_offset,
    }


class TrainingMonitor:
    """One writer thread emits periodic JSONL snapshots, even during assembly.

    The numerical thread calls checkpoint() between indivisible operations.
    Pause/stop acknowledgements therefore describe actual numerical state;
    a long sparse solve may leave the request pending until it returns.
    """
    def __init__(self, log_path, control_path=None, *, interval=1.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("log interval must be finite and positive")
        self.log_path = Path(log_path)
        self.control_path = None if control_path is None else Path(control_path)
        self.interval = interval
        self.best_graph = None
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._command = "run"
        self._start = time.monotonic()
        self._sequence = 0
        self._thread = None
        self._file = None
        self._error = None
        self.data = {"state": "running", "phase": "starting", "revision": 0,
                     "collocation_epoch": 0, "nodes": 0, "mse": None,
                     "rms": None, "train_max": None, "validation_max": None,
                     "training_points": None, "validation_points": None}

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._read_command()
        self._write()
        self._thread = threading.Thread(target=self._heartbeat, name="training-journal", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.finish("stopped" if isinstance(exc, TrainingStopped) else "failed", message=str(exc))
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        self._write()
        self._file.close()
        if self._error is not None and exc is None:
            raise RuntimeError("training log could not be written") from self._error

    def _write(self):
        with self._lock:
            self._sequence += 1
            row = {**self.data, "sequence": self._sequence,
                   "elapsed_s": time.monotonic()-self._start, "timestamp": time.time()}
            self._file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            self._file.flush()

    def _read_command(self):
        if self.control_path is None:
            return
        try:
            command = json.loads(self.control_path.read_text(encoding="utf-8"))["command"]
        except (OSError, ValueError, KeyError):
            return
        if command not in {"run", "pause", "stop"}:
            return
        with self._lock:
            if self._command == "stop":
                return
            self._command = command
            if self.data["state"] in {"completed", "stopped", "failed", "budget_exhausted", "stalled"}:
                return
            if command == "stop":
                self.data["state"] = "stopping"
            elif command == "pause" and self.data["state"] != "paused":
                self.data["state"] = "pausing"
            elif command == "run" and self.data["state"] in {"paused", "pausing"}:
                self.data["state"] = "resuming"

    def _heartbeat(self):
        last = time.monotonic()
        while not self._done.wait(min(.1, self.interval)):
            try:
                self._read_command()
                if time.monotonic()-last >= self.interval:
                    self._write()
                    last = time.monotonic()
            except Exception as exc:
                self._error = exc
                return

    def checkpoint(self):
        while True:
            if self._error is not None:
                raise RuntimeError("training log could not be written") from self._error
            self._read_command()
            with self._lock:
                command = self._command
                if command == "stop":
                    raise TrainingStopped("用户停止训练")
                if command == "run":
                    self.data["state"] = "running"
                    return
                self.data["state"] = "paused"
            self._done.wait(.05)

    def phase(self, name, *, check=True):
        with self._lock:
            self.data["phase"] = name
        if check:
            self.checkpoint()

    def retain(self, graph):
        self.best_graph = graph
        with self._lock:
            self.data["nodes"] = len(graph.response_nodes)

    def record(self, graph, objective, maximum, count, *, new_points=False):
        """Publish only accepted iterates; trial steps never become checkpoints."""
        with self._lock:
            self.best_graph = graph
            self.data.update(nodes=len(graph.response_nodes), mse=float(objective),
                             rms=math.sqrt(objective), train_max=float(maximum),
                             training_points=int(count), validation_max=None)
            self.data["revision"] += 1
            if new_points:
                self.data["collocation_epoch"] += 1
        self._write()

    def validation(self, maximum, count):
        with self._lock:
            self.data.update(validation_max=float(maximum), validation_points=int(count))
        self._write()

    def finish(self, state, **details):
        with self._lock:
            self.data.update(state=state, **details)


class JsonlTail:
    """Incremental reader; retains incomplete UTF-8/JSON lines until flushed."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""

    def read(self):
        try:
            with self.path.open("rb") as source:
                if source.seek(0, 2) < self.offset:
                    self.offset, self.pending = 0, b""
                source.seek(self.offset)
                chunk = source.read(1024*1024)
                self.offset = source.tell()
        except FileNotFoundError:
            return []
        lines = (self.pending+chunk).split(b"\n")
        self.pending = lines.pop()
        result = []
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    result.append(row)
            except (ValueError, UnicodeError):
                continue
        return result
