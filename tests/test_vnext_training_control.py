import json
import threading
import time

import pytest

from sdfmpneo_vnext.training_control import (
    TrainingControl,
    TrainingStopRequested,
    write_control_command,
)


def test_control_pause_resume_and_stop(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    control_path = session / "control.json"
    write_control_command(control_path, "run")

    with TrainingControl(session, heartbeat_interval_s=0.02) as control:
        control.checkpoint(phase="port_training", epoch=1)
        write_control_command(control_path, "pause")

        finished = threading.Event()
        errors = []

        def waiter():
            try:
                control.checkpoint(phase="port_training", epoch=1, batch=2)
            except Exception as exc:  # pragma: no cover - diagnostic
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.08)
        assert not finished.is_set()
        write_control_command(control_path, "run")
        assert finished.wait(1.0)
        thread.join()
        assert not errors

        write_control_command(control_path, "stop")
        with pytest.raises(TrainingStopRequested):
            control.checkpoint(phase="port_training", epoch=1, batch=3)

    rows = [
        json.loads(line)
        for line in (session / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states = {row.get("state") for row in rows}
    assert "paused" in states
    assert "stopping" in states


def test_control_command_write_is_atomic_shape(tmp_path):
    path = tmp_path / "control.json"
    write_control_command(path, "pause")
    assert json.loads(path.read_text(encoding="utf-8")) == {"command": "pause"}
    assert not path.with_suffix(".json.tmp").exists()
