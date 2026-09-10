import json
import threading
import time
from types import SimpleNamespace

import pytest

from sdfmpneo.training.monitor import JsonlTail, TrainingMonitor, TrainingStopped, write_command


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("training control did not acknowledge in time")
        time.sleep(0.01)


def test_periodic_logs_continue_during_pause_and_stop_works_while_paused(tmp_path):
    control, log = tmp_path / "control.json", tmp_path / "metrics.jsonl"
    write_command(control, "pause")
    advanced, stopped = threading.Event(), threading.Event()
    with TrainingMonitor(log, control, interval=0.02) as monitor:
        def numerical_work():
            try:
                monitor.checkpoint()
                advanced.set()
                write_command(control, "pause")
                monitor.checkpoint()
            except TrainingStopped:
                stopped.set()
        worker = threading.Thread(target=numerical_work)
        worker.start()
        try:
            until(lambda: monitor.data["state"] == "paused")
            until(lambda: len(log.read_text().splitlines()) >= 4)
            assert not advanced.is_set()
            write_command(control, "run")
            until(advanced.is_set)
            until(lambda: monitor.data["state"] == "paused")
            write_command(control, "stop")
            until(stopped.is_set)
            monitor.finish("stopped")
        finally:
            write_command(control, "stop")
            worker.join(timeout=3)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(row["state"] == "paused" for row in rows) >= 2
    assert rows[-1]["state"] == "stopped"


def test_tail_waits_for_complete_unicode_and_handles_truncation(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tail = JsonlTail(path)
    text = json.dumps({"phase": "训练中"}, ensure_ascii=False).encode("utf-8")
    path.write_bytes(text[:-2])
    assert tail.read() == []
    with path.open("ab") as output:
        output.write(text[-2:] + b"\n")
    assert tail.read() == [{"phase": "训练中"}]
    path.write_bytes(b'{"revision":1}\n')
    assert tail.read() == [{"revision": 1}]


def test_monitor_retains_current_network_not_graph(tmp_path):
    with TrainingMonitor(tmp_path / "metrics.jsonl", interval=1) as monitor:
        network = SimpleNamespace(response_nodes=[1, 2])
        monitor.retain(network)
        monitor.record(network, 4.0, 3.0, 16)
        monitor.validation(3.5, 8)
        monitor.record(network, 2.0, 2.0, 24, new_points=True)
        assert monitor.best_network is network
        assert not hasattr(monitor, "best_graph")
        assert monitor.data["nodes"] == 2
        assert monitor.data["validation_max"] is None
        assert monitor.data["collocation_epoch"] == 1


@pytest.mark.parametrize("interval", [0, -1, float("nan")])
def test_invalid_logging_interval_is_rejected(tmp_path, interval):
    with pytest.raises(ValueError):
        TrainingMonitor(tmp_path / "metrics.jsonl", interval=interval)
