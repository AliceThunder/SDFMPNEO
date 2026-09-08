"""Offscreen UI/IPC tests; the numerical UWPT case is tested separately."""
import json
import os
import time

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt6 import QtWidgets
except ImportError:
    pytest.skip('optional Qt runtime unavailable', allow_module_level=True)
pytest.importorskip('pyqtgraph')
from PyQt6 import QtCore, QtTest
from sdfmpneo.training.qt_monitor import TrainingWindow


def wait_for(app, predicate, timeout=8):
    deadline = time.monotonic()+timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError('Qt/process state transition timed out')
        time.sleep(.01)


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(tmp_path, app):
    # A separate process exercises the exact journal/control protocol with a
    # predictable workload. It is not a physics accuracy reference.
    worker = tmp_path/'worker.py'
    worker.write_text('''
import argparse, json, time
from pathlib import Path
from types import SimpleNamespace
from sdfmpneo.training.monitor import TrainingMonitor, TrainingStopped
p=argparse.ArgumentParser();p.add_argument('--worker-config');args=p.parse_args()
payload=json.loads(Path(args.worker_config).read_text())
folder=Path(payload['session_dir'])
code=0
with TrainingMonitor(folder/'metrics.jsonl', folder/'control.json', interval=.03) as monitor:
    try:
        for i in range(1000):
            monitor.checkpoint()
            deadline=time.monotonic()+.03
            while time.monotonic()<deadline:
                sum(range(1000))
            monitor.record(SimpleNamespace(response_nodes=[0]*(i+1)), 1/(i+1), 2/(i+1), 16)
        monitor.finish('completed')
    except TrainingStopped:
        monitor.finish('stopped')
        code=130
raise SystemExit(code)
''', encoding='utf-8')
    options = dict(max_plot_points=20, refresh_ms=20, compute_threads=1, auto_start=False)
    w = TrainingWindow(worker, {}, tmp_path/'logs', options)
    w.show()
    yield w
    w.close()
    wait_for(app, lambda: not w.isVisible(), timeout=8)
    assert not w._active()
    assert w.reader is None or not w.reader.isRunning()


def click(button):
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)


def test_buttons_control_process_and_curves_are_read_from_file(window, app):
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    click(window.start_button)
    wait_for(app, lambda: len(window.series['revision']) >= 2)
    click(window.pause_button)
    wait_for(app, lambda: window._current_state == 'paused')
    revision = window._last_revision
    tick_count = len(ticks)
    QtTest.QTest.qWait(120)
    assert window._last_revision == revision
    assert len(ticks) > tick_count
    assert window.resume_button.isEnabled()
    rows = [json.loads(line) for line in (window.run_dir/'metrics.jsonl').read_text().splitlines()]
    assert any(row['state'] == 'paused' for row in rows)
    click(window.resume_button)
    wait_for(app, lambda: window._last_revision > revision)
    click(window.stop_button)
    wait_for(app, lambda: not window._active() and window._current_state == 'stopped')
    assert window.start_button.isEnabled()
    assert not window.pause_button.isEnabled()
    # Each restart uses a fresh file and reader, so old metrics cannot leak.
    previous = window.run_dir
    click(window.start_button)
    wait_for(app, lambda: window.run_dir != previous and window._active())
    wait_for(app, lambda: window._last_revision is not None)
    assert window.reader.thread() == app.thread()
    timer.stop()


def test_closing_a_paused_window_stops_and_drains_threads(window, app):
    click(window.start_button)
    wait_for(app, lambda: window._last_revision is not None)
    click(window.pause_button)
    wait_for(app, lambda: window._current_state == 'paused')
    window.close()
    wait_for(app, lambda: not window.isVisible())
    assert not window._active()
    assert not window.reader.isRunning()
