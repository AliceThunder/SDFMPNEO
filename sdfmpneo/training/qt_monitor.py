"""PyQt6 monitor for fixed analytic-network training."""
from __future__ import annotations

import codecs
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import uuid

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from .monitor import JsonlTail, build_resume_history, write_command

STATES = {
    "running": "运行中", "pausing": "正在暂停", "paused": "已暂停",
    "resuming": "正在恢复", "stopping": "正在停止", "stopped": "已停止",
    "completed": "训练完成", "failed": "运行失败", "stalled": "残差停滞，尚未收敛",
}
PHASES = {
    "starting": "启动", "mesh": "生成 UWPT 网格", "assembly": "组装物理模型",
    "thermal_rank_selection": "自动选择热空间阶数",
    "geometry_em_basis": "构建跨几何共享电磁空间", "loading": "加载当前模型",
    "initial_residual": "计算初始物理残差",
    "weight_refinement": "连续优化解析网络参数", "validation": "独立物理残差验证",
    "structure_pruning": "残差验证剪枝", "saving": "保存模型",
}


class LogReader(QtCore.QThread):
    rows = QtCore.pyqtSignal(list)
    console = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, log_path, console_path, interval_ms=300, parent=None):
        super().__init__(parent)
        self.tail = JsonlTail(log_path)
        self.console_path = Path(console_path)
        self.interval = max(20, int(interval_ms)) / 1000.0
        self.wake = threading.Event()
        self.console_offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def stop(self):
        self.requestInterruption()
        self.wake.set()

    def _read_once(self):
        rows = self.tail.read()
        if rows:
            self.rows.emit(rows)
        try:
            with self.console_path.open("rb") as source:
                source.seek(self.console_offset)
                chunk = source.read(65536)
                self.console_offset = source.tell()
        except FileNotFoundError:
            return
        text = self.decoder.decode(chunk)
        if text:
            self.console.emit(text)

    def run(self):
        try:
            while not self.isInterruptionRequested():
                self._read_once()
                self.wake.wait(self.interval)
            self._read_once()
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, runner_path, settings, log_root, options, *, parent=None):
        super().__init__(parent)
        self.runner_path = Path(runner_path).resolve()
        self.settings = settings
        self.log_root = Path(log_root)
        self.options = options
        self.process = None
        self.reader = None
        self.run_dir = None
        self._stop_requested = False
        self._last_revision = None
        self._last_validation = None
        self._limit = max(10, int(options.get("max_plot_points", 4000)))
        self.series = {key: deque(maxlen=self._limit) for key in (
            "revision", "rms", "train_max", "nodes", "training_points",
            "val_revision", "validation_max")}

        self.setWindowTitle("SDF-MPNEO · Fixed Analytic Network")
        self.resize(1100, 820)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        title = QtWidgets.QLabel("UWPT 电磁–热解析代理 · 实时训练")
        title.setStyleSheet("font-size:20px;font-weight:600;padding:6px;")
        layout.addWidget(title)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("启动")
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.resume_button = QtWidgets.QPushButton("恢复")
        self.stop_button = QtWidgets.QPushButton("停止")
        for button in (self.start_button, self.pause_button, self.resume_button, self.stop_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status_label = QtWidgets.QLabel("就绪")
        self.details_label = QtWidgets.QLabel("等待自动热秩选择和物理残差训练")
        layout.addWidget(self.status_label)
        layout.addWidget(self.details_label)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)
        self.curves = {}
        definitions = (
            ("物理残差", True, (("rms", "训练 RMS"), ("train_max", "训练最大值"), ("validation_max", "验证最大值"))),
            ("有效响应通道", False, (("nodes", "有效通道"),)),
            ("训练配点", False, (("training_points", "配点"),)),
        )
        for index, (title_text, logarithmic, curves) in enumerate(definitions):
            plot = pg.PlotWidget(title=title_text, background="w")
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.addLegend()
            plot.setLogMode(y=logarithmic)
            for key, label in curves:
                self.curves[key] = plot.plot(name=label)
            grid.addWidget(plot, index // 2, index % 2)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(600)
        self.output.setMaximumHeight(180)
        layout.addWidget(self.output)

        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(lambda: self.command("pause"))
        self.resume_button.clicked.connect(lambda: self.command("run"))
        self.stop_button.clicked.connect(lambda: self.command("stop"))
        self._set_buttons(False)
        if options.get("auto_start", False):
            QtCore.QTimer.singleShot(0, self.start_training)

    def _active(self):
        return self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning

    def _set_buttons(self, active):
        self.start_button.setEnabled(not active)
        self.pause_button.setEnabled(active and not self._stop_requested)
        self.resume_button.setEnabled(active and not self._stop_requested)
        self.stop_button.setEnabled(active and not self._stop_requested)

    def _resume_model(self):
        try:
            return self.settings["parameters"]["FILES"].get("resume_model")
        except (KeyError, TypeError, AttributeError):
            return None

    def start_training(self):
        if self._active():
            return
        self._stop_requested = False
        for values in self.series.values():
            values.clear()
        for curve in self.curves.values():
            curve.setData([], [])
        self.output.clear()

        history = build_resume_history(
            self.log_root, self._resume_model(),
            root=self.settings.get("root", self.runner_path.parent))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.run_dir = self.log_root / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_command(self.run_dir / "control.json", "run")
        snapshot = self.run_dir / "worker.settings.json"
        snapshot.write_text(json.dumps({
            "settings": self.settings, "session_dir": str(self.run_dir),
            "history_sessions": [str(path) for path in history["sessions"]],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        self.reader = LogReader(
            self.run_dir / "metrics.jsonl", self.run_dir / "worker.log",
            self.options.get("refresh_ms", 300), self)
        self.reader.rows.connect(self.consume)
        self.reader.console.connect(self.output.insertPlainText)
        self.reader.error.connect(lambda text: self.output.appendPlainText("日志错误：" + text))
        self.reader.start()

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-u", str(self.runner_path), "--worker-config", str(snapshot)])
        self.process.setWorkingDirectory(str(self.runner_path.parent))
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            environment.insert(name, str(self.options.get("compute_threads", 1)))
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.setStandardOutputFile(str(self.run_dir / "worker.log"))
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(lambda _: self.status_label.setText("训练进程启动/运行失败"))
        self.process.start()
        self.status_label.setText("训练进程已启动")
        self._set_buttons(True)

    def command(self, command):
        if not self._active():
            return
        write_command(self.run_dir / "control.json", command)
        if command == "stop":
            self._stop_requested = True
            self.status_label.setText("正在停止")
        elif command == "pause":
            self.status_label.setText("正在暂停")
        else:
            self.status_label.setText("正在恢复")
        self._set_buttons(True)

    def consume(self, rows):
        for row in rows:
            revision = row.get("revision")
            if row.get("rms") is not None and revision != self._last_revision:
                self._last_revision = revision
                for key in ("revision", "rms", "train_max", "nodes", "training_points"):
                    self.series[key].append(row.get(key))
            validation = row.get("validation_max")
            if validation is not None and (revision, validation) != self._last_validation:
                self._last_validation = (revision, validation)
                self.series["val_revision"].append(revision)
                self.series["validation_max"].append(validation)
        for key, curve in self.curves.items():
            x = self.series["val_revision" if key == "validation_max" else "revision"]
            y = self.series[key]
            if key in {"rms", "train_max", "validation_max"}:
                y = [max(float(value), 1e-30) for value in y]
            curve.setData(list(x), list(y))
        if rows:
            row = rows[-1]
            phase = PHASES.get(row.get("phase"), row.get("phase", ""))
            state = STATES.get(row.get("state"), row.get("state", ""))
            self.status_label.setText(f"{state} · {phase}")
            self.details_label.setText(
                f"有效响应通道={row.get('nodes', 0)}  RMS={row.get('rms')}  "
                f"训练最大残差={row.get('train_max')}  验证最大残差={row.get('validation_max')}")

    def _finished(self, exit_code, _status):
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(1500)
        self._set_buttons(False)
        if self._stop_requested or exit_code == 130:
            self.status_label.setText("已停止")
        elif exit_code == 0:
            self.status_label.setText("训练完成并达到残差目标")
        else:
            self.status_label.setText(f"训练结束，退出码 {exit_code}")

    def closeEvent(self, event):  # pragma: no cover
        if self._active():
            write_command(self.run_dir / "control.json", "stop")
            self.process.waitForFinished(1500)
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(1000)
        event.accept()


def launch_window(runner_path, settings, log_root, options):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TrainingWindow(runner_path, settings, log_root, options)
    window.show()
    return application.exec()


__all__ = ["TrainingWindow", "launch_window", "PHASES", "STATES"]
