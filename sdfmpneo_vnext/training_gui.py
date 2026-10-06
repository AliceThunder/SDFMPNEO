from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
import codecs
import json
from pathlib import Path
import sys
import threading
import uuid

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .training_control import SessionPaths, write_control_command


STATE_TEXT = {
    "idle": "就绪",
    "running": "运行中",
    "pausing": "正在暂停",
    "paused": "已暂停",
    "resuming": "正在恢复",
    "stopping": "正在停止",
    "stopped": "已停止，可续训",
    "completed": "训练完成",
    "failed": "训练失败",
}

PHASE_TEXT = {
    "starting": "启动",
    "teacher_cache": "训练数据缓存 / teacher truth",
    "port_training": "端口 FAST 网络训练",
    "spatial_training": "连续空间损耗网络训练",
    "publishing": "发布 bundle",
    "completed": "完成",
}


def _resolve(root: Path, value) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


class MetricsReader(QtCore.QThread):
    rows = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, path, interval_ms=250, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.interval = max(30, int(interval_ms)) / 1000.0
        self.offset = 0
        self.pending = b""
        self.wake = threading.Event()

    def stop(self):
        self.requestInterruption()
        self.wake.set()

    def _read(self):
        try:
            with self.path.open("rb") as source:
                source.seek(0, 2)
                end = source.tell()
                if end < self.offset:
                    self.offset = 0
                    self.pending = b""
                source.seek(self.offset)
                chunk = source.read(1024 * 1024)
                self.offset = source.tell()
        except FileNotFoundError:
            return
        if not chunk:
            return
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        result = []
        for line in lines:
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeError, ValueError):
                continue
            if isinstance(value, dict):
                result.append(value)
        if result:
            self.rows.emit(result)

    def run(self):
        try:
            while not self.isInterruptionRequested():
                self._read()
                self.wake.wait(self.interval)
            self._read()
        except Exception as exc:  # pragma: no cover - GUI failure path
            self.error.emit(str(exc))


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, runner_path, config, *, parent=None):
        super().__init__(parent)
        self.runner_path = Path(runner_path).resolve()
        self.config = deepcopy(config)
        self.root = Path(self.config["ROOT"])
        self.gui_cfg = dict(self.config.get("GUI", {}) or {})
        self.log_root = _resolve(self.root, self.config["FILES"]["log_dir"])
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.reader = None
        self.session = None
        self.paths = None
        self._state = "idle"
        self._stop_requested = False
        self.max_points = max(50, int(self.gui_cfg.get("max_plot_points", 4000)))
        self.series = {
            key: deque(maxlen=self.max_points)
            for key in (
                "port_epoch",
                "port_train",
                "port_val",
                "spatial_epoch",
                "spatial_train",
                "spatial_val",
                "cache_x",
                "cache_y",
            )
        }
        self.curves = {}
        self._build_ui()
        self._set_buttons(False)
        if bool(self.gui_cfg.get("auto_start", False)):
            QtCore.QTimer.singleShot(0, self.start_training)

    def _build_ui(self):
        self.setWindowTitle(str(self.gui_cfg.get("title", "SDF-MPNEO vNext 训练控制台")))
        self.resize(
            int(self.gui_cfg.get("width", 1240)),
            int(self.gui_cfg.get("height", 860)),
        )
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("SDF-MPNEO vNext · 缓存 / CUDA batch / 可恢复训练")
        title.setStyleSheet("font-size:20px;font-weight:600;padding:6px")
        layout.addWidget(title)

        controls = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("启动 / 续训")
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.resume_button = QtWidgets.QPushButton("恢复")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.fresh_checkbox = QtWidgets.QCheckBox("忽略模型 checkpoint 重新训练")
        self.fresh_checkbox.setToolTip("只清除本次模型续训语义，不删除 teacher 数据缓存")
        for button in (
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ):
            button.setMinimumHeight(36)
            controls.addWidget(button)
        controls.addWidget(self.fresh_checkbox)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status = QtWidgets.QLabel("就绪")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            "QLabel{background:#f8fafc;border:1px solid #dbe3ea;"
            "border-radius:6px;padding:8px}"
        )
        layout.addWidget(self.status)
        self.path_label = QtWidgets.QLabel(str(self.log_root))
        self.path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        plots = QtWidgets.QGridLayout()
        layout.addLayout(plots, 1)
        port_plot = pg.PlotWidget(title="Port surrogate loss", background="w")
        port_plot.setLabel("bottom", "epoch")
        port_plot.setLabel("left", "loss")
        port_plot.setLogMode(y=True)
        port_plot.showGrid(x=True, y=True, alpha=0.2)
        port_plot.addLegend()
        self.curves["port_train"] = port_plot.plot(
            name="train", pen=pg.mkPen("#2563eb", width=2)
        )
        self.curves["port_val"] = port_plot.plot(
            name="validation", pen=pg.mkPen("#16a34a", width=2)
        )
        plots.addWidget(port_plot, 0, 0)

        spatial_plot = pg.PlotWidget(title="Spatial surrogate loss", background="w")
        spatial_plot.setLabel("bottom", "epoch")
        spatial_plot.setLabel("left", "loss")
        spatial_plot.setLogMode(y=True)
        spatial_plot.showGrid(x=True, y=True, alpha=0.2)
        spatial_plot.addLegend()
        self.curves["spatial_train"] = spatial_plot.plot(
            name="train", pen=pg.mkPen("#7c3aed", width=2)
        )
        self.curves["spatial_val"] = spatial_plot.plot(
            name="validation", pen=pg.mkPen("#ea580c", width=2)
        )
        plots.addWidget(spatial_plot, 0, 1)

        cache_plot = pg.PlotWidget(title="Teacher cache fill", background="w")
        cache_plot.setLabel("bottom", "update")
        cache_plot.setLabel("left", "cached samples")
        cache_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curves["cache"] = cache_plot.plot(
            name="cached", pen=pg.mkPen("#0891b2", width=2), symbol="o", symbolSize=4
        )
        plots.addWidget(cache_plot, 1, 0, 1, 2)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(int(self.gui_cfg.get("console_blocks", 1000)))
        self.output.setMaximumHeight(int(self.gui_cfg.get("console_height", 190)))
        layout.addWidget(self.output)

        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(lambda: self.command("pause"))
        self.resume_button.clicked.connect(lambda: self.command("run"))
        self.stop_button.clicked.connect(lambda: self.command("stop"))

    def _active(self):
        return (
            self.process is not None
            and self.process.state() != QtCore.QProcess.ProcessState.NotRunning
        )

    def _set_buttons(self, active):
        self.start_button.setEnabled(not active)
        self.pause_button.setEnabled(active and not self._stop_requested)
        self.resume_button.setEnabled(active and not self._stop_requested)
        self.stop_button.setEnabled(active and not self._stop_requested)
        self.fresh_checkbox.setEnabled(not active)

    def _new_session(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_root / f"{stamp}_{uuid.uuid4().hex[:8]}"

    def _snapshot(self, config):
        self.paths.snapshot.write_text(
            json.dumps(
                {"config": config, "session_dir": str(self.session)},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

    def start_training(self):
        if self._active():
            return
        if self.reader is not None and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait(1500)
        self._stop_requested = False
        self.session = self._new_session()
        self.session.mkdir(parents=True, exist_ok=True)
        self.paths = SessionPaths.from_root(self.session)
        write_control_command(self.paths.control, "run")
        worker_config = deepcopy(self.config)
        if self.fresh_checkbox.isChecked():
            worker_config.setdefault("TRAINING", {})["resume"] = False
        self._snapshot(worker_config)
        self.path_label.setText(f"本次会话：{self.session}")
        self.status.setText("正在启动训练 worker……")
        self.output.clear()
        self._clear_plots()

        self.reader = MetricsReader(
            self.paths.metrics,
            interval_ms=int(self.gui_cfg.get("refresh_ms", 250)),
            parent=self,
        )
        self.reader.rows.connect(self.consume_rows)
        self.reader.error.connect(
            lambda text: self.output.appendPlainText("metrics reader: " + text)
        )
        self.reader.start()

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(
            ["-u", str(self.runner_path), "--worker-config", str(self.paths.snapshot)]
        )
        self.process.setWorkingDirectory(str(self.runner_path.parent))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        process_env = dict(self.config.get("RUNTIME", {}).get("environment", {}) or {})
        for name, value in process_env.items():
            env.insert(str(name), str(value))
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._consume_console)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._set_buttons(True)
        self.process.start()

    def command(self, command):
        if not self._active() or self.paths is None:
            return
        write_control_command(self.paths.control, command)
        if command == "pause":
            self.status.setText("已请求暂停；将在当前安全 batch / teacher chunk 后暂停。")
        elif command == "run":
            self.status.setText("已请求恢复。")
        elif command == "stop":
            self._stop_requested = True
            self.status.setText("已请求停止；正在保存可恢复 checkpoint。")
        self._set_buttons(True)

    def _consume_console(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        if text:
            self.output.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            self.output.insertPlainText(text)
            self.output.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def consume_rows(self, rows):
        for row in rows:
            state = str(row.get("state", self._state))
            phase = str(row.get("phase", "starting"))
            self._state = state
            message = str(row.get("message", ""))
            detail = f"状态：{STATE_TEXT.get(state, state)}  |  阶段：{PHASE_TEXT.get(phase, phase)}"
            if row.get("epoch") is not None:
                detail += f"  |  epoch {row.get('epoch')}/{row.get('epochs', '?')}"
            if row.get("batch") is not None:
                detail += f"  |  batch {row.get('batch')}/{row.get('batches', '?')}"
            if row.get("cached") is not None and row.get("requested") is not None:
                detail += f"  |  cache {row['cached']}/{row['requested']}"
            if message:
                detail += "\n" + message
            self.status.setText(detail)

            if phase == "port_training" and row.get("train_loss") is not None:
                epoch = float(row["epoch"])
                self._append("port_epoch", epoch)
                self._append("port_train", float(row["train_loss"]))
                if row.get("validation_loss") is not None:
                    self._append("port_val", float(row["validation_loss"]))
                else:
                    self._append("port_val", float("nan"))
            elif phase == "spatial_training" and row.get("train_loss") is not None:
                epoch = float(row["epoch"])
                self._append("spatial_epoch", epoch)
                self._append("spatial_train", float(row["train_loss"]))
                if row.get("validation_loss") is not None:
                    self._append("spatial_val", float(row["validation_loss"]))
                else:
                    self._append("spatial_val", float("nan"))
            elif phase == "teacher_cache" and row.get("cached") is not None:
                self._append("cache_x", len(self.series["cache_x"]) + 1)
                self._append("cache_y", float(row["cached"]))
        self._refresh_plots()

    def _append(self, key, value):
        self.series[key].append(value)

    def _refresh_plots(self):
        self.curves["port_train"].setData(
            list(self.series["port_epoch"]), list(self.series["port_train"])
        )
        self.curves["port_val"].setData(
            list(self.series["port_epoch"]), list(self.series["port_val"])
        )
        self.curves["spatial_train"].setData(
            list(self.series["spatial_epoch"]), list(self.series["spatial_train"])
        )
        self.curves["spatial_val"].setData(
            list(self.series["spatial_epoch"]), list(self.series["spatial_val"])
        )
        self.curves["cache"].setData(
            list(self.series["cache_x"]), list(self.series["cache_y"])
        )

    def _clear_plots(self):
        for values in self.series.values():
            values.clear()
        for curve in self.curves.values():
            curve.setData([], [])

    def _process_finished(self, exit_code, exit_status):
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(1500)
        self._consume_console()
        self._set_buttons(False)
        if int(exit_code) == 0:
            self.status.setText("训练完成，bundle 已发布。")
        elif int(exit_code) == 2:
            self.status.setText("训练已安全停止。再次点击“启动 / 续训”即可从 checkpoint 继续。")
        else:
            self.status.setText(f"worker 异常退出，exit_code={exit_code}；日志：{self.session}")

    def _process_error(self, error):
        self.output.appendPlainText(f"QProcess error: {error}")

    def closeEvent(self, event):
        if self._active():
            QtWidgets.QMessageBox.information(
                self,
                "训练仍在运行",
                "请先点击“停止”，等待可恢复 checkpoint 保存完成后再关闭窗口。",
            )
            event.ignore()
            return
        if self.reader is not None and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait(1000)
        event.accept()


def run_training_gui(runner_path, config) -> int:
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication(sys.argv[:1])
    window = TrainingWindow(runner_path, config)
    window.show()
    if owns_application:
        return int(application.exec())
    return 0
