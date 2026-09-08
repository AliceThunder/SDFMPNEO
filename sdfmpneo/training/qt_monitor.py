"""PyQt6 UI: numerical process -> JSONL file -> reader QThread -> Qt plots."""
from __future__ import annotations

import codecs
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import uuid

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .monitor import JsonlTail, write_command


STATES = {
    "running": "运行中", "pausing": "正在暂停（等待当前运算结束）", "paused": "已暂停",
    "resuming": "正在恢复", "stopping": "正在停止并保存", "stopped": "已停止",
    "completed": "训练完成", "failed": "运行失败", "budget_exhausted": "预算耗尽，尚未收敛",
    "stalled": "残差停滞，尚未收敛",
}
PHASES = {
    "starting": "启动", "mesh": "生成 UWPT 网格", "assembly": "组装与电磁降阶",
    "loading": "加载已有模型", "initial_residual": "计算初始残差",
    "quadratic_seed": "构造二次热源响应", "weight_refinement": "联合更新权重",
    "candidate_search": "搜索响应节点", "validation": "独立残差检查",
    "saving": "保存模型", "geometry_em_basis": "构建跨几何共享电磁空间",
    "geometry_seed": "构造跨几何物理初始网络",
}


class LogReader(QtCore.QThread):
    rows = QtCore.pyqtSignal(list)
    console = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, log_path, console_path, interval_ms=300, parent=None):
        super().__init__(parent)
        self.tail = JsonlTail(log_path)
        self.console_path = Path(console_path)
        self.interval = max(20, interval_ms)/1000
        self.wake = threading.Event()
        self.console_offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def stop(self):
        self.requestInterruption()
        self.wake.set()

    def _read(self):
        rows = self.tail.read()
        if rows:
            self.rows.emit(rows)
        try:
            with self.console_path.open("rb") as source:
                source.seek(self.console_offset)
                chunk = source.read(65536)
                self.console_offset = source.tell()
            text = self.decoder.decode(chunk)
            if text:
                self.console.emit(text)
        except FileNotFoundError:
            pass

    def run(self):
        try:
            while not self.isInterruptionRequested():
                self._read()
                self.wake.wait(self.interval)
            self._read()  # Drain the final newline after worker termination.
        except Exception as exc:
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
        self._closing = False
        self._restart = False
        self._terminal = None
        self._exit_code = None
        self._stop_requested = False
        self._last_revision = None
        self._last_validation = None
        self._current_state = "idle"
        self._current_phase = ""
        self._limit = max(10, int(options["max_plot_points"]))
        self.series = {key: deque(maxlen=self._limit) for key in
                       ("revision", "mse", "rms", "train_max", "nodes", "training_points",
                        "val_revision", "validation_max")}
        self.setWindowTitle("SDF-MPNEO · UWPT 电磁–热训练")
        available = set(QtGui.QFontDatabase.families())
        for family in ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
            if family in available:
                self.setFont(QtGui.QFont(family, 10))
                break
        self.resize(1180, 820)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        title = QtWidgets.QLabel("UWPT 电磁–热代理模型 · 实时训练")
        title.setStyleSheet("font-size: 22px; font-weight: 600; padding: 8px;")
        layout.addWidget(title)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("启动")
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.resume_button = QtWidgets.QPushButton("恢复")
        self.stop_button = QtWidgets.QPushButton("停止")
        for button in (self.start_button, self.pause_button, self.resume_button, self.stop_button):
            button.setMinimumHeight(36)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status_label = QtWidgets.QLabel("就绪：点击启动开始 UWPT 训练")
        self.status_label.setWordWrap(True)
        self.details_label = QtWidgets.QLabel("训练指标将从日志文件读取；组装期间暂不产生残差曲线。")
        self.path_label = QtWidgets.QLabel(str(self.log_root))
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self.details_label)
        layout.addWidget(self.path_label)
        plots = QtWidgets.QGridLayout()
        layout.addLayout(plots, 1)
        self.curves = {}
        definitions = [
            ("Mean squared physical residual", "MSE", True, [("mse", "MSE", "#2563eb")]),
            ("Physical residual norms", "Residual", True,
             [("rms", "Train RMS", "#2563eb"), ("train_max", "Train maximum", "#d97706"),
              ("validation_max", "Validation maximum", "#16a34a")]),
            ("Accepted response neurons", "Nodes", False, [("nodes", "Nodes", "#7c3aed")]),
            ("Collocation refinement", "Points", False, [("training_points", "Training points", "#0891b2")]),
        ]
        for index, (title_text, units, logarithmic, curves) in enumerate(definitions):
            plot = pg.PlotWidget(title=title_text, background="w")
            plot.setTitle(title_text, color="#1f2937", size="11pt")
            plot.setLabel("bottom", "Accepted update", color="#374151")
            plot.setLabel("left", units, color="#374151")
            for side in ("left", "bottom"):
                plot.getAxis(side).setTextPen("#374151")
                plot.getAxis(side).setPen("#9ca3af")
                plot.getAxis(side).enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=.2)
            plot.addLegend(labelTextColor="#374151")
            plot.setLogMode(y=logarithmic)
            for key, label, color in curves:
                self.curves[key] = plot.plot(name=label, pen=pg.mkPen(color, width=2),
                                             symbol="o", symbolSize=4, symbolBrush=color, symbolPen=color)
            plots.addWidget(plot, index//2, index % 2)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(500)
        self.output.setMaximumHeight(140)
        layout.addWidget(self.output)
        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(lambda: self.command("pause"))
        self.resume_button.clicked.connect(lambda: self.command("run"))
        self.stop_button.clicked.connect(lambda: self.command("stop"))
        self._buttons(False)
        if options.get("auto_start", False):
            QtCore.QTimer.singleShot(0, self.start_training)

    def _active(self):
        return self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning

    def _buttons(self, active, state="running"):
        if active and self._stop_requested:
            state = "stopping"
        self.start_button.setEnabled(not active and not self._closing)
        self.pause_button.setEnabled(active and state in {"running", "resuming"})
        self.resume_button.setEnabled(active and state in {"paused", "pausing"})
        self.stop_button.setEnabled(active and state != "stopping")

    def start_training(self):
        if self._active() or self._closing:
            return
        if self.reader is not None and self.reader.isRunning():
            self._restart = True
            self.start_button.setEnabled(False)
            self.reader.stop()
            return
        self._launch()

    def _launch(self):
        try:
            self._terminal = None
            self._exit_code = None
            self._stop_requested = False
            self._last_revision = self._last_validation = None
            for values in self.series.values():
                values.clear()
            for curve in self.curves.values():
                curve.setData([], [])
            self.output.clear()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
            self.run_dir = self.log_root/stamp
            self.run_dir.mkdir(parents=True)
            control = self.run_dir/"control.json"
            write_command(control, "run")
            worker_settings = {"settings": self.settings, "session_dir": str(self.run_dir)}
            snapshot = self.run_dir/"worker.settings.json"
            snapshot.write_text(json.dumps(worker_settings, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.reader is not None:
                self.reader.deleteLater()
            self.reader = LogReader(self.run_dir/"metrics.jsonl", self.run_dir/"worker.log",
                                    self.options["refresh_ms"], self)
            self.reader.rows.connect(self.consume)
            self.reader.console.connect(self.output.insertPlainText)
            self.reader.error.connect(self._reader_error)
            self.reader.finished.connect(self._reader_finished)
            self.reader.start()
            if self.process is not None:
                self.process.deleteLater()
            self.process = QtCore.QProcess(self)
            self.process.setProgram(sys.executable)
            self.process.setArguments(["-u", str(self.runner_path), "--worker-config", str(snapshot)])
            self.process.setWorkingDirectory(str(self.runner_path.parent))
            environment = QtCore.QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONIOENCODING", "utf-8")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                environment.insert(name, str(self.options["compute_threads"]))
            self.process.setProcessEnvironment(environment)
            self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.process.setStandardOutputFile(str(self.run_dir/"worker.log"))
            self.process.finished.connect(self._process_finished)
            self.process.errorOccurred.connect(self._process_error)
            self._current_state = "running"
            self.status_label.setText("启动 UWPT 训练进程……")
            self.path_label.setText(f"本次日志：{self.run_dir}")
            self._buttons(True)
            self.process.start()
        except Exception as exc:
            self._terminal = f"启动失败：{exc}"
            self.status_label.setText(self._terminal)
            self._buttons(False)
            if self.reader is not None:
                self.reader.stop()

    def command(self, command):
        if not self._active() or (self._stop_requested and command != "stop"):
            return
        try:
            write_command(self.run_dir/"control.json", command)
        except OSError as exc:
            self.output.appendPlainText(f"控制指令写入失败：{exc}")
            return
        if command == "stop":
            self._stop_requested = True
        self._current_state = {"pause": "pausing", "run": "resuming", "stop": "stopping"}[command]
        self.status_label.setText(STATES[self._current_state])
        self._buttons(True, self._current_state)

    @QtCore.pyqtSlot(list)
    def consume(self, rows):
        for row in rows:
            revision = row.get("revision")
            if row.get("mse") is not None and revision != self._last_revision:
                self._last_revision = revision
                for key in ("revision", "mse", "rms", "train_max", "nodes", "training_points"):
                    self.series[key].append(row[key])
            validation = row.get("validation_max")
            if validation is not None and (revision, validation) != self._last_validation:
                self._last_validation = revision, validation
                self.series["val_revision"].append(revision)
                self.series["validation_max"].append(validation)
        for key, curve in self.curves.items():
            x = self.series["val_revision" if key == "validation_max" else "revision"]
            y = self.series[key]
            if key in {"mse", "rms", "train_max", "validation_max"}:
                y = [max(value, 1e-30) for value in y]
            curve.setData(list(x), list(y))
        if rows:
            row = rows[-1]
            self._current_state = row.get("state", "running")
            self._current_phase = row.get("phase", "")
            state = STATES.get(self._current_state, self._current_state)
            if self._active() and self._stop_requested:
                state = STATES["stopping"]
            if self._exit_code is not None:
                self._set_terminal()
            phase = PHASES.get(self._current_phase, self._current_phase)
            self.status_label.setText(self._terminal or f"{state} · {phase}")
            validation_text = (f"{row['validation_max']:.4g}"
                               if row.get('validation_max') is not None else "尚未检查")
            self.details_label.setText(
                f"累计用时 {row.get('elapsed_s', 0):.1f} s  |  更新 {row.get('revision', 0)}  |  "
                f"节点 {row.get('nodes', 0)}  |  配点轮次 {row.get('collocation_epoch', 0)}  |  "
                f"独立残差 {validation_text}"
            )
            self._buttons(self._active(), self._current_state)

    def _process_finished(self, code, exit_status):
        self._exit_code = code
        self._set_terminal()
        if exit_status == QtCore.QProcess.ExitStatus.CrashExit:
            self._terminal = "训练进程异常退出，请查看日志"
            self._exit_code = None
        self.status_label.setText(self._terminal)
        self._buttons(False)
        if self._closing:
            self._finish_close()

    def _set_terminal(self):
        # A bare exit code 2 may also be an argparse/startup error. Claim
        # convergence/budget completion only when the worker journal agrees.
        if self._exit_code == 0 and self._current_state == "completed":
            self._terminal = "训练完成，模型已保存"
        elif self._exit_code == 2 and self._current_state in {"budget_exhausted", "stalled"}:
            self._terminal = "尚未收敛，当前模型和报告已保存"
        elif self._exit_code == 130 and self._current_state == "stopped":
            self._terminal = "已停止；有效检查点信息见日志"
        else:
            self._terminal = f"训练进程已退出（退出码 {self._exit_code}），请查看日志"

    def _process_error(self, error):
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self._terminal = "无法启动训练进程："+self.process.errorString()
            self.status_label.setText(self._terminal)
            self._buttons(False)
            if self._closing:
                self._finish_close()

    def _reader_error(self, message):
        self.output.appendPlainText("日志读取失败："+message)

    def _reader_finished(self):
        if self._closing:
            self.close()
        elif self._restart:
            self._restart = False
            self._launch()

    def _finish_close(self):
        if self.reader is not None and self.reader.isRunning():
            self.reader.stop()
        else:
            self.close()

    def closeEvent(self, event):
        self._closing = True
        if self._active():
            self.command("stop")
            event.ignore()
        elif self.reader is not None and self.reader.isRunning():
            self.reader.stop()
            event.ignore()
        else:
            event.accept()


def launch_window(runner_path, settings, log_root, options):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([str(runner_path)])
    window = TrainingWindow(runner_path, settings, log_root, options)
    window.show()
    return app.exec()
