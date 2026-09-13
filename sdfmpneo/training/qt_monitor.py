"""PyQt6 Chinese training monitor for both neural and legacy training."""
from __future__ import annotations

import codecs
import json
import re
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .monitor import JsonlTail, build_resume_history, write_command

STATES = {
    "running": "运行中",
    "pausing": "正在暂停（等待当前运算结束）",
    "paused": "已暂停",
    "resuming": "正在恢复",
    "stopping": "正在停止并保存",
    "stopped": "已停止",
    "completed": "训练完成",
    "failed": "运行失败",
    "budget_exhausted": "预算耗尽，尚未收敛",
    "stalled": "残差停滞，尚未收敛",
}

PHASES = {
    "starting": "启动",
    "mesh": "生成 UWPT 网格",
    "assembly": "组装与电磁降阶",
    "loading": "加载已有模型",
    "geometry_em_basis": "构建跨几何共享电磁空间",
    "geometry_seed": "构造跨几何物理初始网络",
    "neural_snapshots": "生成 Joule tensor 标签",
    "neural_pod": "拟合 Joule tensor POD",
    "neural_training": "训练神经网络",
    "saving": "保存模型",
    # legacy phases kept for compatibility
    "initial_residual": "计算初始残差",
    "quadratic_seed": "构造二次热源响应",
    "weight_refinement": "联合更新响应权重",
    "candidate_search": "搜索响应神经元",
    "validation": "独立残差检查",
}

_NEURAL_LINE = re.compile(
    r"训练神经网络……\s*([0-9.]+)%\s+epoch=(\d+)/(\d+)\s+"
    r"train=([0-9.eE+\-]+)\s+val=(--|[0-9.eE+\-]+)"
)


def _parse_neural_lines(text: str):
    """Yield neural training metrics from worker console output."""
    for line in text.splitlines():
        match = _NEURAL_LINE.search(line)
        if match is None:
            continue
        percent, epoch, total, train_loss, val_loss = match.groups()
        yield {
            "epoch": int(epoch),
            "total_epochs": int(total),
            "progress_percent": float(percent),
            "train_loss": float(train_loss),
            "validation_loss": None if val_loss == "--" else float(val_loss),
        }


class LogReader(QtCore.QThread):
    rows = QtCore.pyqtSignal(list)
    console = QtCore.pyqtSignal(str)
    neural = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, log_path, console_path, interval_ms=300, parent=None):
        super().__init__(parent)
        self.tail = JsonlTail(log_path)
        self.console_path = Path(console_path)
        self.interval = max(20, interval_ms) / 1000
        self.wake = threading.Event()
        self.console_offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._console_pending = ""

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
                combined = self._console_pending + text
                lines = combined.split("\n")
                self._console_pending = lines.pop()
                for metric in _parse_neural_lines("\n".join(lines)):
                    self.neural.emit(metric)
        except FileNotFoundError:
            pass

    def run(self):
        try:
            while not self.isInterruptionRequested():
                self._read()
                self.wake.wait(self.interval)
            self._read()
            if self._console_pending:
                for metric in _parse_neural_lines(self._console_pending):
                    self.neural.emit(metric)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, runner_path, settings, log_root, options, *, parent=None):
        super().__init__(parent)
        self.runner_path = Path(runner_path).resolve()
        self.settings = settings
        self.log_root = Path(log_root)
        self.options = options
        self.process = self.reader = self.run_dir = None
        self._closing = self._restart = self._stop_requested = False
        self._terminal = self._exit_code = None
        self._last_revision = self._last_validation = None
        self._last_metric_collocation = 0
        self._last_record_reason = "尚未产生残差记录"
        self._last_node_delta = 0
        self._current_state = "idle"
        self._current_phase = ""
        self._history_sessions = []
        self._revision_offset = 0
        self._elapsed_offset_s = 0.0
        self._collocation_offset = 0
        self._latest_neural = None
        self._last_snapshot_point = None
        self._limit = max(10, int(options["max_plot_points"]))
        keys = (
            "revision", "mse", "rms", "train_max", "val_revision", "validation_max",
            "neural_epoch", "neural_train_loss", "neural_val_epoch", "neural_validation_loss",
            "neural_progress", "snapshot_elapsed", "snapshot_points",
        )
        self.series = {key: deque(maxlen=self._limit) for key in keys}

        self.setWindowTitle("SDF-MPNEO · UWPT 电磁–热训练")
        available = set(QtGui.QFontDatabase.families())
        for family in ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
            if family in available:
                self.setFont(QtGui.QFont(family, 10))
                break

        self.resize(1200, 900)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        title = QtWidgets.QLabel("UWPT 电磁–热代理模型 · 实时训练监控")
        title.setStyleSheet("font-size:22px;font-weight:600;padding:8px;")
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

        self.status_label = QtWidgets.QLabel("就绪：点击“启动”开始 UWPT 训练")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.details_label = QtWidgets.QLabel(
            "神经训练指标将在此显示：epoch、train loss、validation loss、Joule 标签数量和总进度。"
        )
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details_label.setStyleSheet(
            "QLabel{background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:8px;}"
        )
        layout.addWidget(self.details_label)
        self.path_label = QtWidgets.QLabel(str(self.log_root))
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        plots = QtWidgets.QGridLayout()
        layout.addLayout(plots, 1)
        self.curves = {}
        definitions = [
            (
                "神经网络损失", "loss", True, "epoch",
                [
                    ("neural_train_loss", "train loss", "#2563eb"),
                    ("neural_validation_loss", "validation loss", "#16a34a"),
                ],
            ),
            (
                "神经训练进度", "%", False, "epoch",
                [("neural_progress", "epoch progress", "#7c3aed")],
            ),
            (
                "Joule tensor 标签", "样本数", False, "累计训练用时 / s",
                [("snapshot_points", "已完成标签", "#0891b2")],
            ),
            (
                "旧训练残差（兼容）", "残差", True, "指标记录序号",
                [
                    ("rms", "训练 RMS", "#2563eb"),
                    ("train_max", "训练最大残差", "#d97706"),
                    ("validation_max", "验证最大残差", "#16a34a"),
                ],
            ),
        ]
        for index, (title_text, units, logarithmic, x_label, curves) in enumerate(definitions):
            plot = pg.PlotWidget(title=title_text, background="w")
            plot.setTitle(title_text, color="#1f2937", size="11pt")
            plot.setLabel("bottom", x_label, color="#374151")
            plot.setLabel("left", units, color="#374151")
            for side in ("left", "bottom"):
                plot.getAxis(side).setTextPen("#374151")
                plot.getAxis(side).setPen("#9ca3af")
                plot.getAxis(side).enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=.2)
            plot.addLegend(labelTextColor="#374151")
            plot.setLogMode(y=logarithmic)
            for key, label, color in curves:
                self.curves[key] = plot.plot(
                    name=label,
                    pen=pg.mkPen(color, width=2),
                    symbol="o",
                    symbolSize=4,
                    symbolBrush=color,
                    symbolPen=color,
                )
            plots.addWidget(plot, index // 2, index % 2)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(500)
        self.output.setMaximumHeight(150)
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

    def _resume_model(self):
        try:
            return self.settings["parameters"]["FILES"].get("resume_model")
        except (KeyError, TypeError, AttributeError):
            return None

    def _training_config(self):
        try:
            value = self.settings["parameters"]["TRAINING"]
            return value if isinstance(value, dict) else {}
        except (KeyError, TypeError):
            return {}

    def _root(self):
        try:
            return Path(self.settings["root"])
        except (KeyError, TypeError):
            return self.runner_path.parent

    @staticmethod
    def _num(value, missing="尚无"):
        if value is None:
            return missing
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    def _record_reason(self, row, previous_nodes, previous_collocation):
        nodes = int(row.get("nodes") or 0)
        collocation = int(row.get("collocation_epoch") or 0)
        delta = nodes - previous_nodes
        phase = row.get("phase", "")
        if collocation > previous_collocation:
            return "独立验证后扩充训练配点", delta
        if delta > 0 and phase == "quadratic_seed":
            return "加入物理二次响应种子", delta
        if delta > 0:
            return "接受新响应神经元", delta
        return PHASES.get(phase, "记录有效训练状态"), delta

    def _consume_neural_metric(self, metric):
        epoch = int(metric["epoch"])
        if self.series["neural_epoch"] and self.series["neural_epoch"][-1] == epoch:
            return
        self.series["neural_epoch"].append(epoch)
        self.series["neural_train_loss"].append(float(metric["train_loss"]))
        self.series["neural_progress"].append(float(metric["progress_percent"]))
        if metric.get("validation_loss") is not None:
            self.series["neural_val_epoch"].append(epoch)
            self.series["neural_validation_loss"].append(float(metric["validation_loss"]))
        self._latest_neural = dict(metric)
        self._update_curves()
        self._update_details(None)

    def _restore_history(self):
        history = build_resume_history(self.log_root, self._resume_model(), root=self._root())
        self._history_sessions = list(history["sessions"])
        self._revision_offset = int(history["revision_offset"])
        self._elapsed_offset_s = float(history["elapsed_offset_s"])
        self._collocation_offset = int(history["collocation_offset"])
        if history["rows"]:
            self._consume_rows(history["rows"], update_status=False)
        for session in self._history_sessions:
            try:
                text = (session / "worker.log").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for metric in _parse_neural_lines(text):
                self._consume_neural_metric(metric)
            if text:
                self.output.appendPlainText(f"===== 历史训练日志：{session.name} =====")
                self.output.insertPlainText(text + ("" if text.endswith("\n") else "\n"))
        if self._history_sessions:
            self.output.appendPlainText("===== 继续训练：以下为本次会话 =====")

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
            self._terminal = self._exit_code = None
            self._stop_requested = False
            self._last_revision = self._last_validation = None
            self._last_metric_collocation = 0
            self._last_record_reason = "尚未产生残差记录"
            self._last_node_delta = 0
            self._history_sessions = []
            self._revision_offset = 0
            self._elapsed_offset_s = 0.0
            self._collocation_offset = 0
            self._latest_neural = None
            self._last_snapshot_point = None
            for values in self.series.values():
                values.clear()
            for curve in self.curves.values():
                curve.setData([], [])
            self.output.clear()
            self._restore_history()

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self.run_dir = self.log_root / stamp
            self.run_dir.mkdir(parents=True)
            write_command(self.run_dir / "control.json", "run")
            worker_settings = {
                "settings": self.settings,
                "session_dir": str(self.run_dir),
                "history_sessions": [str(path.resolve(strict=False)) for path in self._history_sessions],
            }
            snapshot = self.run_dir / "worker.settings.json"
            snapshot.write_text(json.dumps(worker_settings, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.reader is not None:
                self.reader.deleteLater()
            self.reader = LogReader(
                self.run_dir / "metrics.jsonl",
                self.run_dir / "worker.log",
                self.options["refresh_ms"],
                self,
            )
            self.reader.rows.connect(self.consume)
            self.reader.console.connect(self.output.insertPlainText)
            self.reader.neural.connect(self._consume_neural_metric)
            self.reader.error.connect(self._reader_error)
            self.reader.finished.connect(self._reader_finished)
            self.reader.start()

            if self.process is not None:
                self.process.deleteLater()
            self.process = QtCore.QProcess(self)
            self.process.setProgram(sys.executable)
            self.process.setArguments(["-u", str(self.runner_path), "--worker-config", str(snapshot)])
            self.process.setWorkingDirectory(str(self.runner_path.parent))
            env = QtCore.QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONIOENCODING", "utf-8")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                env.insert(name, str(self.options["compute_threads"]))
            self.process.setProcessEnvironment(env)
            self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.process.setStandardOutputFile(str(self.run_dir / "worker.log"))
            self.process.finished.connect(self._process_finished)
            self.process.errorOccurred.connect(self._process_error)
            self._current_state = "running"
            restored = f"；已恢复 {len(self._history_sessions)} 段历史训练" if self._history_sessions else ""
            self.status_label.setText("启动 UWPT 训练进程……" + restored)
            path_text = f"本次日志：{self.run_dir}"
            if self._history_sessions:
                path_text += f"\n历史日志：{len(self._history_sessions)} 段，最近 {self._history_sessions[-1]}"
            self.path_label.setText(path_text)
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
            write_command(self.run_dir / "control.json", command)
        except OSError as exc:
            self.output.appendPlainText(f"控制指令写入失败：{exc}")
            return
        if command == "stop":
            self._stop_requested = True
        self._current_state = {"pause": "pausing", "run": "resuming", "stop": "stopping"}[command]
        self.status_label.setText(STATES[self._current_state])
        self._buttons(True, self._current_state)

    def _update_curves(self):
        mappings = {
            "neural_train_loss": ("neural_epoch", "neural_train_loss", True),
            "neural_validation_loss": ("neural_val_epoch", "neural_validation_loss", True),
            "neural_progress": ("neural_epoch", "neural_progress", False),
            "snapshot_points": ("snapshot_elapsed", "snapshot_points", False),
            "rms": ("revision", "rms", True),
            "train_max": ("revision", "train_max", True),
            "validation_max": ("val_revision", "validation_max", True),
        }
        for key, curve in self.curves.items():
            x_key, y_key, positive = mappings[key]
            x = list(self.series[x_key])
            y = list(self.series[y_key])
            if positive:
                y = [max(float(value), 1e-30) for value in y]
            curve.setData(x, y)

    def _update_details(self, row):
        if self._latest_neural is not None:
            metric = self._latest_neural
            validation = self._num(metric.get("validation_loss"), "等待下一次验证")
            snapshots = "尚未生成"
            if self.series["snapshot_points"]:
                snapshots = str(int(self.series["snapshot_points"][-1]))
            elapsed = 0.0 if row is None else float(row.get("elapsed_s") or 0.0)
            self.details_label.setText(
                f"神经训练：epoch {metric['epoch']}/{metric['total_epochs']}  |  "
                f"进度 {metric['progress_percent']:.1f}%\n"
                f"损失：train {self._num(metric['train_loss'])}  |  validation {validation}\n"
                f"Joule tensor 标签：{snapshots}  |  累计训练用时 {elapsed:.1f} 秒"
            )
            return
        if row is None:
            return
        progress = row.get("progress_percent")
        progress_text = "尚无" if progress is None else f"{float(progress):.1f}%"
        samples = row.get("training_points")
        self.details_label.setText(
            f"当前总进度：{progress_text}  |  Joule tensor 标签："
            f"{samples if samples is not None else '尚未生成'}\n"
            f"当前阶段：{PHASES.get(row.get('phase', ''), row.get('phase', ''))}  |  "
            f"累计训练用时 {float(row.get('elapsed_s') or 0.0):.1f} 秒"
        )

    def _consume_rows(self, rows, *, update_status=True):
        for row in rows:
            samples = row.get("training_points")
            if samples is not None:
                point = int(samples)
                if point != self._last_snapshot_point:
                    self._last_snapshot_point = point
                    self.series["snapshot_elapsed"].append(float(row.get("elapsed_s") or 0.0))
                    self.series["snapshot_points"].append(point)

            revision = row.get("revision")
            if row.get("mse") is not None and revision != self._last_revision:
                previous_nodes = 0
                self._last_record_reason, self._last_node_delta = self._record_reason(
                    row, previous_nodes, self._last_metric_collocation
                )
                self._last_metric_collocation = int(row.get("collocation_epoch") or 0)
                self._last_revision = revision
                self.series["revision"].append(revision)
                self.series["mse"].append(row["mse"])
                self.series["rms"].append(row["rms"])
                self.series["train_max"].append(row["train_max"])
            validation = row.get("validation_max")
            if validation is not None and (revision, validation) != self._last_validation:
                self._last_validation = (revision, validation)
                self.series["val_revision"].append(revision)
                self.series["validation_max"].append(validation)

        self._update_curves()
        if update_status and rows:
            row = rows[-1]
            self._current_state = row.get("state", "running")
            self._current_phase = row.get("phase", "")
            state = STATES.get(self._current_state, self._current_state)
            if self._active() and self._stop_requested:
                state = STATES["stopping"]
            if self._exit_code is not None:
                self._set_terminal()
            phase = PHASES.get(self._current_phase, self._current_phase)
            progress = row.get("progress_percent")
            progress_text = "" if progress is None else f" · {float(progress):.1f}%"
            self.status_label.setText(self._terminal or f"{state} · 当前阶段：{phase}{progress_text}")
            self._update_details(row)
            self._buttons(self._active(), self._current_state)

    @QtCore.pyqtSlot(list)
    def consume(self, rows):
        adjusted = []
        for row in rows:
            item = dict(row)
            item["revision"] = self._revision_offset + int(item.get("revision") or 0)
            item["elapsed_s"] = self._elapsed_offset_s + float(item.get("elapsed_s") or 0.0)
            item["collocation_epoch"] = self._collocation_offset + int(item.get("collocation_epoch") or 0)
            adjusted.append(item)
        self._consume_rows(adjusted)

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
            self._terminal = "无法启动训练进程：" + self.process.errorString()
            self.status_label.setText(self._terminal)
            self._buttons(False)
            if self._closing:
                self._finish_close()

    def _reader_error(self, message):
        self.output.appendPlainText("日志读取失败：" + message)

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
