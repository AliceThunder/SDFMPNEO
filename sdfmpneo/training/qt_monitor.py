"""PyQt6 monitor for the single unified residual-corrected training path."""
from __future__ import annotations
import codecs,json,re,sys,threading,uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from PyQt6 import QtCore,QtGui,QtWidgets
import pyqtgraph as pg
from .monitor import JsonlTail,write_command

STATES={"running":"运行中","pausing":"正在暂停","paused":"已暂停","resuming":"正在恢复","stopping":"正在停止","stopped":"已停止","completed":"训练完成","failed":"运行失败"}
PHASES={"starting":"启动","maxwell_basis":"构建 residual-driven Maxwell 公共空间","maxwell_operator_samples":"生成 Maxwell residual 训练算子","neural_training":"训练 Maxwell 神经初解器","saving":"保存统一模型"}
NEURAL_LINE=re.compile(r"训练神经网络……\s*([0-9.]+)%\s+epoch=(\d+)/(\d+)\s+train=([0-9.eE+\-]+)\s+val=([0-9.eE+\-]+)")

class LogReader(QtCore.QThread):
    rows=QtCore.pyqtSignal(list); console=QtCore.pyqtSignal(str); neural=QtCore.pyqtSignal(dict); error=QtCore.pyqtSignal(str)
    def __init__(self,log_path,console_path,interval_ms=300,parent=None):
        super().__init__(parent); self.tail=JsonlTail(log_path); self.console_path=Path(console_path); self.interval=max(20,interval_ms)/1000; self.wake=threading.Event(); self.console_offset=0; self.decoder=codecs.getincrementaldecoder("utf-8")("replace"); self.pending=""
    def stop(self): self.requestInterruption(); self.wake.set()
    def _read(self):
        rows=self.tail.read()
        if rows: self.rows.emit(rows)
        try:
            with self.console_path.open("rb") as source: source.seek(self.console_offset); chunk=source.read(65536); self.console_offset=source.tell()
        except FileNotFoundError: return
        text=self.decoder.decode(chunk)
        if not text: return
        self.console.emit(text); combined=self.pending+text; lines=combined.split("\n"); self.pending=lines.pop()
        for line in lines:
            m=NEURAL_LINE.search(line)
            if m:
                percent,epoch,total,train,val=m.groups(); self.neural.emit({"percent":float(percent),"epoch":int(epoch),"total":int(total),"train":float(train),"validation":float(val)})
    def run(self):
        try:
            while not self.isInterruptionRequested(): self._read(); self.wake.wait(self.interval)
            self._read()
        except Exception as exc: self.error.emit(str(exc))

class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self,runner_path,settings,log_root,options,*,parent=None):
        super().__init__(parent); self.runner_path=Path(runner_path).resolve(); self.settings=settings; self.log_root=Path(log_root); self.options=options; self.process=self.reader=self.run_dir=None; self._closing=self._restart=self._stop_requested=False; self._terminal=None; self._current_state="idle"; self._current_phase="starting"; self._latest_neural=None; limit=max(20,int(options.get("max_plot_points",4000)))
        self.series={k:deque(maxlen=limit) for k in ("epoch","train","val_epoch","validation","progress","elapsed","samples")}; self.curves={}
        self.setWindowTitle("SDF-MPNEO · 统一神经物理训练"); available=set(QtGui.QFontDatabase.families())
        for family in ("Microsoft YaHei","PingFang SC","Noto Sans CJK SC","WenQuanYi Micro Hei"):
            if family in available: self.setFont(QtGui.QFont(family,10)); break
        self.resize(1180,820); central=QtWidgets.QWidget(); self.setCentralWidget(central); layout=QtWidgets.QVBoxLayout(central); title=QtWidgets.QLabel("统一几何 · Residual-Corrected 神经电磁–热求解器"); title.setStyleSheet("font-size:21px;font-weight:600;padding:8px"); layout.addWidget(title)
        buttons=QtWidgets.QHBoxLayout(); self.start_button=QtWidgets.QPushButton("启动"); self.pause_button=QtWidgets.QPushButton("暂停"); self.resume_button=QtWidgets.QPushButton("恢复"); self.stop_button=QtWidgets.QPushButton("停止")
        for b in (self.start_button,self.pause_button,self.resume_button,self.stop_button): b.setMinimumHeight(36); buttons.addWidget(b)
        layout.addLayout(buttons); self.status_label=QtWidgets.QLabel("就绪"); self.status_label.setWordWrap(True); layout.addWidget(self.status_label); self.details_label=QtWidgets.QLabel("等待固定背景与 Maxwell residual 训练数据。\n网络只负责初解，最终 Maxwell 解由真实残差修正。" ); self.details_label.setWordWrap(True); self.details_label.setStyleSheet("QLabel{background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:8px}"); layout.addWidget(self.details_label); self.path_label=QtWidgets.QLabel(str(self.log_root)); self.path_label.setWordWrap(True); self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse); layout.addWidget(self.path_label)
        plots=QtWidgets.QGridLayout(); layout.addLayout(plots,1); definitions=[("Maxwell residual loss","loss",True,[("train","train","#2563eb"),("validation","validation","#16a34a")]),("神经训练进度","%",False,[("progress","epoch progress","#7c3aed")]),("物理算子准备","样本数",False,[("samples","已完成物理样本","#0891b2")])]
        for i,(name,unit,logy,curves) in enumerate(definitions):
            plot=pg.PlotWidget(title=name,background="w"); plot.setLabel("bottom","epoch" if i<2 else "累计用时 / s"); plot.setLabel("left",unit); plot.showGrid(x=True,y=True,alpha=.2); plot.addLegend(); plot.setLogMode(y=logy)
            for key,label,color in curves: self.curves[key]=plot.plot(name=label,pen=pg.mkPen(color,width=2),symbol="o",symbolSize=4,symbolBrush=color)
            plots.addWidget(plot,0,i)
        self.output=QtWidgets.QPlainTextEdit(); self.output.setReadOnly(True); self.output.setMaximumBlockCount(800); self.output.setMaximumHeight(180); layout.addWidget(self.output)
        self.start_button.clicked.connect(self.start_training); self.pause_button.clicked.connect(lambda:self.command("pause")); self.resume_button.clicked.connect(lambda:self.command("run")); self.stop_button.clicked.connect(lambda:self.command("stop")); self._buttons(False)
        if options.get("auto_start",False): QtCore.QTimer.singleShot(0,self.start_training)
    def _active(self): return self.process is not None and self.process.state()!=QtCore.QProcess.ProcessState.NotRunning
    def _buttons(self,active): self.start_button.setEnabled(not active and not self._closing); self.pause_button.setEnabled(active and not self._stop_requested); self.resume_button.setEnabled(active and not self._stop_requested); self.stop_button.setEnabled(active and not self._stop_requested)
    def start_training(self):
        if self._active() or self._closing: return
        if self.reader is not None and self.reader.isRunning(): self._restart=True; self.reader.stop(); return
        self._launch()
    def _launch(self):
        self._terminal=None; self._stop_requested=False; self._latest_neural=None
        for q in self.series.values(): q.clear()
        for c in self.curves.values(): c.setData([],[])
        self.output.clear(); stamp=datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]; self.run_dir=self.log_root/stamp; self.run_dir.mkdir(parents=True); write_command(self.run_dir/"control.json","run"); snapshot=self.run_dir/"worker.settings.json"; snapshot.write_text(json.dumps({"settings":self.settings,"session_dir":str(self.run_dir)},ensure_ascii=False,indent=2),encoding="utf-8")
        self.reader=LogReader(self.run_dir/"metrics.jsonl",self.run_dir/"worker.log",self.options.get("refresh_ms",300),self); self.reader.rows.connect(self.consume); self.reader.console.connect(self.output.insertPlainText); self.reader.neural.connect(self.consume_neural); self.reader.error.connect(lambda x:self.output.appendPlainText("日志读取失败："+x)); self.reader.finished.connect(self._reader_finished); self.reader.start(); self.process=QtCore.QProcess(self); self.process.setProgram(sys.executable); self.process.setArguments(["-u",str(self.runner_path),"--worker-config",str(snapshot)]); self.process.setWorkingDirectory(str(self.runner_path.parent)); env=QtCore.QProcessEnvironment.systemEnvironment(); env.insert("PYTHONIOENCODING","utf-8")
        for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): env.insert(name,str(self.options.get("compute_threads",1)))
        self.process.setProcessEnvironment(env); self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels); self.process.setStandardOutputFile(str(self.run_dir/"worker.log")); self.process.finished.connect(self._process_finished); self.process.errorOccurred.connect(self._process_error); self.status_label.setText("启动统一训练进程……"); self.path_label.setText(f"本次日志：{self.run_dir}"); self._buttons(True); self.process.start()
    def command(self,command):
        if not self._active(): return
        write_command(self.run_dir/"control.json",command)
        if command=="stop": self._stop_requested=True; self.status_label.setText("正在停止并保存可恢复检查点……")
        elif command=="pause": self.status_label.setText("正在暂停……")
        else: self.status_label.setText("正在恢复……")
        self._buttons(True)
    @QtCore.pyqtSlot(dict)
    def consume_neural(self,m):
        self._latest_neural=m; self.series["epoch"].append(m["epoch"]); self.series["train"].append(max(m["train"],1e-30)); self.series["progress"].append(m["percent"]); self.series["val_epoch"].append(m["epoch"]); self.series["validation"].append(max(m["validation"],1e-30)); self.curves["train"].setData(self.series["epoch"],self.series["train"]); self.curves["validation"].setData(self.series["val_epoch"],self.series["validation"]); self.curves["progress"].setData(self.series["epoch"],self.series["progress"]); self._update_details(None)
    @QtCore.pyqtSlot(list)
    def consume(self,rows):
        if not rows: return
        row=rows[-1]; self._current_state=row.get("state","running"); self._current_phase=row.get("phase","starting"); samples=row.get("training_points")
        if samples is not None:
            self.series["elapsed"].append(float(row.get("elapsed_s") or 0)); self.series["samples"].append(int(samples)); self.curves["samples"].setData(self.series["elapsed"],self.series["samples"])
        state=STATES.get(self._current_state,self._current_state); phase=PHASES.get(self._current_phase,self._current_phase); progress=row.get("progress_percent"); suffix="" if progress is None else f" · 总进度 {float(progress):.0f}%"; self.status_label.setText(f"{state} · {phase}{suffix}"); self._update_details(row)
    def _update_details(self,row):
        m=self._latest_neural; epoch="尚未开始" if m is None else f"{m['epoch']}/{m['total']} ({m['percent']:.1f}%)"; train="--" if m is None else f"{m['train']:.5g}"; val="--" if m is None else f"{m['validation']:.5g}"; samples="--" if row is None or row.get("training_points") is None else str(int(row["training_points"])); elapsed="--" if row is None else f"{float(row.get('elapsed_s') or 0):.1f}s"; self.details_label.setText(f"神经训练：epoch {epoch}  |  train residual loss={train}  |  validation residual loss={val}\n物理准备样本：{samples}  |  累计用时：{elapsed}\n最终推理不使用训练域 Gate；每次 Maxwell 解都以真实背景 residual 达标为准。")
    def _process_finished(self,code,status):
        if status==QtCore.QProcess.ExitStatus.CrashExit: self._terminal="训练进程异常退出，请查看日志"
        elif code==0: self._terminal="训练完成，统一模型已保存"
        elif code==130: self._terminal="训练已停止；神经训练阶段可从检查点继续"
        else: self._terminal=f"训练进程退出（{code}），请查看日志"
        self.status_label.setText(self._terminal); self._buttons(False)
        if self._closing: self._finish_close()
    def _process_error(self,error):
        if error==QtCore.QProcess.ProcessError.FailedToStart: self.status_label.setText("无法启动训练进程："+self.process.errorString()); self._buttons(False)
    def _reader_finished(self):
        if self._closing: self.close()
        elif self._restart: self._restart=False; self._launch()
    def _finish_close(self):
        if self.reader is not None and self.reader.isRunning(): self.reader.stop()
        else: self.close()
    def closeEvent(self,event):
        self._closing=True
        if self._active(): self.command("stop"); event.ignore()
        elif self.reader is not None and self.reader.isRunning(): self.reader.stop(); event.ignore()
        else: event.accept()

def launch_window(runner_path,settings,log_root,options):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([str(runner_path)]); window=TrainingWindow(runner_path,settings,log_root,options); window.show(); return app.exec()

__all__=["launch_window"]
