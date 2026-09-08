# UWPT 实时训练窗口

```bash
python -m pip install -e ".[cad,gui]"
python run.py --mode train
```

直接运行 UWPT 算例。训练窗口打开后点击“启动”；也可把 `run.py` 的 `MONITOR["auto_start"]` 设为 `True`。修改训练域、材料或几何仍在 `run.py` 顶部完成。

## 运行与控制

| 按钮 | 行为 |
|---|---|
| 启动 | 创建本次独立日志目录，按当前配置快照启动训练进程 |
| 暂停 | 请求训练在下一个数值检查点暂停，保留进程、网络和正在进行的训练状态 |
| 恢复 | 继续同一个已暂停任务，不重新构建模型或重新训练 |
| 停止 | 在检查点结束任务，保存最近一次已经接受的有效网络到 `model.stopped.npz` |

残差评估、候选搜索、权重优化和独立参数检查均有控制检查点。当前的 CAD 构建、组装或单次矩阵运算不会被强行打断，因此界面会先显示“正在暂停”或“正在停止”，真正到达检查点后才确认已暂停/停止。界面在此期间仍可响应。

关闭窗口也会请求停止，待训练进程完成保存、日志线程退出后关闭。若在模型构建完成之前停止，还没有可保存的网络，日志会明确记录 `checkpoint: null`。停止的模型不会标记为数值收敛，也不会覆盖正常完成的 `model.npz`。

停止后按钮“恢复”不再可用；要从停止文件继续训练，在 `FILES["resume_model"]` 中填写其路径，再启动任务。这会恢复已保存的物理空间和有效网络，但不会恢复停止时尚未接受的临时试步或候选搜索位置。暂停后的恢复则保留原进程中的这些状态。

## 并发结构

1. **Qt 主线程**处理按钮和绘图，不执行训练、模型加载或日志文件读取。
2. **训练进程**运行原来的 UWPT 训练入口，防止 Python GIL 或数值计算阻塞窗口。进程内另有一个 Python 日志线程，按周期写入已刷新到文件的 JSONL。
3. **日志读取 QThread**增量读取 JSONL 和标准输出文件，仅通过信号传递新记录到 Qt 主线程。只处理完整换行记录，写入一半的数据留待下一轮读取。
4. 按钮通过原子替换的控制文件传递指令，训练侧确认执行后再通过日志反馈真实状态。

Qt 对象的跨线程通信遵循其[信号与槽机制](https://doc.qt.io/qt-6/signalsandslots.html)，曲线使用 [PyQtGraph](https://www.pyqtgraph.org/)。

## 曲线含义

- **MSE**：配点上模态方程残差向量平方范数的平均值。
- **Train RMS / maximum**：同一训练配点集上的 RMS 和最大残差。
- **Validation maximum**：实际执行独立参数检查时的最大残差；不是外部解标签误差。未检查时为空，不伪造每轮验证值。
- **Nodes**：已接受的响应神经元数。
- **Training points**：训练配点数量，用于观察配点扩充。

横轴是已接受更新的编号，不是传统神经网络的 epoch。曲线在初始评估、二次热源初始化、接受的 Gauss–Newton 权重更新和节点增长时更新。回溯中被拒绝的试步不进入曲线。配点集改变时会记录 `collocation_epoch`，不同配点集上的损失不能直接解释为同一目标的单调下降。

组装或长时间搜索期间仍有周期心跳和用时更新，但不会编造新的损失值。对数曲线的显示下限是 `1e-30`，原始日志保留实际零值。窗口保留最近 `max_plot_points` 个曲线点，完整历史始终留在日志文件中。

## 配置与文件

`MONITOR` 提供：`enabled`、`auto_start`、`log_dir`、`log_interval_s`、`refresh_ms`、`max_plot_points` 和 `compute_threads`。最后一项设置后台进程的 BLAS/OpenMP 线程数。

每次图形任务生成独立目录：

- `worker.settings.json`、`settings.json`：实际运行配置快照。
- `metrics.jsonl`：周期指标、已接受更新及最终状态；每行是独立 JSON。
- `worker.log`：标准输出及异常堆栈。
- `control.json`：`run`、`pause` 或 `stop` 指令。

指标包含 `timestamp`、`elapsed_s`、`sequence`、`state`、`phase`、`revision`、`collocation_epoch`、`mse`、`rms`、`train_max`、`validation_max`、`nodes` 及配点数量。`elapsed_s` 是包含暂停时间的墙钟用时；尚未产生的数值为 `null`。

无需界面时：

```bash
python run.py --mode train --headless
python run.py --mode predict
```

无界面训练同样周期记录 JSONL，标准输出保留在终端。Linux 下 Qt/Gmsh 可能需要系统运行库，例如 Ubuntu 的 `libegl1`、`libglu1-mesa`；使用桌面 X11 时还需系统 Qt 所要求的 XCB 库。
