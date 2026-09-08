# SDF-MPNEO — 电磁–热多物理代理模型

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

0.10.0 面向科学研究，提供从三维材料/网格到无解标签训练、模型保存与加载、任意时间推理及独立数值对照的完整工作流。

研究对象是水下 WPT 的磁准静态电磁场与瞬态热场：铜、封装和海水保留为空间材料，温度改变电导率，电磁焦耳热反馈到热方程。当前范围是电磁–热耦合。

## 快速运行

推荐直接使用根目录的 [run.py](run.py)。所有运行配置集中在文件顶部，按模式/路径、几何网格、材料与电磁、热截断、训练和推理分类，并附单位说明。

```bash
python -m pip install -e '.[cad,gui]'
python run.py --mode train
python run.py --mode predict
```

直接运行 UWPT 线圈—封装—海水算例，模型与输出保存在 `results/uwpt/`。也可以直接修改 `MODE="train"` 或 `MODE="predict"` 后运行 `python run.py`，适用于 IDE 的运行按钮。

训练默认打开 **PyQt6 实时窗口**，点击“启动”后执行任务；窗口提供暂停、恢复和停止按钮。`MONITOR` 配置控制日志周期、曲线刷新、显示点数和计算线程数。使用 `python run.py --mode train --headless` 可仅训练并记录日志，推理模式保持原有命令行输出。

训练会自动生成网格、构建物理模型、训练并保存。设置 `MESH["generate"]=False` 可导入已有网格。相对路径统一以 `run.py` 所在目录为基准，无需手动切换目录或编辑另一份 JSON。

推理参数在 `PREDICTION` 中设置；仅加载保存模型，不会重新训练或生成网格。`FILES["resume_model"]` 可指定继续训练的模型。每次运行自动保存配置，训练另保存残差报告。

实时曲线包括 MSE、训练 RMS/最大残差、独立检查最大残差、响应节点数和训练配点数。后台训练进程的日志线程周期性写入 JSONL；另一个 `QThread` 增量读取文件，通过信号通知主线程绘图。每次任务的日志保存在 `results/uwpt/logs/<时间_编号>/`。完整操作和日志格式见 [训练监控说明](docs/TRAINING_MONITOR.md)。

训练仅评估给定输入上的控制方程残差，不使用瞬态轨迹、FEM/Maxwell/COMSOL 或实验解标签。原有 `python -m sdfmpneo` 接口也保留，其中 `validate` 才调用独立 Radau 积分和全阶稀疏电磁求解；一键脚本不自动执行这类验证。

`train` 在达到给定数值残差目标时退出码为 0；预算耗尽时仍保存模型和报告，退出码为 2，并明确记录未收敛。

## 真实线圈、封装与海水网格

```bash
python -m pip install -e '.[dev,cad]'
python examples/create_uwpt_mesh.py examples/configs/uwpt_research.json
python -m sdfmpneo train --config examples/configs/uwpt_research.json --output results/uwpt.npz
python -m sdfmpneo predict results/uwpt.npz --a0 0 0 --operating 5 0 --times 0 .1 .5 1 --output results/uwpt_predictions.json
```

配置包含几何尺寸、接收线圈平移/旋转、材料参数、频率、实体端子、热空间截断及训练域。将 `geometry.shape` 改为 `rounded_square` 可生成圆角方形线圈。自有 Gmsh 2.2 ASCII 四面体网格也可以直接通过材料和端子物理标签导入。

Linux 的 Gmsh Python wheel 可能需要系统 OpenGL/X11 运行库，例如 Ubuntu 的 `libglu1-mesa`、`libxft2`、`libopengl0`。不使用 CAD 时无需安装 Gmsh。

## 模型如何工作

1. 同一四面体网格上的 Nédélec/P1 离散提供电磁算子和热质量/刚度矩阵。
2. 温度相关铜电阻率、海水损耗、热源及其导数由同一材料定义计算。
3. 残差–Riesz 增广建立电磁降阶空间，不采集全阶电磁解快照。
4. 电磁平衡消元后得到 `da/dt = -Lambda a + q_em(a,U)`。
5. 初始热坐标、静态电流参数和响应神经元组成解析 DAG。参考态的焦耳热二次结构初始化响应节点，随后根据耦合残差增长并联合更新权重。
6. 加载保存的网络和空间基后，可直接查询任意时间，无需从零逐步积分。

`predict --state-only` 只执行解析网络和温度重构，不调用电磁求解。默认完整推理还返回 `Z/R/L`、互感所在的电感矩阵、各材料焦耳损耗、物理残差及电磁误差诊断。

每个保存模型对应一个固定几何、频率和材料定义，支持所声明域内的任意初始热坐标、静态端口电流和时间。改变几何或频率后可用同一工作流重新构建训练；已有跨几何研究模块仍保留，不把单几何模型当作未经验证的通用几何模型。

## 科研误差的含义

- 新工作流的 `numerically_converged` 表示训练点和独立参数检查点达到数值残差目标，不等同于连续域或真实设备误差证明。
- `thermal_rank` 是显式的科研截断阶数；应通过增加阶数检查结果收敛。原有按误差证书选秩的接口也保留。
- 示例独立瞬态对照默认使用完整电磁空间，但保持相同热空间；它检验解析演化与电磁降阶误差，不替代热秩/网格收敛性研究或实验验证。
- 端口电流采用峰值相量；平均功率为 `0.5 * Re(I^H Z I)`。温度输出为包含边界和参考温度的全节点开尔文温度。

## 代码与说明

- [科研工作流、数学关系和配置说明](docs/RESEARCH_WORKFLOW.md)
- [完整 Python 算例](examples/research_workflow.py)
- [真实 UWPT 配置](examples/configs/uwpt_research.json)
- [科研模型构建、保存、加载和推理](sdfmpneo/research.py)
- [无标签残差训练](sdfmpneo/training/research.py)
- [原有理论推导](docs/SDFMPNEO_theory.tex)

`docs/PRODUCTION_STATUS_0_9.md` 等文件保留为先前版本的研究背景；当前运行入口和完成状态以本文及 `docs/RESEARCH_WORKFLOW.md` 为准。

```bash
python -m pytest -q -W error::numpy.exceptions.ComplexWarning -W error::scipy.linalg.LinAlgWarning
```

可选 Gmsh 可用时，测试还会实际生成圆形与圆角方形导体、封装和海水网格，并检查材料界面共形及端子标签。
