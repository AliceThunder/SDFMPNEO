# SDF-MPNEO — 电磁–热多物理代理模型

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

0.10.0 面向科学研究，提供从三维材料/网格到无解标签训练、模型保存与加载、任意时间推理及独立数值对照的完整工作流。

研究对象是水下 WPT 的磁准静态电磁场与瞬态热场：铜、封装和海水保留为空间材料，温度改变电导率，电磁焦耳热反馈到热方程。当前范围是电磁–热耦合。

## 快速运行

```bash
python -m pip install -e '.[dev]'
python -m sdfmpneo train --demo --output results/demo.npz
python -m sdfmpneo predict results/demo.npz --a0 1 --operating 2000 500 --times 0 .05 .1 .2 .25 --output results/predictions.json
python -m sdfmpneo validate results/demo.npz --a0 1 --operating 2000 500 --times 0 .05 .1 .2 .25 --output results/validation.json
```

或者一次运行：

```bash
python examples/research_workflow.py
```

演示使用四个体四面体、一个内部热自由度、铜/海水和两个闭合电流端口，用于快速核对全链路；它不是实际线圈性能的标定案例。大电流数值用于在这个极小离散例子中激发可见的电热反馈，不是设备工作电流建议。

训练仅评估给定输入上的控制方程残差，不使用瞬态轨迹、FEM/Maxwell/COMSOL 或实验解标签。`validate` 才调用独立 Radau 积分和全阶稀疏电磁求解，用于检查训练后的模型。

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
