# SDF-MPNEO — 统一几何、自动降阶、Residual-Corrected 神经电磁–热求解器

SDF-MPNEO 现在只有一条正式模型路线：

\[
\boxed{
\text{解析/隐式几何}
\rightarrow
\text{固定多尺度背景物理空间}
\rightarrow
\text{自动 thermal / Maxwell 公共空间}
\rightarrow
\text{神经 Maxwell 初解}
\rightarrow
\text{真实 Maxwell residual 修正}
\rightarrow
\text{严格 Joule 热源}
\rightarrow
\text{结构保持热动力学}
}
\]

核心原则：**神经网络负责速度，物理方程负责答案。**

网络不直接预测温度、不直接预测 Joule tensor、不学习时间演化，也不决定结果是否可信。对任意合法查询几何，最终电磁解都必须满足当前真实背景 Maxwell 方程的 residual 容差。

## 直接运行

所有正常使用参数都集中在根目录 `run.py`：

```bash
python -m pip install -e '.[gui,neural,dev]'

python run.py --mode train
python run.py --mode predict
```

训练默认打开 PyQt 窗口；纯控制台使用：

```bash
python run.py --mode train --headless
```

不需要为每个线圈几何生成新的 Gmsh 四面体网格，也不需要手工填写任何 Maxwell rank 或 thermal rank。

## 统一几何

`run.py` 中的 `DEFAULT_GEOMETRY` 是默认查询几何。线圈支持：

- `circle`
- `rounded_square`
- `polyline`
- `spline`

几何可以改变线圈尺寸、匝数、pitch、线宽、厚度、三维平移和 roll/pitch/yaw；封装尺寸和姿态也属于同一几何描述。

几何变化不会创建另一套模型，也不会切换 fast/general/legacy mode。它只改变固定背景中的材料占据和激励，从而形成当前几何的真实物理算子。

`GEOMETRY_SAMPLING` 只用于教网络更快找到 Maxwell 解附近的位置，同时为自动物理公共空间提供代表性几何 anchor。它不是模型有效域，推理不会因为几何离开这些采样范围而拒绝计算。

唯一的几何硬边界是 `BACKGROUND['bounds']`：查询几何必须真实落在计算物理域内。若超出，应扩大背景物理域，而不是扩大所谓神经网络有效域。

## 固定多尺度背景

空间离散使用固定的非均匀 Cartesian 背景：核心 UWPT 区域较细，外部海水区域逐渐变粗。背景拓扑与几何无关。

海水不是稀疏点近似。其电导率、介电常数和三维体积电场都进入真实 Maxwell 算子：

\[
A_{\rm em}
=
C^T H_{\mu^{-1}} C
-\omega^2H_\epsilon
+i\omega H_\sigma.
\]

因此海水涡流和海水体积 Joule 发热不会因为去掉贴体网格而被忽略。

线圈采用 sub-cell thin-wire 电流源表示，不要求背景单元细到导体横截面的毫米尺度；导体自身 AC 电阻损耗以温度相关物理电阻项加入热源。

## 两个 rank 都是结果，不是配置

SDF-MPNEO 不再接受：

```python
THERMAL_RANK = 24
em_basis_max_rank = 64
```

这类人为降阶维数。

用户只提供物理 residual 目标，算法从空基开始自动增广；满足目标时当前维数就是最终 rank。

### 自动 Maxwell rank

对于每个 `(geometry, material temperature, port)` anchor，公共 Maxwell 空间 \(V_E\) 控制：

\[
\min_C \frac{\|B-A_{\rm em}V_EC\|_2}{\|B\|_2}.
\]

每次选择全体 anchor 中 residual 最大的方向增广。空间扩大后使用 minimum-residual image projection，因此最佳 residual 在数值误差范围内只能下降。

停止条件：

\[
\max_{s,p}
\frac{\|B_{s,p}-A_sV_EC_{s,p}\|_2}{\|B_{s,p}\|_2}
\leq \varepsilon_{E,\mathrm{basis}}.
\]

得到的：

\[
r_E=\dim(V_E)
\]

就是自动 Maxwell rank。

### 自动 thermal rank

热空间 \(\Phi_T\) 也从空基开始，不使用固定正弦模态数量。

对每个代表性几何，代码首先通过真实 Maxwell 方程生成包括铜损和海水体积 Joule loss 在内的物理热源 anchor \(q\)。随后同时控制两类热方程：

稳态导热：

\[
K T=q,
\]

以及初始动态：

\[
M\dot T=q.
\]

公共热空间要求：

\[
\max
\left(
\frac{\|q-K\Phi_T a\|_2}{\|q\|_2},
\frac{\|q-M\Phi_T v\|_2}{\|q\|_2}
\right)
\leq \varepsilon_{T,\mathrm{basis}}.
\]

每次增广都选择当前全体几何/热源/方程中 residual 最大的方向。满足目标后：

\[
r_T=\dim(\Phi_T)
\]

就是自动 thermal rank。

因此日志应该类似：

```text
背景空间：9261 cells，25200 Maxwell edge DOFs，thermal rank=自动计算
...
thermal 公共空间完成：自动 rank=...，maximum anchor residual=...，target=...
...
Maxwell 公共空间完成：自动 rank=...，maximum anchor residual=...，target=...
```

而不是预先打印 `thermal rank=24`。

## Maxwell 网络到底学习什么

固定背景上得到自动 Maxwell 公共空间 \(V_E\) 后，对于当前热状态和当前几何：

\[
X_0=V_E C_0.
\]

训练和网络特征统一使用 minimum-residual reduced physics：

\[
Q=(A V_E)^H(A V_E),
\qquad
S=(A V_E)^H B.
\]

神经网络预测所有端口对应的低维初始系数：

\[
C_0=\mathcal N_\theta(Q,S,\ldots).
\]

训练没有 Maxwell 解标签、没有温度解标签、没有 Joule tensor 标签。训练数据只保存真实物理 residual 二次型：

\[
\|B-AV_EC\|_2^2
=
\|B\|_2^2
-2\operatorname{Re}(C^HS)
+C^HQC.
\]

网络本体可使用 float32/CUDA，但 residual 二次型收缩固定使用 float64，避免接近收敛时的大数消减造成虚假的低 loss。

Maxwell 训练温度覆盖使用物理材料温升：

```python
"em_temperature_rise_bounds": [0.0, 80.0]
```

它不再依赖 thermal rank，也不存在固定长度的 `state_lower/state_upper`。

## 推理中的物理闭环

网络给出：

\[
X_0=V_EC_0.
\]

随后立即在完整固定背景上计算：

\[
R_0=B-A_{\rm em}X_0.
\]

如果 residual 未达到：

\[
\frac{\|B-A_{\rm em}X\|}{\|B\|}
\leq \varepsilon_{\rm em},
\]

则继续使用同一个真实 Maxwell 矩阵进行 Krylov 修正；必要时直接完成剩余物理修正。最终停止条件只有真实物理 residual。

如果极端新几何使神经网络产生 NaN/Inf，神经初解会被丢弃，求解器直接从物理 correction 继续；网络数值失效不能污染最终答案。

没有 `domain probe`、`Gate` 或额外 certification 工作流。

## Joule 发热

修正后的多端口电磁响应 \(X\) 用于构造真实二次热源。导电体积区域，特别是海水，直接使用：

\[
q_j^{\rm volume}=X^H H_j X.
\]

对任意端口电流向量 \(\zeta\)：

\[
q_j=\zeta^T G_j\zeta.
\]

这里的 \(G_j\) 由当前已经通过 Maxwell residual 检查的电磁解构造，不由 MLP 直接预测。

## 热动力学

自动生成热空间后，当前几何的完整热算子仍然来自硬物理：

\[
M_r(g)=\Phi_T^TM(g)\Phi_T,
\qquad
K_r(g)=\Phi_T^TK(g)\Phi_T,
\]

并积分：

\[
M_r(g)\dot a=-K_r(g)a+q(a,g,u).
\]

`M_r(g)`、`K_r(g)` 不由网络学习。时间也不进入神经网络。有限时间查询使用 ETD2 / adaptive ETD2；`"inf"` 使用非线性稳态求解。

## 训练配置

默认配置在 `run.py -> TRAINING`：

```python
TRAINING = {
    "basis_samples": 24,
    "thermal_basis_anchor_residual": 5e-2,
    "em_basis_anchor_residual": 2e-1,
    "em_temperature_rise_bounds": [0.0, 80.0],
    "n_operator_samples": 512,
    "device": "cuda",
    "network": {
        "width": 256,
        "blocks": 4,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "patience": 150,
        "validation_interval": 5,
    },
}
```

这里没有任何 rank 参数。

`epochs` 是最大 epoch；若 validation residual loss 连续 `patience` 个 epoch 没有改善，会提前停止并恢复 best epoch 权重。

如果 thermal 或 Maxwell 公共空间本身达不到对应 residual 目标，训练会在进入 neural epoch 前直接停止，而不是让神经网络去补偿一个不充分的物理空间。

## 推理配置

初态也不再要求用户提供一个长度等于 thermal rank 的向量。默认使用物理温升：

```python
PREDICTION = {
    "initial_temperature_rise": 0.0,
    "operating": [5.0, 0.0],
    "geometry": None,
    "times": [0.0, 0.001, 1.0, 1000.0, "inf"],
}
```

`initial_temperature_rise=0.0` 表示环境温度。若给标量非零温升或完整背景 cell 温升场，运行时会投影到训练得到的自动 thermal basis。

运行：

```bash
python run.py --mode predict
```

输出包含温度、阻抗、热源以及 Maxwell 初始/最终 residual 和 correction iteration 数。

## 缓存与模型文件

物理配置不变时可复用：

```text
results/uwpt/unified.cache.json
results/uwpt/unified.thermal_basis.npy
results/uwpt/unified.em_basis.npy
results/uwpt/unified.operator_dataset.npz
```

神经网络训练阶段的可恢复状态：

```text
results/uwpt/model.training.pt
```

正常训练完成后 checkpoint 自动删除。

最终模型：

```text
results/uwpt/model.npz
```

模型文件显式保存自动得到的 thermal basis、Maxwell basis、网络 normalizer/权重和固定背景坐标。加载模型时直接恢复这些基，不会根据某个 rank 重新生成。

训练报告包含：

```text
thermal_basis_rank
em_basis_rank
thermal_basis.maximum_anchor_relative_residual
maxwell_basis.maximum_anchor_relative_residual
training.best_validation_residual_loss
```

## 正式统一模型测试

```bash
python -m pytest -q \
  tests/test_unified_geometry.py \
  tests/test_unified_background.py \
  tests/test_unified_thermal.py \
  tests/test_unified_basis.py \
  tests/test_unified_basis_minres.py \
  tests/test_unified_residual.py \
  tests/test_unified_end_to_end.py
```

`test_unified_thermal.py` 检查 thermal rank 完全由 residual 目标自动决定、热基的体积加权正交性，以及更严格的 residual 目标不会反而得到更低 rank。

`test_unified_end_to_end.py` 覆盖：自动 thermal basis → Maxwell operator 数据 → residual NN 训练 → 模型保存 → 模型加载 → `t=0` 真实 Maxwell correction / Joule / 温度推理。

仓库中仍有历史研究代码文件，但它们不再由顶层 `sdfmpneo` API、`run.py` 或正式 CLI 自动加载，也不再作为当前统一模型的兼容目标。
