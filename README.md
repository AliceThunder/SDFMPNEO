# SDF-MPNEO — 统一几何、Residual-Corrected 神经电磁–热求解器

SDF-MPNEO 现在只有一条正式模型路线：

\[
\boxed{
\text{解析/隐式几何}
\rightarrow
\text{固定多尺度背景物理空间}
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

网络不直接预测温度、不直接预测 Joule tensor、不学习时间演化，也不决定结果是否可信。对任意合法查询几何，最终电磁解都必须满足当前真实背景 Maxwell 方程的残差容差。

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

不需要为每个线圈几何生成新的 Gmsh 四面体网格，也不需要额外手写物理 JSON。

## 统一几何

`run.py` 中的 `DEFAULT_GEOMETRY` 是默认查询几何。线圈支持：

- `circle`
- `rounded_square`
- `polyline`
- `spline`

几何可以改变线圈尺寸、匝数、pitch、线宽、厚度、三维平移和 roll/pitch/yaw；封装尺寸和姿态也属于同一几何描述。

几何变化不会创建另一套模型，也不会切换 fast/general/legacy mode。它只改变固定背景中的材料占据和激励，从而形成当前几何的真实物理算子。

`GEOMETRY_SAMPLING` 只用于教网络更快找到 Maxwell 解附近的位置。它**不是模型有效域**，推理不会因为几何离开这些采样范围而拒绝计算。

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

## Maxwell 公共空间的 rank 自动确定

固定背景上构造 residual-driven 公共电磁空间 \(V\)，但 **rank 不是配置参数**。

算法从空基开始，对所有训练 anchor（几何、热状态、端口）计算真实 Maxwell 相对 residual：

\[
r_{s,p}(V)
=
\frac{\|B_{s,p}-A_s V c_{s,p}\|_2}{\|B_{s,p}\|_2}.
\]

每次选择当前最差的 `(geometry, state, port)` residual，将新的物理 residual lift 正交加入公共空间，然后重新检查所有 anchor。只有当

\[
\max_{s,p} r_{s,p}(V)
\leq \varepsilon_{\rm basis}
\]

时停止。此时

\[
r=\dim V
\]

就是自动得到的 Maxwell basis rank。

因此不存在 `em_basis_max_rank=64/96/128` 这类需要人工猜测的模型参数。冗余 anchor 不会无意义增加 rank；更复杂的物理解空间会自动得到更高 rank。如果 residual 中已经不存在新的数值独立方向但仍无法达到目标，训练会在进入神经 epoch 前明确失败，而不是让用户继续手工调大 rank。

`em_basis_anchor_residual` 是公共空间作为**神经初解空间**的误差目标，不是最终物理解容差。最终推理始终还要满足 `PHYSICS['maxwell_residual_tolerance']`。

## 网络到底学习什么

对于当前热状态和当前几何：

\[
A_r=V^H A_{\rm em}V,
\qquad
B_r=V^HB.
\]

神经网络只预测所有端口对应的低维初始系数：

\[
C_0=\mathcal N_\theta(A_r,B_r).
\]

网络输入显式包含当前 reduced operator/RHS 的物理特征，因此新几何不是简单的几何参数黑箱外推。

训练没有 Maxwell 解标签、没有温度解标签、没有 Joule tensor 标签。训练数据只保存由真实物理算子得到的 residual 二次型：

\[
\|B-AVC\|_2^2
=
\|B\|_2^2
-2\operatorname{Re}(C^H(AV)^HB)
+C^H(AV)^H(AV)C.
\]

网络直接最小化真实背景 Maxwell relative residual 的平方均值。网络本体可使用 float32/CUDA，但 residual 二次型收缩固定使用 float64，避免接近收敛时的大数消减造成虚假的低 loss。

## 推理中的物理闭环

网络给出：

\[
X_0=VC_0.
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

则继续使用同一个真实 Maxwell 矩阵进行 Krylov 修正；必要时直接完成剩余物理修正。最终停止条件只有真实 Maxwell residual。

如果极端新几何使神经网络产生 NaN/Inf，神经初解会被丢弃，求解器直接从物理 correction 继续；网络数值失效不能污染最终答案。

因此训练分布只影响需要多少修正迭代，不定义模型是否能处理某个几何。没有 `domain probe`、`Gate` 或额外 certification 工作流。

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

热质量和热扩散仍然是硬物理：

\[
M_r(g)\dot a=-K_r(g)a+q(a,g,u).
\]

`M_r(g)`、`K_r(g)` 来自当前固定背景材料占据并投影到固定热多尺度基，不由网络学习。Maxwell basis/operator 训练数据生成阶段不会无意义地装配热矩阵；只有真实热查询需要当前几何的热算子。

时间不进入神经网络。有限时间查询使用 ETD2 / adaptive ETD2；`"inf"` 使用非线性稳态求解。

## 训练配置

默认配置在 `run.py -> TRAINING`：

```python
TRAINING = {
    "basis_samples": 24,
    "em_basis_anchor_residual": 2e-1,
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

`epochs` 是最大 epoch；若 validation residual loss 连续 `patience` 个 epoch 没有改善，会提前停止并恢复 best epoch 权重。

PyQt 窗口显示：

- Maxwell 公共空间/物理算子样本准备进度；
- train residual loss；
- validation residual loss；
- epoch 与总训练进度。

## 训练缓存与停止

固定背景、公共电磁空间和 operator-residual dataset 与物理配置绑定，配置不变时直接复用：

```text
results/uwpt/unified.cache.json
results/uwpt/unified.em_basis.npy
results/uwpt/unified.operator_dataset.npz
```

神经网络训练阶段的可恢复状态：

```text
results/uwpt/model.training.pt
```

正常训练完成后 checkpoint 自动删除。

## 推理

`run.py -> PREDICTION`：

```python
PREDICTION = {
    "a0": [...],
    "operating": [5.0, 0.0],
    "geometry": None,
    "times": [0.0, 0.001, 1.0, 1000.0, "inf"],
}
```

`geometry=None` 使用 `DEFAULT_GEOMETRY`；也可以直接传新的合法几何。

```bash
python run.py --mode predict
```

输出包含温度、阻抗、热源以及 Maxwell 初始/最终 residual 和 correction iteration 数。模型文件显式保存 EM basis、thermal basis、网络 normalizer/权重和固定背景坐标，加载后不会重新猜测训练时的基。

## 主要输出

```text
results/uwpt/model.npz
results/uwpt/training.report.json
results/uwpt/predictions.json
results/uwpt/unified.cache.json
results/uwpt/unified.em_basis.npy
results/uwpt/unified.operator_dataset.npz
results/uwpt/logs/
```

## 正式统一模型测试

```bash
python -m pytest -q \
  tests/test_unified_geometry.py \
  tests/test_unified_background.py \
  tests/test_unified_basis.py \
  tests/test_unified_residual.py \
  tests/test_unified_end_to_end.py
```

`test_unified_basis.py` 验证公共 Maxwell rank 由 residual 自动确定；`test_unified_end_to_end.py` 覆盖：无解标签 operator 数据生成 → residual NN 训练 → 模型保存 → 模型加载 → `t=0` 真实 Maxwell correction/Joule/温度推理。

仓库中仍有历史研究代码文件，但它们不再由顶层 `sdfmpneo` API、`run.py` 或正式 CLI 自动加载，也不再作为当前统一模型的兼容目标。
