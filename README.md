# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

SDF-MPNEO 面向 UWPT 磁准静态电磁–瞬态热耦合代理。训练不使用瞬态解标签，而是直接最小化控制方程残差；温度相关电导率、电磁平衡、焦耳热和热扩散始终由物理模型计算。

当前生产路径只有一条：

\[
\boxed{
\text{自动热秩}
\rightarrow
\text{三层解析响应网络}
\rightarrow
\text{t=0 source prefit}
\rightarrow
\text{有限时间物理残差训练}
\rightarrow
\text{reachable restart consistency}
\rightarrow
\text{独立验证/物理验证剪枝}
\rightarrow
\text{长时间分段 rollout}
}
\]

网络主体始终由解析 response neurons 构成，不使用普通 MLP 时间网络。不再存在候选神经元枚举、Grow / Enrich / Split、`candidate_search` 或历史 DAG 训练器。

## 安装

```bash
python -m pip install -e '.[cad,gui]'
```

开发测试：

```bash
python -m pip install -e '.[dev,cad,gui]'
```

## 推荐入口：`run.py`

训练：

```bash
python run.py --mode train
```

无界面训练：

```bash
python run.py --mode train --headless
```

推理：

```bash
python run.py --mode predict
```

通常只需要修改 `run.py` 顶部的几何、材料、误差目标、训练域和推理输入。要使用当前 v7 三层漏斗网络重新训练，请保持：

```python
FILES["resume_model"] = None
```

旧 v6 checkpoint 可以读取，但 resume 会保留旧 checkpoint 自身的网络深度/宽度，不会自动扩成新的三层结构。

## 1. 自动热秩与分层缓存

默认：

```python
THERMAL_RANK = None
```

自动选秩做一次完整离散热谱和一次物理响应 envelope，然后纯代数选出满足尾部判据的最小前缀。缓存分成三层：完整 thermal spectrum、完整 EM modal-response envelope、最终 tolerance-dependent rank/report。只改选秩容差时直接复用 envelope；电学/频率/端口变化只重算 envelope；网格或热学材料变化才重算 thermal spectrum。

热空间写为

\[
T(x,t)\approx T_{\rm ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad
K\phi_j=\lambda_jM\phi_j.
\]

自动 envelope 给出的逐模态 restart bounds 现在只作为**安全边界**，不再把它们的 Cartesian product 当成训练分布。训练初态默认只参数化少数慢/大幅热模态（`initial_training_rank=16`），在低维椭球邻域采样；restart consistency 的第二段起点由第一段响应产生，因此位于实际可达状态集上。

`thermal.rank.json` 还会在完整热谱仍被保留时给出 1/10/30/100 s 的 finite-horizon rank 诊断：

\[
E_j(H)=E_j(\infty)\left(1-e^{-\lambda_j H}\right).
\]

该结果只用于比较时间窗；生产 thermal rank 仍采用保守的稳态 envelope，不会静默降低物理热空间精度。

## 2. 三层 finite-horizon analytic response network

网络只学习单段

\[
\hat\Phi_{\Delta t}(a_0,G,U),
\qquad 0\le\Delta t\le H,
\]

默认：

```python
MAX_RESPONSE_TIME = 100.0
```

自由热响应始终精确保留：

\[
a^{(0)}(t)=e^{-\Lambda t}a_0.
\]

其上叠加三层解析响应修正：

\[
a^{(1)}=a^{(0)}+\mathcal R_1[S_1],
\qquad
a^{(2)}=a^{(1)}+\mathcal R_2[S_2],
\qquad
a^{(3)}=a^{(2)}+\mathcal R_3[S_3].
\]

每个 response neuron 对应一个目标 thermal pole：

\[
\mathcal R_{\lambda_i}[s](t)
=
\int_0^t e^{-\lambda_i(t-\tau)}s(\tau)\,d\tau.
\]

时间信号保持为有限 exponential-polynomial 组合，响应和时间导数均解析计算，不使用时间步进神经网络。

高热秩默认采用漏斗结构。若 `rank=198`：

```text
Response Layer 1: 198 neurons
Response Layer 2:  64 neurons
Response Layer 3:  32 neurons
channels_per_mode: 1
```

Layer 1 覆盖所有 retained thermal modes；Layer 2/3 根据上一阶段的 modal residual energy 自动选择 target modes。默认 source coupling ranks 为：

```text
Layer 1: linear=8, thermal/static quadratic=16, square=8
Layer 2: hidden-linear=6, hidden/static=6, hidden-nonlinear=4
Layer 3: hidden-linear=4, hidden/static=4, hidden-nonlinear=2
```

因此物理 thermal rank 可以较高，而训练复杂度主要由少量低秩 source factors 控制。两个电流输入的 quadratic bank 会显式初始化 `I1² / I2² / I1·I2` 特征（归一化坐标中），其余特征保持低秩可训练表示。

## 3. Stage 0：t=0 source prefit

解析网络结构严格满足

\[
\hat a(0)=a_0.
\]

因此 fresh training 先拟合：

\[
S_1(a_0,G,U)
\approx
F(a_0,G,U)+\Lambda a_0.
\]

Layer 1 的 source amplitudes 对固定低秩 feature bank 是线性的，所以这一步使用多点 least-squares，而不是只在盒中心设置一个 bias。`training.report.json` 会单独报告：

```text
source_prefit_rms_residual
source_prefit_max_residual
```

如果 source map 本身无法拟合物理 vector field，可以在进入昂贵 finite-time optimization 前直接看到容量问题。

## 4. Stage 1：逐层解析 residual correction

有限时间 governing-equation residual 为

\[
R_{\rm phys}=\dot{\hat a}-F(\hat a,G,U).
\]

高秩多层训练不再构造每个解析项对全部参数的 dense tangent。训练采用逐层 residual correction：前层固定，当前层只优化其线性 response-amplitude block；第二、三层在前一层停滞后按 modal residual energy 激活。

焦耳热前向仍使用批量局部投影，不为每个 retained thermal mode 单独组装全局 sparse loss operator。电磁温度反馈 Jacobian 仍采用热扩散基线 + 少数 exact directional corrections，只用于提出搜索方向。

真正的步长接受完全由**完整真实物理 residual**决定。每个 LM 方向会按：

```text
1 -> 1/2 -> 1/4 -> 1/8 -> ...
```

做真实 backtracking；一个方向全部失败后增大 damping 并重新求方向，只有连续多次 trust-region contraction 都失败才停止当前层。不存在“某个点一旦低于 tolerance 就永远不能重新升高”的硬约束，也不再用预测排序提前删掉所有小步长。

训练日志会记录 `layer / retry / backtrack / factor / damping / predicted_* / actual_* / accept/reject reason`，便于直接判断 Jacobian 失真还是网络容量不足。

## 5. Stage 2：reachable restart consistency

只有 physics residual 达到目标后才开启：

\[
R_{\rm sg}
=
\frac{
\hat\Phi_{t_1+t_2}(a_0)
-
\hat\Phi_{t_2}(\hat\Phi_{t_1}(a_0))
}{H},
\qquad t_1+t_2\le H.
\]

训练 seed 位于低维 restart 邻域，而第二段初态是第一段实际输出，因此不再对任意 198 维超盒角点强加 semigroup 约束。restart 优化使用最后一层作为有界修正层，并设置 physics guard，不能为了降低 restart defect 明显破坏 governing-equation residual。

## 6. Gate 与剪枝

训练期间所有结构 gate 固定为 1；不再同时优化 `gate × output_weight`，从而避免尺度不唯一导致的病态 Jacobian。

只有训练和独立验证都达到 tolerance 后才允许剪枝。每个候选 response neuron 删除后都重新计算完整训练 residual 和独立 validation residual；任一超标就拒绝删除。

## 7. 长时间 rollout 与稳态

长时间查询由有限单段自动组合。例如 `H=100 s`、查询 `350 s` 时内部执行：

```text
0 -> 100 -> 200 -> 300 -> 350
```

真正稳态不调用网络 `t=inf`，而是解

\[
F(a_\infty,G,U)=0.
\]

`model.predict(float("inf"), ...)` 只是路由到物理稳态求解器。

## 8. GUI 与监控

`source_prefit`、`response_layer_1/2/3`、physics residual、restart consistency 和 validation 都会发布工作进度。残差曲线保留显式 point marker；即使只有一个 revision 也能看到。resume 历史会先加载，当前 run 的 revision 接在历史之后。

如果训练 residual 尚未达到进入 validation 的条件，报告会明确：

```text
validation_performed = false
```

不会再把训练 residual 冒充成独立验证结果，也不会白跑昂贵 validation sweep。

## 9. 输出

默认训练会写出：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/train.settings.json
results/uwpt/training.report.json
results/uwpt/thermal.rank.json
results/uwpt/network.structure.json
results/uwpt/geometry.domain.json
results/uwpt/logs/
```

自动热秩缓存会另外生成 spectrum / envelope / selection 三层缓存文件。

## 10. 模型格式

当前 fixed analytic response network metadata 为 **v7**；fixed-geometry / geometry research model 仍为 v5。

v7 新增 per-layer widths、target modes、per-layer coupling ranks 和 analytic state-feature term budget。v6 等宽 fixed-response checkpoint 仍可读取；但 resume 不会把旧 depth/width 自动迁移成 v7 默认 `r→64→32` 架构。要测试新架构请 fresh train。
