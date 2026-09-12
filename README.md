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
\text{t=0 source-factor prefit}
\rightarrow
\text{逐层有限时间物理残差训练}
\rightarrow
\text{reachable restart consistency}
\rightarrow
\text{独立验证/物理验证剪枝}
\rightarrow
\text{长时间分段 rollout}
}
\]

网络主体始终由解析 response neurons 构成，不使用普通 MLP 时间网络。

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

v6 checkpoint 可以读取；resume 会保留 checkpoint 自身的网络深度/宽度，不会静默扩成新的三层结构。v7 checkpoint resume 会从当前最深 active response layer 继续，不会清零或重新选择已经训练过的深层 target modes。

## 1. 自动热秩与有限时间诊断

默认：

```python
THERMAL_RANK = None
```

自动选秩先计算完整离散热谱和完整物理 modal-response envelope，再选出满足尾部判据的最小 thermal prefix。缓存分成：

```text
thermal spectrum
→ EM/modal response envelope
→ tolerance-dependent rank/report
```

热空间写为

\[
T(x,t)\approx T_{\rm ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad K\phi_j=\lambda_jM\phi_j.
\]

自动 envelope 给出的逐模态 restart bounds 只作为安全边界，不再把它们的 Cartesian product 当成训练分布。训练初态默认只参数化少数慢/大幅热模态：

```python
TRAINING["initial_training_rank"] = 16
```

这些坐标在低维椭球邻域采样；其他 thermal coordinates 保持在中心值。这样 thermal rank 可以保持物理精度，而训练域不会变成 198 维随机超盒。

`thermal.rank.json` 会复用 full-spectrum cache，额外输出 1/10/30/100 s 的 finite-horizon rank 诊断：

\[
E_j(H)=E_j(\infty)\left(1-e^{-\lambda_jH}\right).
\]

该结果只用于判断缩短 `MAX_RESPONSE_TIME` 是否真的能显著降热秩。生产 thermal rank 仍采用保守稳态 envelope，不会静默降低物理空间精度。

## 2. 三层 analytic response funnel

网络只学习一个有限段：

\[
\hat\Phi_{\Delta t}(a_0,G,U),\qquad 0\le\Delta t\le H.
\]

默认：

```python
MAX_RESPONSE_TIME = 100.0
```

自由热响应始终精确保留：

\[
a^{(0)}(t)=e^{-\Lambda t}a_0.
\]

在此基础上叠加三层解析响应修正：

\[
a^{(1)}=a^{(0)}+\mathcal R_1[S_1],
\]

\[
a^{(2)}=a^{(1)}+\mathcal R_2[S_2],
\]

\[
a^{(3)}=a^{(2)}+\mathcal R_3[S_3].
\]

每个 response neuron 对应一个目标 thermal pole：

\[
\mathcal R_{\lambda_i}[s](t)
=
\int_0^t e^{-\lambda_i(t-\tau)}s(\tau)\,d\tau.
\]

时间信号保持为有限 exponential-polynomial 组合，响应与时间导数均解析计算，不把 `t` 输入普通神经网络。

高热秩默认采用漏斗宽度。若 `rank=198`：

```text
Response Layer 1: 198 neurons
Response Layer 2:  64 neurons
Response Layer 3:  32 neurons
channels_per_mode: 1
```

Layer 1 覆盖所有 retained thermal modes；Layer 2/3 根据上一阶段的 modal residual energy 自动选择 target modes。

默认低秩 source coupling：

```text
Layer 1:
  linear rank              = 8
  thermal/static rank      = 16
  modewise-square rank     = 8

Layer 2:
  hidden-linear rank       = 6
  hidden/static rank       = 6
  hidden-nonlinear rank    = 4

Layer 3:
  hidden-linear rank       = 4
  hidden/static rank       = 4
  hidden-nonlinear rank    = 2
```

因此 thermal rank 决定物理温度场精度，而网络训练复杂度主要由少量 source factors 控制。

## 3. Stage 0：t=0 source-factor prefit

解析结构严格满足

\[
\hat a(0)=a_0.
\]

fresh training 首先拟合

\[
S_1(a_0,G,U)
\approx
F(a_0,G,U)+\Lambda a_0.
\]

这一步不是“随机 features + 只调 output amplitude”。source snapshots 会直接学习低秩输入方向：

- affine/linear source：完整线性回归后做 truncated SVD；
- modewise thermal square：平方坐标回归后做 truncated SVD；
- thermal/static 与 static/static 双线性 source：回归后做低秩 CP 分解；
- 最后固定这些数据驱动 factor directions，再统一做一次 output-amplitude least squares。

因此 Stage 0 真正识别的是物理 source 子空间。训练报告单独写出：

```text
source_prefit_rms_residual
source_prefit_max_residual
```

如果连 `t=0` source 都无法达到需要的精度，就可以在进入昂贵 finite-time training 前直接看到容量瓶颈。

## 4. Stage 1：逐层物理 residual correction

控制方程残差：

\[
R_{\rm phys}=\dot{\hat a}-F(\hat a,G,U).
\]

高秩训练不构造所有解析项对全部参数的 dense tangent。训练按 response layer 分块：

```text
Layer 1 amplitude block
→ Layer 2 residual-corrector block
→ Layer 3 residual-corrector block
```

前层固定；当前层只构造其解析 amplitude Jacobian。Layer 2/3 的 target modes 由上一层的 modal residual energy 自动选择。

未激活的深层 amplitude 为 0，前向会直接短路，不构造无用 hidden/state exponential products；只有该层真正开始训练时才展开对应 source bank。

焦耳热前向使用批量局部投影。电磁温度反馈 Jacobian 使用热扩散基线 + 少量 exact directional corrections，只负责提出搜索方向。

真正的步长接受完全由完整真实物理 residual 决定。每个 LM 方向按：

```text
1 → 1/2 → 1/4 → 1/8 → ...
```

做 actual-physics backtracking；一个方向全部失败后增大 damping 并重新求方向。只有连续多次 trust-region contraction 仍失败，才停止当前层。

不存在“某个点一旦低于 tolerance 就永远不能重新升高”的硬约束，也不再用预测排序提前删除所有小步长。

`metrics.jsonl` 会记录：

```text
layer
iteration
retry
backtrack
factor
damping
predicted_max
predicted_weighted_rms
actual_rms
actual_max
actual_physics_max
actual_restart_max
accept/reject reason
```

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

第一段 seed 位于低维 restart 邻域；第二段初态由第一段网络输出产生，所以真正施加 consistency 的 restart state 是可达状态，而不是 198 维超盒角点。

restart 优化使用当前最深 active response layer 作为有界修正层，并设置 physics guard，不能为了降低 restart defect 明显破坏 governing-equation residual。

## 6. Gate 与剪枝

训练期间所有结构 gate 固定为 1，不同时训练 `gate × output_weight`，避免尺度不唯一造成 Jacobian 病态。

未激活 response layer 虽然 gate=1，但 amplitude=0，因此 GUI 和结构报告不会把它计作 active neuron。

只有训练和独立验证都达到 tolerance 后才允许 pruning。每个候选 response neuron 删除后都会重新计算训练 residual 和独立 validation residual；任一超标就拒绝删除。

## 7. Validation

只有 physics residual 与 restart residual 都达到训练目标后才执行独立 validation。

如果训练尚未达到进入 validation 的条件：

```text
validation_performed = false
```

不会用训练残差冒充验证结果，也不会白跑昂贵的独立 validation sweep。

## 8. 长时间 rollout 与稳态

长时间查询通过有限段自动组合。例如 `H=100 s`、查询 `350 s`：

```text
0 → 100 → 200 → 300 → 350
```

真正稳态不调用网络 `t=inf`，而是直接解

\[
F(a_\infty,G,U)=0.
\]

`model.predict(float("inf"), ...)` 只是路由到物理稳态求解器。

## 9. GUI 与监控

GUI 会显示：

```text
source_prefit
response_layer_1
response_layer_2
response_layer_3
restart_consistency
validation
structure_pruning
```

残差曲线保留 point marker，即使只有一个 revision 也可见。resume 会加载历史曲线；当前 run 的 revision 接在历史之后。

## 10. 主要输出

默认训练输出：

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

自动热秩缓存另外生成 spectrum / envelope / selection 三层缓存文件。

## 11. 模型格式

当前 fixed analytic response network metadata 为 **v7**；fixed-geometry / geometry research model 仍为 v5。

v7 新增：

- per-layer widths；
- per-layer target modes；
- per-layer hidden/cross/state coupling ranks；
- analytic state-feature term budget。

v6 等宽 fixed-response checkpoint 仍可读取。要验证当前默认 `r→64→32` 架构，使用 fresh training：

```python
FILES["resume_model"] = None
```
