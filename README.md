# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

SDF-MPNEO 面向 UWPT 磁准静态电磁–瞬态热耦合代理。训练不使用瞬态解标签，而是直接最小化控制方程残差；温度相关电导率、电磁平衡、焦耳热和热扩散始终由物理模型计算。

当前生产路径只有一条：

\[
\boxed{
\text{自动热秩}
\rightarrow
\text{固定最大解析响应网络}
\rightarrow
\text{有限时间物理残差训练}
\rightarrow
\text{restart consistency}
\rightarrow
\text{验证剪枝}
\rightarrow
\text{长时间分段 rollout}
}
\]

不再存在候选神经元枚举、Grow / Enrich / Split、`candidate_search` 或历史 DAG 训练器。

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

通常只需要修改 `run.py` 顶部的几何、材料、误差目标、训练域和推理输入。

## 1. 自动热秩与分层缓存

默认：

```python
THERMAL_RANK = None
```

自动选秩只做一次完整离散热谱和一次完整物理响应 envelope，然后纯代数选出满足尾部判据的最小前缀。缓存分成三层：完整 thermal spectrum、完整 EM modal-response envelope、最终 tolerance-dependent rank/report。只改选秩容差时直接复用 envelope；电学/频率/端口变化只重算 envelope；网格或热学材料变化才重算 thermal spectrum。

热空间写为

\[
T(x,t)\approx T_{\rm ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad
K\phi_j=\lambda_jM\phi_j.
\]

自动选秩还根据允许的初始扰动和物理响应 envelope 生成 restart-state box，供分段 rollout 的后续段使用。

## 2. finite-horizon fixed analytic network

网络只学习单段

\[
\hat\Phi_{\Delta t}(a_0,G,U),
\qquad 0\le\Delta t\le H,
\]

默认：

```python
MAX_RESPONSE_TIME = 100.0
```

高热秩下默认采用浅层、每个 retained thermal mode 一个 response channel 的可扩展结构；例如 `rank=198` 时默认 198 个 response channels，而不是旧设计的 1584 个。

## 3. 高热秩物理残差训练

焦耳热前向不再为每个 retained thermal mode 单独组装全局 sparse loss operator。当前实现恢复一次局部 Nedelec 电场，在每个四面体积分四个 P1 焦耳热矩，再一次性投影到所有 retained thermal modes；该计算与逐模态

\[
q_j=x^H H_j(a)x
\]

代数等价。

完整电磁热源 Jacobian 在 `rank≈200` 时会产生约 `r²` 次 loss-operator derivative 组装，因此训练不再把完整 `J_q` 当作每轮 GN 必需量。GN 先使用热扩散 Jacobian 作为基线，并在网络当前最敏感的少数热状态方向上，用精确快速物理 residual 的中心差分加入低秩温度反馈修正。trial、训练 residual 和 validation residual 始终使用完整真实物理，因此收敛判据未放宽。

LM/backtracking 也不再把所有 damping/步长组合都拿到完整训练集上试。候选步先用已经构造好的 hard-point Jacobian 做廉价预测排序，每次优化迭代最多只对 3 个候选执行完整精确 residual sweep。resume/fine-tune 时会直接跳过 amplitude-only warm start，避免恢复网络参数后又重复一轮无意义的幅值预热。

## 4. restart consistency

第一阶段先最小化 governing-equation residual：

\[
R_{\rm phys}=\dot{\hat a}-F(\hat a,G,U).
\]

物理残差达到目标后，再开启

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

训练和独立验证都必须满足 physics residual 与 restart-rate defect 的目标。

## 5. 长时间 rollout 与稳态

长时间查询由有限单段自动组合。例如 `H=100 s`、查询 `350 s` 时内部执行：

```text
0 -> 100 -> 200 -> 300 -> 350
```

真正稳态不调用网络 `t=inf`，而是解

\[
F(a_\infty,G,U)=0.
\]

`model.predict(float("inf"), ...)` 只是路由到物理稳态求解器。高热秩稳态同样使用精确物理 residual 与廉价热扩散方向/回溯验收，避免构造完整电磁热源 Jacobian。

## 6. GUI 与监控

`initial_residual`、`physics_jacobian`、`physics_residual`、restart 阶段都会发布逐点工作进度；首个 RMS 点尚未形成时 GUI 仍会显示当前处理进度。残差曲线使用显式高对比 pen，不再依赖浅灰默认线条。

## 7. 输出

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

## 8. 模型格式

当前 fixed analytic response network metadata 为 v6；fixed-geometry / geometry research model 为 v5。旧 fixed-network 参数布局和历史 DAG 模型不会自动迁移，需要重新训练。当前 v6 `.stopped.npz` 可以继续训练。
