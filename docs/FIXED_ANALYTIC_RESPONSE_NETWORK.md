# 自动热秩 + Finite-Horizon Fixed Analytic Response Network

当前 SDF-MPNEO 生产训练路径：

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
\text{独立验证剪枝}
\rightarrow
\text{分段 rollout}
}
\]

训练不使用瞬态解标签。热动力学目标始终来自真实降阶物理场

\[
\dot a=F(a,G,U),
\qquad
R_{phys}=\dot{\hat a}-F(\hat a,G,U).
\]

## 1. 热空间与一次性完整诊断

温度场写为

\[
T(x,t)\approx T_{ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad
K\phi_j=\lambda_jM\phi_j.
\]

默认 `THERMAL_RANK=None`。

自动热秩不再为 `4,8,16,32,...` 分别构建多个 thermal/EM probe。首次运行直接构建一次完整离散热谱，并在该基上计算一次完整 response envelope

\[
E_j\approx \max\frac{|q_j|}{\lambda_j}.
\]

然后只对缓存数组做前缀扫描，选择满足

\[
s\|E_{r+1:n}\|_2
\le
\varepsilon_{abs}
+
\varepsilon_{rel}\|E\|_2
\]

的最小 `r`。完整谱用于选秩诊断，不代表代理网络训练完整热空间。

零热态的全部 operating anchors 共用一次 EM factorization；每个 `H_j` 只组装一次，并同时评价所有 operating anchors。

## 2. 三层缓存

默认启用：

```text
model.config.thermal_rank.cache.spectrum.npz
model.config.thermal_rank.cache.envelope.npz
model.config.thermal_rank.cache.json
```

三层 key 分离：

- **spectrum cache**：mesh + thermal conductivity + volumetric heat capacity + thermal boundary treatment；
- **envelope cache**：spectrum key + frequency + electrical material + port/current mapping + operating box + thermal-state probe strategy；
- **selection cache**：envelope key + thermal tail tolerance + safety factors。

因此：

- 只修改 thermal rank tolerance：不重算热谱、不重跑 EM，只重新扫描 envelope；
- 修改电导率/频率/电流范围：热谱仍可复用，只重算 envelope；
- 修改 mesh/热导率/热容量：热谱与 envelope 都失效；
- 完全相同配置：直接命中最终 selection cache。

缓存使用 mesh 内容 SHA-256，而不是文件时间戳。稳定网格建议关闭重复生成：

```python
MESH["generate"] = False
```

旧 cache format 不迁移，版本不匹配时直接重新构建。

有限锚点 envelope 仍只标记为 `certified_continuous_domain=false`；严格 thermal-tail certificate 路径独立保留。

## 3. restart 状态域

自动热秩的完整 envelope 同时生成 restart-state box。第 2 段以后上一段终态作为下一段 `a0`，因此训练域覆盖允许初始扰动和安全放大的物理热响应，而不是统一的任意小盒子。

## 4. 单段有限时间解析网络

网络只定义

\[
\hat\Phi_t(a_0,G,U),\qquad 0\le t\le H,
\]

其中默认

```python
MAX_RESPONSE_TIME = 100.0
```

每个响应通道绑定真实热衰减率：

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},
\qquad h_{jc}^{(\ell)}(0)=0.
\]

source 包含显式 bias、低秩线性映射、动态热态×静态工况/几何项，以及逐模态热态平方项。动态层不使用 ReLU。

高热秩时默认容量自动收紧；`rank >= 96` 时默认 `depth=1`、`channels_per_mode=1`，使参数/Jacobian 规模近线性增长。

## 5. 两阶段无标签训练

先训练

\[
R_{phys}=\dot{\hat a}-F(\hat a,G,U),
\]

再加入

\[
R_{sg}
=
\frac{
\hat\Phi_{t_1+t_2}(a_0)
-
\hat\Phi_{t_2}(\hat\Phi_{t_1}(a_0))
}{H}.
\]

Gauss–Newton 只在当前 hard points 上构造 Jacobian，但所有 trial step 都用完整训练 residual 集合判定是否接受。

## 6. 长时间 rollout

对有限总时间

\[
T=NH+\Delta t
\]

使用

\[
\hat\Phi_T
=
\hat\Phi_{\Delta t}
\circ
\hat\Phi_H^N.
\]

每段终态成为下一段初值，并检查 restart-state box。

## 7. 稳态

稳态不通过网络 `t=\infty` 外推，而直接解

\[
F(a_\infty,G,U)=0.
\]

`model.predict(float("inf"), ...)` 只路由到物理 damped Newton。

## 8. 当前格式

当前模型只支持 segmented fixed-network 架构。旧 DAG、旧 fixed-network 参数布局和旧无限时间模型不转换；需要重新训练。同一当前格式 checkpoint 可正常 resume。
