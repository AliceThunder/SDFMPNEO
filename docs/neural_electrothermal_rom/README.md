# 几何参数化电磁张量神经电热 ROM

本目录冻结一条生产理论主线：

```text
validated Maxwell truth physics
-> transient-aware shared thermal basis
-> geometry-dependent Z_field / Hermitian Joule tensors H_j
-> neural geometry surrogate
-> exact complex-current quadratic contraction
-> true thermal ROM / ODE
-> T(t), Z(t), power, steady state
```

神经网络只学习几何到低维电磁参数的映射。它不学习时间、不学习 thermal mass/stiffness、不把端口电流当普通网络输入，也不直接学习最终温度轨迹。

## 当前材料模型下的关键简化

当前正式模型中，体 Maxwell constitutive parameters 不随温度变化，因此

\[
A_{\rm em}=A_{\rm em}(g).
\]

固定几何只需一组 Maxwell truth response。thermal state 只通过线圈集中电阻模型进入 wire heating 和总阻抗：

\[
Z(a,g)=Z_{\rm field}(g)+R_{\rm wire}(a,g).
\]

体 Joule modal source 为

\[
q_{{\rm vol},j}(g,c)=\operatorname{Re}(c^H H_j(g)c),
\qquad H_j=H_j^H.
\]

因此任意电流幅值与相位都由解析二次型覆盖，无需对每个 operating current 单独训练。

当前 `c` 表示规定的复端口电流；若真实工作点由 source/load/compensation circuit 决定，则应在本模型的 `Z(a,g)` 外接显式电路方程求解 `c`。

## 生产主架构

神经代理学习

\[
g\rightarrow \beta_\theta(g)
\rightarrow \{\widehat Z_{\rm field}(g),\widehat H_j(g)\}.
\]

output POD 只是数值压缩，不是理论前提。生产解码器必须按构造恢复：

- reciprocal 情况下的复对称 `Z_field`；
- `Herm(Z_field) >= 0` 的被动端口约束；
- 每个 thermal mode 的复 Hermitian `H_j`。

注意：这些结构是必要条件，但不自动保证 `Z` 与所有 `H_j` 之间完整的 Poynting/功率一致性；该一致性仍需单独 audit。

在线 thermal model 为

\[
M_r(g)\dot a
=-K_r(g)a
+\widehat q_{\rm vol}(g,c)
+q_{\rm wire}(a,g,c)
+f_T(g).
\]

有限时间使用 ETD2 / adaptive ETD2 / IMEX；`t = inf` 单独求稳态。

## Thermal basis 依赖顺序

`H_j` 依赖 thermal mode `phi_j`，因此理论顺序必须是：

```text
validated spatial Maxwell/Joule/source truth bank
-> build and freeze shared thermal basis Phi
-> project truth Joule operators onto Phi
-> generate final Z/H_j tensor dataset
-> POD + neural training
```

不能先生成最终 `H_j` labels 再构建 thermal basis。

thermal basis 也不能只靠 steady `K^-1 Q` 和初始 derivative `M^-1 Q` 两个端点。当前理论采用跨 geometry/source/time-scale 的 resolvent anchors：

\[
(K+sM)u=Q,
\]

并直接检查 reduced Galerkin resolvent residual。shift `s` 覆盖 steady 与主要 transient inverse-time scales，最终仍以 full-vs-ROM transient/steady trajectory audit 为准。

## Physics Gate

正式 tensor dataset 和 neural training 之前，底层 truth model 必须先通过：

1. Maxwell formulation、开放域/域扩展和 mesh convergence；
2. 端口符号、reciprocity、passivity 和 Poynting/power balance；
3. Joule heat 与 Maxwell conductivity Hodge 的离散能量一致性；
4. 任意复端口相位下的 Hermitian quadratic contraction；
5. closed-loop source 的电流归一、离散连续性/拓扑闭合；
6. 几何无穿透、包封、边界裕量和连续守恒 source/material deposition；
7. thermal boundary/domain sensitivity；
8. transient-aware thermal basis residual 与 full-vs-ROM trajectory；
9. filament-conductor、AC resistance、海水热传递等模型假设的适用性。

没有通过这些 Gate 时，降低 neural loss 没有物理意义。

## 数据生成与训练

数据生成分两层：

```text
geometry
-> validated Maxwell multi-RHS response + spatial heat/source bank
-> build/freeze Phi
-> Z_field + complex Hermitian H_1 ... H_r
-> reciprocity / passivity / power-balance audit
-> structured real encoding y(g)
```

只有 audit 通过的数据可以进入 neural train/validation。还必须保留一组既未参与 neural training、也未参与 thermal-basis enrichment 的 end-to-end frozen audit geometries。

只用 neural training split 构造 normalization 和 POD：

\[
y\approx \bar y+U_K\beta.
\]

普通 residual MLP 学习

\[
g\mapsto\beta.
\]

validation/audit 除 latent error 外，还必须检查 decoded `Z_field/H_j`、complex-current Joule、impedance/power consistency、thermal trajectory 和 steady-state error。

周期姿态不能直接使用跨越 `-pi/pi` 的裸角度；应采用 `sin/cos`、连续 rotation representation 等无分支跳变编码。

## 推理

对一个 geometry：

```text
geometry
-> one neural forward
-> passive/reciprocal/Hermitian decode
-> cache Z_field / H_j
-> exact current contraction
-> update true R_wire(a,g)
-> true thermal ROM integration / steady solve
```

同一几何的多个时间、电流和初值查询复用同一组 neural EM tensors。

## 理论文档

主文档：`main.tex`

章节包括：系统边界、Maxwell/thermal governing model、passive impedance/Hermitian Joule、低维 tensor surrogate、dataset/training protocol、thermal inference、error budget、Physics Gate、implementation boundaries 和 innovation positioning。

这一目录中的理论只保留上述生产主链，不并列维护其他神经求解架构。
