# 几何参数化电磁张量神经电热 ROM

本目录冻结一条干净的生产理论主线：

```text
validated Maxwell truth physics
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

## 生产主架构

神经代理学习

\[
g\rightarrow \beta_\theta(g)
\rightarrow \{\widehat Z_{\rm field}(g),\widehat H_j(g)\}.
\]

其中 output POD 只是数值压缩，不是理论前提。解码器必须按构造恢复：

- reciprocal 情况下的复对称 `Z_field`；
- 每个 thermal mode 的复 Hermitian `H_j`。

在线 thermal model 为

\[
M_r(g)\dot a
=-K_r(g)a
+\widehat q_{\rm vol}(g,c)
+q_{\rm wire}(a,g,c)
+f_T(g).
\]

有限时间使用 ETD2 / adaptive ETD2 / IMEX；`t = inf` 单独求稳态。

## Physics Gate

正式 tensor dataset 和 neural training 之前，底层 truth model 必须先通过：

1. Maxwell formulation、开放域/域扩展和 mesh convergence；
2. 端口符号、reciprocity、passivity 和 Poynting/power balance；
3. Joule heat 与 Maxwell conductivity Hodge 的离散能量一致性；
4. 任意复端口相位下的 Hermitian quadratic contraction；
5. 几何无穿透、包封、边界裕量和连续守恒 source/material deposition；
6. thermal boundary/domain sensitivity；
7. thermal basis 的真实 Galerkin steady/dynamic residual；
8. filament-conductor、AC resistance、海水热传递等模型假设的适用性。

没有通过这些 Gate 时，降低 neural loss 没有物理意义。

## Truth snapshot

每个 geometry 的 truth sample 应包含：

```text
geometry
-> validated Maxwell multi-RHS solve
-> Z_field
-> complex Hermitian H_1 ... H_r
-> reciprocity / passivity / power-balance audit
-> structured real encoding y(g)
```

只有 audit 通过的数据可以进入 train/validation/test。

## 训练

只用 training split 构造 normalization 和 POD：

\[
y\approx \bar y+U_K\beta.
\]

普通 residual MLP 学习

\[
g\mapsto\beta.
\]

validation/test 除 latent error 外，还必须检查：

- decoded `Z_field` error；
- decoded `H_j` error；
- random/structured complex-current Joule error；
- impedance/power error；
- frozen electrothermal trajectory / steady-state error。

## 推理

对一个 geometry：

```text
geometry
-> one neural forward
-> cache Z_field / H_j
-> exact current contraction
-> update true R_wire(a,g)
-> true thermal ROM integration / steady solve
```

同一几何的多个时间、电流和初值查询复用同一组 neural EM tensors。

## 理论文档

主文档：`main.tex`

章节包括：

- 系统边界与冻结架构；
- Maxwell / thermal governing model；
- passive impedance、Hermitian Joule 与功率守恒；
- low-dimensional neural tensor surrogate；
- dataset/training protocol；
- thermal inference；
- error budget；
- Physics Gate；
- implementation boundaries；
- innovation positioning。

这一目录中的理论只保留上述生产主链，不并列维护其他神经求解架构。
