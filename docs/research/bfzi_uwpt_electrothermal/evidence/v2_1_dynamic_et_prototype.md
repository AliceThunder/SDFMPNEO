# V2.1：动态电磁—热双向反作用首个可运行代码原型

**执行方案：** [V2.1 动态电磁热实施方案](../UMFC_BFZI_DYNAMIC_ETNS_V2_1_EXECUTION_PLAN.md)  
**状态：** 代码已提交；合成小型算例在沙箱运行通过；真实 BFZI 计算、原独立 BFZI V148 后端、空间热参考精度与训练代理仍 **OPEN**。

## 1. 本轮新增功能

- **`sdfmpneo_bfzi/dynamic_et.py`**：`BFZIDynamicET`，将已存在的 BFZI R/L/K 海水响应按空间热态重组，计算每个时间步的完整复数 `Zsea, Ztotal`，包括自阻抗、互阻抗的实部与虚部；铜内部复阻抗直接调用 `sources.conductor_internal_impedance(Tcu)`。
- **空间热场**：在既有海水正权求积点上构造三维体积加权热单元，分别为铜及两个绝缘封装添加热状态；海水—海水、铜—封装、封装—海水之间使用对称非负热导，外层海水与环境换热。海水为三维热离散，铜和封装为聚集热节点。
- **动态闭环**：两端口复电压与给定被动补偿/负载阻抗确定电流，当前空间海水 J 及铜内部损耗计算焦耳热，热容量/热导隐式积分，重复求解热反作用并推进真实时间。关断后电流和 Joule 源消失，导热与环境换热继续。
- **`examples/bfzi_dynamic_et.py`**：`python run.py --mode bfzi-et-dynamic`。默认采用现有双 BFZI 演示几何，依次通电、关断冷却、重新通电。JSON 含每时刻全复 2×2 Z、海水复 Z、有效电感矩阵、实际复电流、铜/海水功率、三维海水热单元温度与热源、两个铜温度与两个封装温度、储热量。
- **`tests/test_bfzi_dynamic_et.py`**：两个聚焦测试覆盖空间温度改变复海水互阻抗、两端口互易、源功率/海水热源一致、通断电温度历史及冷却。

## 2. 温度—电磁反作用物理路径

沿用本仓库已有 BFZI Green–Galerkin 的实无散度电流基和几何固定的电磁矩阵：

\[
R_{mn}^{(g)}=\sum_{k\in g}\frac{w_k}{\sigma_0}
\mathbf W_m(\mathbf x_k)\cdot\mathbf W_n(\mathbf x_k), \qquad
\kappa_g(t)=1+\alpha_\sigma[T_g(t)-T_0],
\]

\[
\mathbf R(t)=\sum_g \frac{\mathbf R^{(g)}}{\kappa_g(t)},\quad
\mathbf X(t)=[\mathbf R(t)+j\omega\mathbf L]^{-1}(-j\omega\mathbf K),
\]

\[
\mathbf Z_{\mathrm{sea}}(t)=j\omega\mathbf K^\mathsf T\mathbf X(t),\qquad
\mathbf Z_{\mathrm{total}}(t)=
j\omega\mathbf L_{\mathrm{air}}+
\mathbf Z_{\mathrm{sea}}(t)+
\operatorname{diag}\mathbf Z_{\mathrm{cu}}(T_{\mathrm{cu}}(t)).
\]

**即使几何和空气磁链不变，海水温度改变 `R` 后，`X` 与 `Zsea` 的实部、虚部和交叉阻抗也同时发生变化。** 这对应随时间变化的有效自感、互感以及海水损耗电阻；铜温度同时改变导体内部电阻和电抗。

两个端口的电流 `I` 经完整复阻抗及外部补偿电路计算。每个海水积分点：

\[
\mathbf J_k(t)=\sum_p I_p(t)\sum_m W_{km}X_{mp}(t),\qquad
q_k(t)=\frac{\lvert\mathbf J_k(t)\rvert^2}{2\sigma_k(t)}.
\]

先将双端口复电流相加再取模平方，交叉损耗保留。三维热单元热输入为 `∑_{k∈g}w_k q_k`，铜节点输入为 `0.5 Re(Zcu,p)|I_p|²`。空间温度演化采用热容 `C` 与对称非负热导矩阵 `H` 的隐式步：

\[
[\mathbf C+\Delta t\,\mathbf H]\theta_{n+1}
=\mathbf C\theta_n+\Delta t\,\mathbf p[T_{n+1},I_{n+1}],
\]

用少量固定点迭代协调同一热步的电磁响应和热源。`θ` 是相对环境温度的升温状态，支持恢复和继续积分。

## 3. 调用方式

```bash
python run.py --mode bfzi-et-dynamic --out runs/bfzi_et_dynamic.json
python run.py --mode bfzi-et-dynamic --voltage 12 --step 60 --heat 240 --cool 180 --restart 180 --sea-step 0.16 --sea-alpha 0.02 --out runs/bfzi_et_dynamic.json
python -m pytest tests/test_bfzi_dynamic_et.py -q
```

初始演示采用串联无源外阻抗并抵消参考空气自感对角电抗的固定电容补偿，采用峰值复相量，输电端电压开启/关断期间变化。

## 4. 本轮实际验证与重要边界

在本轮沙箱的独立合成算例中（6 个模拟海水求积点、2 个电流模态、两个铜/两个封装热状态），已实际执行下列测试并得到 **PASS**：空间海水升温改变双端口复 Z 的实部与虚部、互阻抗变化；相同 `J` 产生的海水 Joule 热与端口海水耗散一致；通电升温、关断后电流/焦耳热归零且冷却；热时间步正常运行。合成算例用于验证程序耦合逻辑，**不是物理 BFZI 场景精度证据**。

本轮未在完整 SDFMPNEO 依赖环境运行 BFZI 真实几何的完整时程；新仓库专项测试已编写，尚未报告项目整体 pytest 通过数。需要在实际完整环境运行上述命令，并解决实际发现的问题。

**本轮电磁后端使用 SDFMPNEO 现有 `sea_reference.py` Green–Galerkin R/L/K 状态；独立 `AliceThunder/BFZI` 仓库的 V148 dense Green/resolvent 与 V149 FFT 求解器尚未接入。** 下一步在 BFZI 既有同源空间电流/阻抗状态上建立可调用的温度依赖入口，并对照原 BFZI 等温复阻抗输出；不能把当前原型宣称为原 V148 后端的直接运行结果。

铜和封装热态当前为各两节点，尚不是完整的铜线和封装内部三维热场。海水 3D 热单元之间目前使用保正的邻近体积热导，后续应通过真实热传导 FEM/FV 的温度、热点与热量收支进行校准。该原型用于尽快打通 V2.1 连续时间反馈，数值参考精度、热物性参数与训练教师认证仍 OPEN。

## 5. 下一优先任务

1. 在真实双 BFZI 演示场景运行第一条完整时间轨迹，检查 `Z11(t),Z12(t),Z22(t)` 实虚部、温度热点、海水/铜空间损耗、停机降温和再启动响应；修复实际功能问题。
2. 把原 BFZI V148/V149 的复阻抗、电流/源 Green 解算路径接到共同接口，允许按原求积单元非均匀 `σ_k(T_k)` 更新电磁算子并保留同源 J/q；等温结果回归原仓库数据。
3. 用真实铜/封装/海水热导及界面边界进一步完善三维热场，构造可复现的跨历史参考时程，之后实现电磁和热神经 ROM。
