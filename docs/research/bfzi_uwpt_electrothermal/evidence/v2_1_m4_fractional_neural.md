# UMFC–BFZI V2.1：分数阶扩散记忆神经代理首次真实训练与留出验证

日期：2026-11-12。代码和实测均在 `research/bfzi-uwpt-electrothermal-surrogate`；原始物理教师为独立 `AliceThunder/BFZI` 的 V150 分支 `bfzi-v150-turn-scan`，M4 采用原 V148 Green/resolvent 算子。模型训练、运行和测试均在 XEnergy 上执行，未使用 GitHub Actions。

## 1. 分数阶特性为什么适用于电磁热代理

M4 的三维热传导模型本身属于有限体积离散的整数阶热方程，真实**历史记忆存在于完整空间温度状态**。只向网络提供当前电压将丢失这些历史信息；把 450 维海水—封装—铜温度压缩成低维代理时，需要附加有限历史状态。

可选的低维有效传热核取幂律记忆：

```text
k_alpha(t) = t^(alpha-1)/Gamma(alpha), 0 < alpha < 1
m_alpha(t) = integral_0^t k_alpha(t-s) u(s) ds
```

采用正权重指数谱（SOE）实现：

```text
lambda_j: logarithmically spaced over [1/2400,1/12] 1/s, 14 rates
h_j(t+dt)=exp(-lambda_j dt)*h_j(t)
          +(1-exp(-lambda_j dt))*u_interval
w_j(alpha)=softmax_j(-alpha*log(lambda_j))
m_alpha=sum_j w_j(alpha)*h_j
```

其中 `h_j` 是经过规范化的指数响应，因此权重 `lambda_j^(-alpha)` 对应未规范化卷积核的 `lambda_j^(1-alpha)`，是 `t^(alpha-1)` 的有限时间窗指数和表示。每步只更新当前 **3×14** 状态，计算量不随历史长度增长。可学习的三个 α 分别作用于三个损耗/电流强度通道。

**历史驱动与物理反馈**：本模型在每次预测时由网络预测的完整复 `Z(t)` 和真实电路串联补偿/负载 `Zext` 解

```text
I(t) = [Z_pred(t-previous)+Zext]^-1 [V(t),0]^T
u(t) = [ 0.5 Re(I^H Z_pred I), 0.5|I_1|², 0.5|I_2|² ]
```

三通道再进入分数阶历史状态。热态与阻抗由神经输出在 20 维训练 POD 空间恢复；完整热输出包括原海水 384 单元、两个封装各 32 单元及两个铜节点，共 **450 维**。网络还输出互易的 2×2 复 Z（6 个实数独立分量）；用预测 Z 重新求解电流、效率与下一次记忆更新，从而实现有明确电路反作用的**因果连续时步神经闭环**。

这一版神经热源利用总电磁损耗及双铜电流强度作为学习输入，并未直接代替上一阶段原 BFZI 的三维弱形式 `q(x,t)` 场求解；需要空间 q/J 时沿用已验证的物理 ROM 或后续增加空间场输出头。

## 2. 最接近的已有研究和具体新增价值

1. S. M. Sivalingam、V. Govindaraj，*Neural fractional order differential equations*，Expert Systems with Applications 267 (2025) 126041，DOI: https://doi.org/10.1016/j.eswa.2024.126041。使用 Caputo 分数阶导数给神经模型引入记忆，并在其他动力学/时间序列任务上验证。本文属于通用分数阶神经动力学方法。
2. Jiang et al.，*Fast Evaluation of the Caputo Fractional Derivative and Its Applications to Fractional Diffusion Equations*，Communications in Computational Physics 21 (2017) 650–678，https://arxiv.org/abs/1511.03453。建立分数阶历史核的指数和快速求积，证明可将长时历史计算降至有限记忆状态。
3. Zhu、Xu，*A Fast High Order Method for the Time-Fractional Diffusion Equation*，SIAM Journal on Scientific Computing，DOI: https://doi.org/10.1137/18M1231225。用指数和压缩分数阶扩散历史并分析时间离散。

**本研究新增的集成点**是：用独立 BFZI V150 的真实双端口复阻抗、三维海水和双封装、铜温度高功率启停轨迹监督；分数阶记忆与**实际复端口电路重新求解**组成自回归闭环，预测全空间低阶热重构、`Z11/Z12/Z22` 的实部和虚部、两端口电流及效率；同时对不同记忆核结构执行完全相同的跨电压/启停时序留出比较。已有通用分数阶神经算法和快速核求积可直接借鉴，研究价值在 BFZI 耦合问题、守恒物理教师及多材料历史预测的结合。当前验证仅覆盖 M4 圆形极限，跨圆角方形/位姿与更长时间窗仍需进一步验证。

## 3. 真实独立 BFZI 教师数据

`examples/bfzi_frac_data.py` 在同一进程中只构建一次原 BFZI M4 Green/空气电感及封装热算子，连续运行 **30 个**由不同电压和启停程序组成的完整 600 s 时程。

- 原 BFZI 载波 200 kHz、σ0=5 S/m、初始温度 293.15 K、源/负载 0.5 Ω，原 V150 M4 几何/材料；
- 海水 384 FV、双封装 `4×2×4` 每盒 32 FV、双铜节点；海水 `--sea-heat port-weak`，原端口复阻抗与海水热源积分逐步一致；
- 每 30 s 采样一次，总 **21 时刻/条、450 热状态**；
- **20 条训练候选**：电压 25/55/90/120/150 V × 四组不同开/关时程；其中每个电压的第 4 组作为验证集（15 条实际训练、5 条模型选择）；
- **10 条最终留出**：40/75/105/135 V × 两组未参与训练的启停时序，加 65/115 V 各一组**三段启动、两次关断**的不同历史；这些轨迹不参与训练和超参数选择；
- 原 BFZI M4 `Zsea` 与独立原求解器冷态相对差 **4.4857e-18**，实际生成总耗时约 **75.8 s**。

教师：`runs/m4_frac_teacher.npz`，含输入电压、21 时刻完整 450 温度、6 个完整复 Z 分量、实际端口电流、效率、分组和每条时序定义。数据由真实原 BFZI–三维热直接法生成，无人工代理标签。

## 4. 三类记忆结构的相同数据消融

使用 PyTorch，20 维温度 POD 输出 + 6 实数复阻抗输出；两层隐藏宽度 96、SiLU，Adam 学习率 0.003；最多 2400 epoch，按五条验证轨迹保存最佳权重。

| 10 条独立留出时程的平均误差 | 分数阶记忆 | 单指数记忆 | 混合分数阶+指数 |
|---|---:|---:|---:|
| 海水温度最大空间误差 (K) | 0.042738 | 0.034182 | **0.028611** |
| 双封装温度最大空间误差 (K) | 0.435091 | 0.372122 | **0.347823** |
| 双铜温度最大误差 (K) | **1.097630** | 1.500756 | 1.301813 |
| 复 Z 相对 L2（含冷态本底） | **6.437e-6** | 8.674e-6 | 7.497e-6 |
| **动态 ΔZ(t) 相对 L2** | **0.049752** | 0.059838 | 0.054986 |
| 两端口电流相对 L2 | **2.735e-5** | 3.551e-5 | 2.762e-5 |
| 效率最大绝对差 | 0.0002491 | 0.0002900 | **0.0002337** |

**必须用动态 ΔZ 评价热反馈学习**，因为完整 Z 中约 20 Ω 的冷态电抗会压低相对误差。温度指标为每条时程全空间最大绝对误差再在 10 条上平均，不能误称为逐单元 MAE。

- 学到的纯分数阶 α：`[0.238532, 0.774547, 0.903257]`。
- 混合网络分数阶 α：`[0.320767, 0.494379, 0.702272]`；同时学到指数时间常数 **[646.38, 264.76, 141.44] s**。
- 在当前 600 s 工况下，混合记忆的封装/海水/效率表现最佳，纯分数阶的铜/动态阻抗表现最佳。现有结果明确支持保留和深化分数阶记忆，并给出后续改进的方向：根据物理材料响应自适应组合两套历史信息，优化全时间轨迹的误差及跨步长一致性。

## 5. 真实端到端同网格时程速度与更细时间步

由 `examples/bfzi_frac_bench.py` 重新构建独立原 BFZI V150 M4，然后**只对相同输入历史的 21/41 个时步求解计时**，基建模和神经训练均不计入：

| 留出历史：115 V 三次通电 [0,120]、(210,330]、(420,510] s | 原 BFZI 直接法 | 神经混合记忆 | 时程加速 |
|---|---:|---:|---:|
| **30 s 步，21 个时刻** | 2.827378 s | 0.015101 s | **187.2×** |
| **15 s 步，41 个时刻（训练未使用此步长）** | 4.881299 s | 0.012707 s | **384.1×** |

两次同工况的推理计时包含 PyTorch 模型在 CPU 上的自回归计算及 numpy 电路求解，受进程/缓存/线程影响；将其视为当前机器的时程测量结果。

| 115 V 三次启动，直接法 vs 神经网络最大空间温度误差 | 30 s | 15 s |
|---|---:|---:|
| 海水 (K) | 0.079615 | 0.075937 |
| 双封装 (K) | 0.503694 | 0.916199 |
| 双铜 (K) | 3.507926 | 4.278618 |
| 动态 ΔZ 相对 L2 | 0.073587 | 0.075696 |

15 s 的独立直接法验证揭示跨时间步长的热态精度仍需改进。当前连续时间核支持任意正时步递推；训练集仅包含 30 s 的教师时步。下一步优先增加多步长教师训练与相同历史不同步长一致性目标。

## 6. 可运行入口及数据

```powershell
# 原独立 BFZI V150 三维温度 + 复阻抗教师，30条真正不同启动/关断历史
python -m examples.bfzi_frac_data --bfzi-root D:\work\bfzi-et-v2.1\BFZI --step 30 --out runs/m4_frac_teacher.npz

# 三组结构同数据训练，分别保存纯分数阶/单指数/混合网络
python -m examples.bfzi_frac train --data runs/m4_frac_teacher.npz --model runs/m4_frac.pt --out runs/m4_frac_metrics.json --epochs 2400

# 完全不调用 BFZI 物理求解器的神经推理（此例是15s网格）
python -m examples.bfzi_frac infer --model runs/m4_frac_hybrid.pt --voltage 100 --heat 210 --cool 210 --step 15 --out runs/m4_frac_infer.json

# 真正相同三次通电历史，重新调用原 BFZI 跑同网格时程计时/误差
python -m examples.bfzi_frac_bench --bfzi-root D:\work\bfzi-et-v2.1\BFZI --model runs/m4_frac_hybrid.pt --step 30 --out runs/m4_frac_bench.json
python -m examples.bfzi_frac_bench --bfzi-root D:\work\bfzi-et-v2.1\BFZI --model runs/m4_frac_hybrid.pt --step 15 --out runs/m4_frac_bench_15s.json
```

生成内容：
- `sdfmpneo_bfzi/frac_memory.py`：因果 SOE 分数阶/指数/混合网络与自回归电路；
- `examples/bfzi_frac_data.py`、`examples/bfzi_frac.py`、`examples/bfzi_frac_bench.py`：教师、训练/消融/推理和同网格计时；
- `tests/test_bfzi_frac.py`：冷态严格复现、不同历史、可学习 α 与指数时间常数、断电零电流等；
- `runs/m4_frac_teacher.npz`：真实 30 条时程；
- `runs/m4_frac.pt`、`runs/m4_frac_exp.pt`、`runs/m4_frac_hybrid.pt`：三个已训练权重，可直接加载；
- `runs/m4_frac_metrics.json/.npz`：训练/留出/消融指标与分场预测；
- `runs/m4_frac_infer.json/.npz`：全程可独立推理示例；
- `runs/m4_frac_bench.json`、`runs/m4_frac_bench_15s.json`：原 BFZI 对照实测。

### 下一步主任务

1. 多时间步长、更多断电/再启动次数的真实 BFZI 教师，训练可跨步长与更长时间预测的统一神经记忆模型；
2. 把 `port-weak` 三维海水损耗/单位 J 输出也加入神经空间 POD 读出，闭合 `Z/I/效率/J/q/T` 的快速代理；
3. 将原 BFZI 统一圆形—圆角矩形几何/双端口位姿参数接入神经条件输入，形成跨几何模型；
4. 持续与同源原 BFZI 直接法和物理 ROM 比较，在复杂历史上评价动态 ΔZ、空间温度、能量守恒及速度。
