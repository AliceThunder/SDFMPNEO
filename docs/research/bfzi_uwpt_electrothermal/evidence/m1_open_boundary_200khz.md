# M1 海水电磁直接法：自由空间反应场边界修正

**研究方案：** [UMFC–BFZI-ETNS V2.0](../UMFC_BFZI_ETNS_V2_EXECUTION_PLAN.md)  
**日期：** 2026-11-11  
**状态：** 自由空间边界原型已实现并实算；外域复阻抗与空间场数值精度 **OPEN**。

## 1. 面向的主问题

原 M1 三维 A–φ 直接法采用解析闭合线圈主场，海水反应磁矢势满足有限立方体外边界的 `n × A_react = 0`。这是人为截断条件，在 200 kHz、5 S/m 海水下外域 R=0.52–1.05 m 的复端口阻抗变化显著。

此前将同源体积耦合 `Z_sea = jω∫A_source·J_sea dV` 接为端口读出后，`P_port` 与 `∫q dV` 在**零反应场边界的离散模型内**达到浮点一致。这一恒等式属于有限维方程和算子结构检查，本身不构成无限海水参考精度证明。

本轮直接改进**海水反应磁场外边界**，保留原双线圈统一 BFZI 几何、A–φ 物理源与三维材料网格。

## 2. 实现方法

在原 `solve_closed_primary_secondary` 上增加可选参数 `open_steps`，先计算零边界解，再用得到的海水电流近似外域反应矢势：

\[
\mathbf A_{\mathrm{react}}(\mathbf x)
\approx \frac{\mu_0}{4\pi}
\sum_{t\in\Omega_{\mathrm{sea}}}
\frac{V_t\overline{\mathbf J}_t}{\lVert\mathbf x-\mathbf x_t\rVert},
\qquad \mathbf x\in\partial\Omega.
\]

这里 `x_t` 是海水四面体中心、`V_t` 为单元体积、`J̄_t` 为该单元的实际求解海水平均复电流。

将上式作为边界反应磁矢势，使用已有 `solve_aphi(..., outer_bc="prescribed_total")` 实施**边元有向环量边界值**，重新求解内部 A–φ 物理方程。重复指定次数，使边界值与迭代后海水感应电流适配。默认 `open_steps=0` 保留原模型，`open_steps=2` 为本轮实际评估。

此实现复用已有相容 H(curl) 矢势/P1 标量势体系，不另造电流模态或求积参考。当前外边界 Green 积分使用**单元常量电流矩**，后续精度提升应针对该积分及边界迭代本身。

## 3. 同一真实双 BFZI 场景的计算结果

固定两根连续圆角方形 BFZI 各 2 匝、两个轴对齐绝缘封装；200 kHz、海水 σ=5 S/m、每圆角 12 段闭合源、双端口电流 `I=(1,0.4+0.3j) A`；采用已有 `core_radius=0.30 m, near_step=0.12 m, far_step=0.30 m` 近场保持方案。

| 外域半宽 | 四面体 | 原零边界 Psea (W) | 自由空间边界 Psea (W) |
| --- | ---: | ---: | ---: |
| R=0.52 m | 4,224 | 0.00649494771219 | 0.00636132760481 |
| R=0.75 m | 7,800 | 0.00674864457712 | 0.00639975061901 |

外域相邻差，均按 R=0.75 m 的结果归一化：

| 指标 | 零反应场边界 | 2 次自由空间边界修正 |
| --- | ---: | ---: |
| 海水复 2×2 Z 相对差 | **6.3560%** | **5.0133%** |
| 同源端口海水功率相对差 | **3.7592%** | **0.6004%** |

自由空间边界修正显著减少了海水耗散对外域半宽的敏感度。**复阻抗仍有 5.0% 的外域相邻变化，不能判定为收敛。**

### 端口功率和空间损耗一致性

修正后 R=0.52 m 的体积 Joule 功率为 `0.0063639105500383 W`，源—电流体积耦合端口功率为 `0.006361327604811717 W`，相对差 **0.0406%**。

修正后 R=0.75 m 的体积 Joule 功率为 `0.006410723595900431 W`，源—电流体积耦合端口功率为 `0.006399750619012095 W`，相对差 **0.1712%**。

这部分差异与新增边界近似及离散残余有关。原零边界求解的精确代数功率闭合不应直接移植为修正边界方法的验收结果。

原有限域反应势沿闭合导线积分 `z_closed` 保留为独立数值诊断；它在两档修正边界上的端口功率分别为 `0.00558221390895` 和 `0.00565250263726 W`，目前也未完成物理读出精度认证。

## 4. 实际代码与复现

- `sdfmpneo_bfzi/direct3d_primary_secondary.py`：由海水 `J` 构造自由空间边界 A，进行指定次数的边界—内部场迭代。
- `sdfmpneo_bfzi/m1_direct.py`：统一 `solve_m1(..., open_steps=N)`。
- `examples/bfzi_m1_direct.py`：加入 `run.py --mode bfzi-m1 --open-steps N` 参数与输出记录。
- `tests/test_bfzi_m1_direct.py`：确保边界修正改变物理响应、保留有限互易与正耗散，且使源—海水电流读出和体积耗散在测试算例的 1% 内一致。联合相关 D2 测试 **13 passed**。

```bash
python run.py --mode bfzi-m1 --outer-radius 0.52 --core-radius 0.30 --near-step 0.12 --far-step 0.30 --panels 12 --open-steps 2 --output runs/m1_open_R052.json
python run.py --mode bfzi-m1 --outer-radius 0.75 --core-radius 0.30 --near-step 0.12 --far-step 0.30 --panels 12 --open-steps 2 --output runs/m1_open_R075.json
python -m pytest tests/test_bfzi_m1_direct.py tests/test_bfzi_d2_galerkin.py -q
```

[真实数值 JSON](m1_open_boundary_200khz.json)

## 5. 下一项主线及精度状态

本轮已经建立**真实场电流 → 外域 Green 边界 → 三维场重求解**的物理算法链。下一项优先检查同一内核网格下更大 R 的复阻抗变化，以及四面体中心点 Green 边界求积误差；使复阻抗、体积损耗和空间 B/J 共同收敛。

最终的独立高保真物理参考、真正三维场 L² 收敛和正式神经教师认证仍处于 **OPEN**。本方法目前为可选边界实验路径，不替代此前已经通过的 M1 功能闭环。
