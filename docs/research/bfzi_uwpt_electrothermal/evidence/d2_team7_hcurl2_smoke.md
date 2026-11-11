# D2 TEAM-7 h10：二阶 H(curl) 边元一步求解试算

BFZI-PDR V1.1 · 2026-11-11 · **SMOKE 未通过，D2 OPEN**

## 目标与冻结条件

在真实 Elmer 26.2 官方 TEAM-7 h10 129305 四面体、200 Hz、R=0.20m 和原三分量线圈源不变的条件下，隔离二阶边元空间离散这一轴。基准文件为 XEnergy 的 `D:/work/BFZI-D2-verify/runs/team7_hr/h10/case_ene200s64.sif`，原为 BDF2 64 步四周期。为快速检查二阶有限元可求解性，本轮保持 `dt=0.0003125s`，仅运行第一时间步；这一步仅用于 smoke，不进入四周期空间收敛比较。

依据[ElmerCSC WhitneyAVSolver.F90](https://github.com/ElmerCSC/elmerfem/blob/02e7cd36e1a56da9354c03ba4717b8b838588c51/fem/src/modules/MagnetoDynamics/WhitneyAVSolver.F90) 与 [ElemInfo.F90](https://github.com/ElmerCSC/elmerfem/blob/02e7cd36e1a56da9354c03ba4717b8b838588c51/fem/src/ElemInfo.F90)，`Quadratic Approximation = Logical True` 选用二阶边元基及 Piola 模式。因此在 `Solver 2`（WhitneyAVSolver）和 `Solver 3`（MagnetoDynamicsCalcFields）均加入：

```text
Quadratic Approximation = Logical True
Use Piola Transform = Logical True
```

输入 SIF 与日志在 XEnergy `runs/team7_hr/h10/case_quad1.sif`、`quad1.log`、`case_quadilu.sif`、`quadilu.log`。原来的 64/128/256 步 SIF、原生能量和场文件未修改。

## 两次真实 ElmerSolver 一步执行

| 配置 | Whitney Solver 2 预条件 | 实际迭代证据 | 结果 |
|---|---|---|---|
| `case_quad1.sif` | 原始 none | BiCGStabL 第 10/40/100 次残差约 1.172e−1 / 3.637 / 98.39 | 发散，人工停止 |
| `case_quadilu.sif` | ILU0，其余源/网格相同 | 第 10/40/50 次残差约 5.783e7 / 3.372e12 / 1.945e13 | 发散，人工停止 |

两个试算均已真正启动 ElmerSolver、读取到线性求解器迭代日志。均未完成首步原生积分，**没有可供比较的二阶 B/J/能量数据**。这一轮否定的是当前两种迭代配置，尚未评价二阶 H(curl) 离散在合适规范/预条件处理下的精度。

## 决策

当前 BFZI-PDR D2 继续采用已实算的原始一阶 Whitney 三维参考，优先改善空气域/线圈邻近实体网格和直接 curl B 恢复；二阶边元在确认稳定线性系统（边元零空间/规范、求解器与预条件器配置）后再进入 200 Hz 四周期 h/R 对照。已完成的真实 [同终点 h/R/Δt 预算](d2_team7_200hz_budget.md) 表明 B 的 h/R 空间相邻差远大于 h10 时间差；J 的独立时间后处理预算继续保留。

**TEAM-7 数值精度 OPEN；D2 OPEN；旧 G1 FAIL；神经代理教师冻结。**
