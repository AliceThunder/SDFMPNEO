# D2：原始 Elmer Whitney 旋度 B 与原生磁能量同源验证

BFZI-PDR V1.1，2026-11-11。真实 TEAM-7 h10 三维网格（22,851 节点、129,305 四面体、155,796 边）上，实际完成 BDF2 200 Hz **第一时间步与完整四周期终点**两项 Elmer 原生求解。使用相同材料、导体、源及原始 3D 网格，无实验拟合。

## 原始自由度重建与独立积分

`examples/bfzi_d2_team7_raw_b.py` 从 Elmer ASCII `.result` 提取真实 `A` 原始全局自由度（共 178,647 = 22,851 节点势 + 155,796 个边势），根据 Elmer `MeshBasics.F90::FindMeshEdges3D` 的四面体边创建顺序、min-to-max 边方向，在每个四面体构造原始 Whitney 旋度 B，并自行积分：
\[
B_K=\sum_{e\subset K} a_e\nabla\times N_e,\qquad
W_B=\sum_K\frac{|K|\,|B_K|^2}{2\mu_0}.
\]

**关键一致性：** 原始 CSC TEAM-7 SIF 的 `Permeability of Vacuum = 1.256e-6`。代码直接读取同一 SIF 的 \(\mu_0\)，与 Elmer 保持完全相同的物理常数。此前独立积分采用 \(4\pi\times10^{-7}\) 时出现的 0.0506957% 差异完全来自物理常数差异，已在代码和机器证据中消除。

| 同一原始网格与物理时间 | t=0.0003125 s（第一步） | t=0.0200000 s（四周期终点） |
|---|---:|---:|
| 原始 Whitney B 独立积分 (J) | 0.453905102235415 | 0.523786635518865 |
| Elmer 原生 `SaveScalars` 磁能量 (J) | 0.453905102235500 | 0.523786635518900 |
| 相对差 | **1.87236×10⁻¹³** | **6.71916×10⁻¹⁴** |

这是两个真实 Elmer 时间状态下，原始边自由度恢复与内置能量积分的浮点精度一致性验证。无需使用 DG/VTU 节点重建或采样。`tests/test_bfzi_d2_team7_raw_b.py` 中制造常量 B 的边顺序、方向、末时间状态及磁导率测试在 XEnergy **1 passed**。

机器证据：[第一步 JSON](d2_team7_whitney_raw_b.json)、[四周期 JSON](d2_team7_whitney_raw_b64.json)。Elmer 本机求解 SIF/日志为 `case_edge1.sif`/`edge1.log` 与 `case_edge64.sif`/`edge64.log`；大体积原生数据保留在 XEnergy `D:/work/BFZI-D2-verify/runs/team7_hr/h10/TEAM7/edge.result` 及 `edge64.result`。

**验收状态：** 原始 Whitney curl B 能量积分的同源实现一致性已通过；不同 h/R 网格的原始 B 场数值极限仍 OPEN。TEAM-7 数值精度、D2 继续 OPEN，旧 G1 FAIL，神经代理教师冻结。
