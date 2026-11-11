# D2：原始 Elmer Whitney 边自由度与旋度 B 独立积分

BFZI-PDR V1.1，2026-11-11。真实官方 TEAM-7 h10 三维网格：22,851 节点、129,305 四面体、155,796 有向边。原材料、闭合线圈源、200 Hz 和 BDF2 保持不变。

## 实际实现与数值结果

采用独立的 Elmer 一步运行（t=0.0003125 s），在 Simulation 添加 `Output File = "edge.result"`，导出原生 `a` 变量，包含 **178,647 个 A–φ 自由度**。原始文件保存在 XEnergy `D:/work/BFZI-D2-verify/runs/team7_hr/h10/TEAM7/edge.result`；输入、日志及原生积分分别为 `case_edge1.sif`、`edge1.log` 和 `res_edge1/phys.dat`。

新增 `examples/bfzi_d2_team7_raw_b.py`，读取真实 A 全局自由度，根据 Elmer `MeshBasics.F90::FindMeshEdges3D` 的 TetraEdgeMap、全局首次出现编号及 min-to-max 边方向，在每个四面体直接计算 Whitney 旋度并积分：
\[
B_K=\sum_{e\subset K}a_e(\nabla\times N_e),\qquad
W_B=\sum_K\frac{|K|\,|B_K|^2}{2\mu_0}.
\]
不经过 DG/VTU 节点重建。`tests/test_bfzi_d2_team7_raw_b.py` 对制造均匀 B 场检验真实边编号算法、边方向及体积分，XEnergy **1 passed**。

| 同源同时间步的磁能量 | J |
|---|---:|
| 原始 Whitney curl B 独立体积分 | **0.453674991693** |
| Elmer 原生 SaveScalars 积分 | **0.453905102236** |
| 相对差 | **0.0506957383%** |

机器可读[原始场数值证据](d2_team7_whitney_raw_b.json)。保留 0.0507% 残余差以继续检查积分实现一致性。相同步的 DG 导出节点场积分差约 4.75%，不能用其替代原始 Whitney 旋度。

下一项是四周期终点原始边数据验收与 h/R 跨网格原始 B 数值误差。TEAM-7 数值精度与 D2 **OPEN**，旧 G1 FAIL，神经代理教师冻结。
