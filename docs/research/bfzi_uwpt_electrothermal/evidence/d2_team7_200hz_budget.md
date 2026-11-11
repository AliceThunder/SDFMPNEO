# TEAM-7 200 Hz 同终点 h/R/Δt 数值误差预算

BFZI-PDR V1.1 · 2026-11-11 · D2 OPEN · 无实验拟合

## 数据及比较口径

全部来自官方 TEAM-7 真实 Elmer 26.2 三维 BDF2 四周期计算，同一物理终点 t=0.020 s、固定 101 点测线与对应场定义。h/R 扫描为 64 步；时间扫描在 h10、R=0.20m 上采用 64/128/256 步。数据来自[原有空间 h/R JSON](d2_team7_numerical_conv.json)及[200 Hz 实际场时间 JSON](d2_team7_200hz_field_time.json)。本轮 `examples/bfzi_d2_team7_budget.py` 逐项报告完整的同观测量配对、相邻变化与空间/时间比值：[机器结果 JSON](d2_team7_200hz_budget.json)。

| 观测场 | h10 的 64→128 时间差 | 两档 h 相邻差 | 两档 R 相邻差 |
|---|---:|---:|---:|
| A1 Bz | 0.2190% | 13.5350% / 12.7059% | 12.2332% / 14.2841% |
| A1 Bxyz | 0.1393% | 8.2942% / 7.6816% | 7.3756% / 8.0044% |
| A2 Bz | 0.2588% | 23.0022% / 19.8957% | 12.0894% / 11.3899% |
| 上表面 Jy | 1.6867% | 4.2093% / 3.3185% | 2.5066% / 1.6529% |
| 下表面 Jy | 3.4054% | 7.8343% / 6.1540% | 4.1776% / 5.1634% |

A1 Bz 的 h/R 相邻差为 h10 时间差的约 56–65 倍，A2 Bz 为约 44–89 倍。**下一项应优先改善三维空气/线圈邻近的 H(curl) 空间场精度并独立收敛 R。** J 的 h/R 差异约为时间差的 0.98–2.50 倍，必须保留独立 Δt 误差预算。上述各项均为数值解相邻变化，尚未认证连续极限。

## 实现：原始 Whitney 边自由度的 BDF2 电场与损耗

新增 `sdfmpneo_bfzi/direct3d_whitney.py::bdf2_joule`：原始有向边自由度历史与 P1 导体势构成积分点瞬时电场：

\[
E_n=-\sum_e\frac{3a_{e,n}-4a_{e,n-1}+a_{e,n-2}}{2\Delta t}N_e-\nabla\phi_n,\qquad
P_n=\int_{\Omega_c}\sigma |E_n|^2\,dV .
\]

四面体四点求积计算 E/q/P，绝缘四面体功率为零。制造解验证二阶时间精度、时变规范不变性及积分，原有真实 TEAM-7 自研边元求解验证原始 Whitney curl B。XEnergy 当轮相关 **6 passed**（含实场时间对齐、h/R 时间预算测试）。

[ElmerCSC 2026-06-22 源码版本 CalcFields.F90](https://github.com/ElmerCSC/elmerfem/blob/02e7cd36e1a56da9354c03ba4717b8b838588c51/fem/src/modules/MagnetoDynamics/CalcFields.F90) 约第 1152–1156 行的 `pSolver % TimeOrder==1` 分支采用 `(A_n-A_{n-1})/dt`，约第 1599 行用于 E、2011–2024 行用于涡流功率。该源码版本早于 XEnergy Elmer 26.2 二进制 2026-08-05 编译日期，与真实 B 二阶、J 一阶观测和功率相位延迟现象相符。二进制当前运行时 `TimeOrder` 分支和原始 Elmer 边/φ 历史仍待取出核实。**新 BDF2 核没有冒充已经完成 Elmer 原生积分替换。**

## 复现

```powershell
python -m pytest tests/test_bfzi_d2_whitney.py tests/test_bfzi_d2_team7_field_time.py tests/test_bfzi_d2_team7_budget.py -q
python -m examples.bfzi_d2_team7_field_time --root D:/work/BFZI-D2-verify/runs/team7_hr --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_200hz_field_time.json
python -m examples.bfzi_d2_team7_budget --spatial docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_numerical_conv.json --time docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_200hz_field_time.json --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_200hz_budget.json
```

**状态：TEAM-7 数值精度 OPEN，D2 OPEN，旧 G1 FAIL，神经代理教师冻结。**
