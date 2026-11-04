# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-04T10:54:24+00:00
- 结束（UTC）：2026-11-04T10:55:07+00:00
- 当时检出的代码提交：`ce086b1`
- 执行：`python run.py --mode bfzi-p0-source`
- 同批 pytest：77 passed in 41.67s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## P0 物理源：独立直线解析与曲线 Gauss 收敛（非电磁教师认证）

- `p0_physics: model=linear_MQS, source=prescribed_closed_ideal_line, f_hz=200000,sigma_s=5,mu_rel=1, sea_skin_depth_m=0.5032921, omega_eps_over_sigma_if_epsr80=0.000178024, air_k0_R=0.002179679, insulating_boxes=2,outer_sphere=artificial_no_normal_current, finite_wire_proximity=excluded,displacement_current=excluded`
- `p0_reference: port=0,probe_count=28, corner_24_to_32_dC=3.23941e-16, corner_24_to_32_dB=2.79896e-16`
- `p0_source: port=0,order=3, dC=0.000507298,dB=0.00114483, dC_near=0.000508098, dB_near=0.00114491, legacy_dC=0.00296249, legacy_dB=0.031072, line_closure_over_length=3.45706e-08`
- `p0_source: port=0,order=4, dC=2.24176e-05,dB=0.000195457, dC_near=2.24532e-05, dB_near=0.000195471, legacy_dC=0.00252616, legacy_dB=0.0117353, line_closure_over_length=9.70638e-11`
- `p0_source: port=0,order=6, dC=3.08869e-07,dB=1.74811e-06, dC_near=3.09379e-07, dB_near=1.74823e-06, legacy_dC=8.68585e-05, legacy_dB=0.00466037, line_closure_over_length=1.40757e-16`
- `p0_source: port=0,order=8, dC=2.47805e-09,dB=5.23552e-09, dC_near=2.48214e-09, dB_near=5.23589e-09, legacy_dC=6.34857e-05, legacy_dB=0.000943337, line_closure_over_length=6.18479e-17`
- `p0_source: port=0,order=12, dC=8.38606e-14,dB=1.93691e-12, dC_near=8.39991e-14, dB_near=1.93705e-12, legacy_dC=1.17728e-06, legacy_dB=3.80127e-05, line_closure_over_length=6.77102e-17`
- `p0_gate: port=0,source_order=4, source_dC_le_0p1pct=True, source_dB_le_0p1pct=True`
- `p0_gate: port=0,source_order=12, source_dC_le_0p1pct=True, source_dB_le_0p1pct=True`
- `p0_reference: port=1,probe_count=28, corner_24_to_32_dC=3.22292e-16, corner_24_to_32_dB=2.79523e-16`
- `p0_source: port=1,order=3, dC=0.000507054,dB=0.00114486, dC_near=0.000508097, dB_near=0.0011449, legacy_dC=0.00296077, legacy_dB=0.0310729, line_closure_over_length=3.45706e-08`
- `p0_source: port=1,order=4, dC=2.24063e-05,dB=0.000195463, dC_near=2.24532e-05, dB_near=0.00019547, legacy_dC=0.00252468, legacy_dB=0.0117356, line_closure_over_length=9.70638e-11`
- `p0_source: port=1,order=6, dC=3.08689e-07,dB=1.74816e-06, dC_near=3.09379e-07, dB_near=1.74822e-06, legacy_dC=8.68078e-05, legacy_dB=0.00466051, line_closure_over_length=1.40757e-16`
- `p0_source: port=1,order=8, dC=2.4766e-09,dB=5.23568e-09, dC_near=2.48214e-09, dB_near=5.23587e-09, legacy_dC=6.34486e-05, legacy_dB=0.000943366, line_closure_over_length=6.18479e-17`
- `p0_source: port=1,order=12, dC=8.38116e-14,dB=1.93697e-12, dC_near=8.39989e-14, dB_near=1.93704e-12, legacy_dC=1.17659e-06, legacy_dB=3.80138e-05, line_closure_over_length=6.77102e-17`
- `p0_gate: port=1,source_order=4, source_dC_le_0p1pct=True, source_dB_le_0p1pct=True`
- `p0_gate: port=1,source_order=12, source_dC_le_0p1pct=True, source_dB_le_0p1pct=True`
- `p0_note: C/B fields are checked against analytic straight lines + independent high-order curved-corner Gauss, NOT against a full-wave/copper-proximity or infinite-seawater solution. No source order is certified until the measured differences meet a predeclared accuracy target.`

P0 只检验已指定的线圈/回流源与磁准静态尺度；线元 Gauss 闭合或 C/B curl 同源不足以证明全 Maxwell、海水 R/K/L 或无限外域精度。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
