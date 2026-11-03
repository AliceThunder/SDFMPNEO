# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-03T14:07:16+00:00
- 结束（UTC）：2026-11-03T14:08:57+00:00
- 当时检出的代码提交：`8193aa8`
- 执行：`python run.py --mode bfzi-convergence --basis-ball-targeted --basis-ball-check-l`
- 同批 pytest：60 passed in 42.59s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## 嵌套物理空间

| 层级 | 模态数 | 父空间电流上界 | 海水功率 (W) |
|---|---:|---:|---:|
| one_tangent | 15 | inf | 8.2328615e-05 |
| two_tangents | 27 | inf | 0.00010618058 |
| thin_face_patches | 59 | inf | 0.00050007693 |
| guided_port_curls | 63 | inf | 0.00050011153 |
| edge_face_patches | 95 | inf | 0.00050660164 |
| corner_face_patches | 159 | inf | 0.00064835592 |
| source_face_guided | 167 | inf | 0.00064902825 |
| edge_lip_patches | 199 | inf | 0.00065096889 |
| anisotropic_face_curls | 207 | inf | 0.00065199181 |
| global_short_scale | 210 | 0.293046 | 0.0011707542 |
| global_multiscale | 213 | 0.00564875 | 0.0012180167 |
| corner_source_curls | 229 | 2.41273e-13 | 0.0012180773 |
| adaptive_00 | 63 | inf | — |
| adaptive_01 | 71 | 0.433629 | — |
| adaptive_02 | 79 | 0.376312 | — |
| adaptive_03 | 87 | 0.328692 | — |
| adaptive_04 | 95 | 0.268696 | — |
| adaptive_05 | 103 | 0.22081 | — |
| adaptive_06 | 111 | 0.178947 | — |
| adaptive_07 | 119 | 0.132607 | — |
| adaptive_08 | 127 | 0.102532 | — |
| adaptive_09 | 135 | 0.079656 | — |
| adaptive_10 | 143 | 0.0589633 | — |
| adaptive_11 | 151 | 0.0355562 | — |
| adaptive_12 | 159 | 0.0274798 | — |
| adaptive_13 | 167 | 0.0215824 | — |
| adaptive_14 | 175 | 0.0172601 | — |
| adaptive_15 | 183 | 0.0131364 | — |
| adaptive_16 | 191 | 0.0106604 | — |
| adaptive_17 | 199 | 0.00945136 | — |
| adaptive_18 | 207 | 0.00744141 | — |
| adaptive_19 | 215 | 0.0053332 | — |
| adaptive_20 | 223 | 0.0041384 | — |
| adaptive_21 | 229 | 1.49836e-13 | — |

## 相邻空间变化（相对于前一级，不是连续真解误差）

| 阶段 | dZ | dJ_volume | dJ_ports_volume | dq |
|---|---:|---:|---:|---:|
| one_tangent_to_two_tangents | 0.121705 | 0.542916 | 0.414885 | 2.26485 |
| two_tangents_to_thin_face_patches | 1.79218 | 1.92606 | 1.53428 | 355.132 |
| thin_face_patches_to_guided_port_curls | 6.70036e-05 | 0.00831809 | 0.00822483 | 0.000394647 |
| guided_port_curls_to_edge_face_patches | 0.0106665 | 0.113921 | 0.100973 | 0.327308 |
| edge_face_patches_to_corner_face_patches | 0.239214 | 0.528979 | 0.492544 | 0.50494 |
| corner_face_patches_to_source_face_guided | 0.000968301 | 0.0322005 | 0.031283 | 0.0165902 |
| source_face_guided_to_edge_lip_patches | 0.00292632 | 0.0546874 | 0.0542316 | 0.0982027 |
| edge_lip_patches_to_anisotropic_face_curls | 0.00149159 | 0.0396445 | 0.0387992 | 0.030114 |
| anisotropic_face_curls_to_global_short_scale | 1.79013 | 0.892147 | 1.13071 | 0.600084 |
| global_short_scale_to_global_multiscale | 0.0614532 | 0.201029 | 0.226387 | 0.0469662 |
| global_multiscale_to_corner_source_curls | 2.74777e-05 | 0.00706703 | 0.00561699 | 0.0016402 |

## 收敛判定与诊断

- `global_scale_condition_global_short_scale: added=3, min_R_independent_fraction=0.177163, normalized_R_cond=156908`
- `global_scale_condition_global_multiscale: added=3, min_R_independent_fraction=0.117633, normalized_R_cond=157749`
- `global_scale_condition_corner_source_curls: added=16, min_R_independent_fraction=0.293691, normalized_R_cond=166047`
- `ball_229_B_vs_A: dL=0.131987, dZ=0.000735625, dReZ=2.35217e-05, dImZ=0.0222468, dJ=0.000461274, dJ_ports=0.000480709, dJ_volume=0.00153044, dJ_ports_volume=0.00167289, dJ_port0=0.000495128, dJ_port1=0.000465924, dq=0.000577146, max_probe_dq=0.00225707, dP=1.69521e-05, B_on_A_Rinv_residual=0.00167176, min_L_B_eig=1.43433e-10, operator_gate=FAIL, response_gate=PASS, combined_gate=FAIL`
- `ball_229_dual_B_vs_A: dZ=0.000734059, dJ=0.000461126, dq=0.000577213, dP=1.69227e-05`
- `adaptive_screen: FULL_PARENT_ONLY; selected=229/229, matched all fixed heat probes to <=5%, both full-volume J norms to <=1%, plus port and aggregate heat criteria relative to this parent; NOT continuum-certified`

**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
