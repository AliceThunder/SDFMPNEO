# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-04T14:52:46+00:00
- 结束（UTC）：2026-11-04T14:53:32+00:00
- 当时检出的代码提交：`0863a41`
- 执行：`python run.py --mode bfzi-p1-coulomb`
- 同批 pytest：87 passed in 45.04s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## P1 原 Coulomb 磁能算子：三解析无散度球电流（尚非双盒认证）

- `p1_exact: domain=conducting_ball, radius_m=1,modes=3,divJ=0,wall_normal_J=0,original_kernel=1/r,analytical_L_diag_basis_SI=2.00525295768e-07,analytical_offdiag=0,no_softening=True`
- `p1_decomp: variant=base,x_exact_rel=0.001875,far_y_rel=0.02294011,near_local_l11_rel=0.0002203548,global_const_sub_rel=0.005298825,global_const_sub_L11=2.01145634166e-07`
- `p1_ball: variant=base,x_nodes=120,y_nodes=240,rel_L=0.02120667,diag_rel=0.0322072,offdiag_rel=4.039067e-17,raw_skew=2.281279e-17,L11_basis_SI=1.97004036581e-07,true_L11_basis_SI=2.00525295768e-07`
- `p1_ball: variant=x_only,x_nodes=480,y_nodes=240,rel_L=0.01556394,diag_rel=0.02141865,offdiag_rel=3.209671e-17,raw_skew=1.111072e-17,L11_basis_SI=1.99382459841e-07,true_L11_basis_SI=2.00525295768e-07`
- `p1_ball: variant=y_only,x_nodes=120,y_nodes=756,rel_L=0.01146122,diag_rel=0.0126181,offdiag_rel=3.476739e-17,raw_skew=2.173027e-17,L11_basis_SI=2.03055544745e-07,true_L11_basis_SI=2.00525295768e-07`
- `p1_decomp: variant=both,x_exact_rel=2.473078e-15,far_y_rel=0.006309556,near_local_l11_rel=0.0002269113,global_const_sub_rel=0.001497178,global_const_sub_L11=2.00549170739e-07`
- `p1_ball: variant=both,x_nodes=480,y_nodes=756,rel_L=0.006144858,diag_rel=0.008792633,offdiag_rel=1.340878e-17,raw_skew=1.44222e-17,L11_basis_SI=1.98762150395e-07,true_L11_basis_SI=2.00525295768e-07`
- `p1_partition: near_radius_over_R=0.25,rel_L=0.0169853,diag_rel=0.02537611`
- `p1_partition: near_radius_over_R=0.55,rel_L=0.02113111,diag_rel=0.03211401`
- `p1_taylor: x_nodes=480,y_nodes=756,linear_affine_rel=3.576688e-16,cubic_outer_exact_rel=5.90514e-06,cubic_const_rel=0.002453187,cubic_affine_rel=0.000159619,cubic_affine_pass_0p1pct=True,linear_exact_residual_by_construction=True,cubic_source_nonaffine=True`
- `p1_sub_gate: target_rel=0.001,global_constant_subtraction_frob_pass=False,global_constant_subtraction_rel=0.001497178,domain_scalar_potential=analytic_ball_only`
- `p1_gate: target_rel=0.001,ball_frob_pass=False,ball_diag_pass=False,ball_offdiag_pass=True,partition_L11_gap=0.004230034,partition_L11_pass=False,scope=exact_ball_only`
- `p1_note: p1_decomp splits x-integration, exactly integrated local spherical C2 kernel and y-far cubature; the independent whole-ball constant subtraction has an EXACT ball scalar kernel, which cannot be reused for two-box geometry without obtaining its actual scalar geometry potential. Original 1/r is unchanged. Exact-vortex BALL benchmark tests the ORIGINAL Coulomb double integral against an analytic three-mode matrix. It does NOT certify rotated two-box wet-domain L, its near-wall integrals, or the 229-mode and infinite-ocean physical reference.`

以上只检验不软化 1/r 的解析球域对照。原 BFZI 双绝缘盒的奇异自项、近邻块、R/K、外域及神经教师仍须各自通过独立验收。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
