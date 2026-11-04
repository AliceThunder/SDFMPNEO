# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-04T07:57:12+00:00
- 结束（UTC）：2026-11-04T07:58:32+00:00
- 当时检出的代码提交：`a95a367`
- 执行：`python run.py --mode bfzi-g1-foundations`
- 同批 pytest：65 passed in 39.18s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## G1 三条基础路线的实验基准（尚非认证）

- `g1_structured_basis: base=27, level1=71, level2=211, compact_level1=44`
- `g1_coulomb_patch: modes=5, clip_vs_clearance_dL=0.047611, face_panels_vs_uniform_dL=0.00789322, radius_055_to_080_dL=0.0208704, raw_skew_clearance=0.00246786, raw_skew_unpanel=0.00280427, raw_skew_clip_055=0.00283556, raw_skew_clip_080=0.0021507`
- `g1_coulomb_axes: modes=5, dL_x_only=0.111862, dL_y_only=0.013673, dL_both=0.0114144, nonadditive_dL=0.127241, raw_skew_base=0.00283556, raw_skew_x_only=0.000344436, raw_skew_y_only=0.00221105, raw_skew_both=0.000398456, dimensionless_dL_x=0.00503948, dimensionless_dL_y=0.00113808, x_cubature_dR=0.0102768, x_cubature_dK=0.0395661, dZ_x_only=0.004642, dZ_y_only=0.00012835, dq_x_only=0.00081723, dq_y_only=0.000122256`
- `g1_coulomb_split: modes=5, x_nodes=4008, y_nodes=4584, local_angular_dL=0.00180892, ball_vs_split_dL=0.0794068, local_angular_dZ=1.15794e-05, ball_vs_split_dZ=0.00402279, local_angular_dJ_R=1.1802e-05, volume_x_nodes=4584, volume_y_nodes=6124, volume_dL=0.0114144, dimensionless_dL=0.00107494, refined_on_coarse_residual=0.000799618, volume_dZ=0.000697653, volume_probe_dq=0.000324187, min_L_split_eig=8.73551e-10`
- `g1_coulomb_note: this is a 5-mode fixed-domain, unsoftened singularity-split pilot; x/y cubatures, face-aligned angular panels and near-patch radius must all converge independently. No 229-mode certificate.`
- `g1_exact_exterior: outer_radius=0.52, skin_depth=0.503292, dirichlet_rel_dJ=0.0934458, transparent_log_derivative=-4.66172-1.60492j`
- `g1_exact_exterior: outer_radius=0.85, skin_depth=0.503292, dirichlet_rel_dJ=0.0148812, transparent_log_derivative=-3.47714-1.78985j`
- `g1_exact_exterior: outer_radius=1.3, skin_depth=0.503292, dirichlet_rel_dJ=0.00199346, transparent_log_derivative=-2.89742-1.88508j`
- `g1_exterior_multipole: lmax=3, boundary_projection_error=6.4802e-16, l1_l2_tail_error=4.56147e-16, tail_ohmic_power_for_unit_trace=0.0470848`
- `g1_exterior_note: multiple exact toroidal exterior harmonics and vector spherical-harmonic continuation now implemented; their DtN must still be coupled to interior BFZI J at a matching sphere with divergence compatibility. G1 infinite-domain FAIL remains.`

Green L 当前仍为小空间原核积分试验，新增真实海水几何裁剪与分割半径对照；可选 J/q 仅为选取方向的嵌套实验；已实现均匀外海水的球面多极 DtN 和 Joule 尾场，但尚未耦合完整双盒求解。所有 G1 物理精度仍待独立验收。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
