# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-05T03:32:55+00:00
- 结束（UTC）：2026-11-05T03:34:13+00:00
- 当时检出的代码提交：`8c26357`
- 执行：`python run.py --mode bfzi-p1-full-sea`
- 同批 pytest：99 passed in 47.16s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## P1-C 真实双盒全海水五个 curl 模式原核 L （仅数值对照，尚非物理验收）

- `p1_full_sea_exact: domain=outer_ball_minus_two_actual_rotated_boxes,modes=5,mode_indices=0-1-12-13-24,kernel=original_unsoftened_1/r,inner=point_centered_exact_wet_ray_intervals,inner_measure=r_dr_dsolidangle,outer=geometry_aligned_positive_box_shell_cubature,wall_normal=log_panels,gate=not_certified`
- `p1_full_sea: variant=base,method=radial,x_nodes=2716,inner_rule=inner_radial_3_polar_3_azimuth_8,norm_L=4.791769187e-08,trace_L=5.050727444e-08,raw_skew=0.02347326,min_eig=3.747919685e-10,max_eig=4.78985733e-08`
- `p1_full_sea_self: variant=base,L00=5.213758672e-10,L11=7.738013133e-10,L22=5.35827356e-10,L33=7.779127523e-10,L44=4.789835715e-08`
- `p1_full_sea_mutual: variant=base,L01=-3.203197291e-25,L02=-1.536398221e-10,L03=-1.348744881e-25,L04=-8.76460913e-25,L12=3.027630055e-25,L13=1.050967683e-10,L14=-6.132118957e-11,L23=1.696576305e-25,L24=3.707695544e-25,L34=8.029299847e-11`
- `p1_full_sea: variant=x_only,method=radial,x_nodes=4336,inner_rule=inner_radial_3_polar_3_azimuth_8,norm_L=5.053674333e-08,trace_L=5.394680769e-08,raw_skew=0.02176694,min_eig=7.749438156e-10,max_eig=5.050735644e-08`
- `p1_full_sea_self: variant=x_only,L00=8.461027502e-10,L11=8.772781146e-10,L22=8.461027502e-10,L33=8.772781146e-10,L44=5.050004596e-08`
- `p1_full_sea_mutual: variant=x_only,L01=4.62902666e-26,L02=7.115893453e-11,L03=9.737974002e-28,L04=-7.125765142e-25,L12=-1.068491059e-25,L13=1.9053606e-11,L14=4.26003949e-10,L23=-3.984507767e-26,L24=1.065854469e-24,L34=-4.26003949e-10`
- `p1_full_sea: variant=inner_only,method=radial,x_nodes=2716,inner_rule=inner_radial_4_polar_4_azimuth_10,norm_L=6.509124132e-08,trace_L=6.776012064e-08,raw_skew=0.02002767,min_eig=4.924858296e-10,max_eig=6.507697887e-08`
- `p1_full_sea_self: variant=inner_only,L00=5.937453069e-10,L11=7.531987194e-10,L22=5.907594909e-10,L33=7.508919714e-10,L44=6.507152515e-08`
- `p1_full_sea_mutual: variant=inner_only,L01=2.105731165e-26,L02=9.975539872e-11,L03=-2.408729242e-26,L04=-2.126355386e-24,L12=-2.914149551e-25,L13=7.514136008e-11,L14=4.11343098e-10,L23=-1.005081804e-25,L24=1.346547632e-24,L34=-4.266339108e-10`
- `p1_full_sea: variant=inner_radial_only,method=radial,x_nodes=4336,inner_rule=inner_radial_4_polar_3_azimuth_8,norm_L=6.581031382e-08,trace_L=6.888271355e-08,raw_skew=0.01031976,min_eig=7.013725518e-10,max_eig=6.579209074e-08`
- `p1_full_sea_self: variant=inner_radial_only,L00=7.251044669e-10,L11=8.243550298e-10,L22=7.251044669e-10,L33=8.243550298e-10,L44=6.578379456e-08`
- `p1_full_sea_mutual: variant=inner_radial_only,L01=-8.345381258e-27,L02=2.373191511e-11,L03=-1.217864105e-27,L04=-9.331985023e-25,L12=-1.641481986e-25,L13=5.065427764e-13,L14=5.191282677e-10,L23=1.179715083e-26,L24=1.235980251e-24,L34=-5.191282677e-10`
- `p1_full_sea: variant=both,method=radial,x_nodes=4336,inner_rule=inner_radial_4_polar_4_azimuth_10,norm_L=6.475686132e-08,trace_L=6.801801557e-08,raw_skew=0.01745615,min_eig=6.975082334e-10,max_eig=6.473576119e-08`
- `p1_full_sea_self: variant=both,L00=8.125184958e-10,L11=8.397474365e-10,L22=8.125184958e-10,L33=8.397474365e-10,L44=6.471348371e-08`
- `p1_full_sea_mutual: variant=both,L01=1.325891185e-25,L02=-4.679579858e-11,L03=7.39678549e-26,L04=-2.07758092e-25,L12=3.515229744e-25,L13=1.199617238e-10,L14=8.444271931e-10,L23=1.329125225e-26,L24=2.620493891e-24,L34=-8.444271931e-10`
- `p1_full_sea_inner_axes: at_fixed_x=x1,radial_step=0.3024768,angle_step=0.01935873,self_radial_step=0.3024947,self_angle_step=0.01637708,mutual_radial_step=0.002794017,mutual_angle_step=0.007301322`
- `p1_full_sea_change: method=radial,x_only_vs_base=0.05937477,inner_only_vs_base=0.3590569,both_vs_base=0.3531091,both_vs_x_only=0.2817736,both_vs_inner_only=0.01545355,self_x=0.0551804,self_inner=0.3584016,self_both=0.3510289,mutual_x=0.0155017,mutual_inner=0.01541341,mutual_both=0.02710676,pass_status=not_certified_without_independent_reference`
- `p1_full_sea_note: previous difference pilot had 23.56018-percent y-axis L change, 80.94096-percent Taylor cross-formulation gap, and 98-percent subtraction ratio. New radial Green operator has no large matrix subtraction. It must still pass independent x/inner angular/radial and near-wall/corner convergence; a small raw skew or one PSD matrix is not a true solution certificate. G1 remains FAIL.`

此运行是实际双盒全海水 curl 模式及原核 L 的数值收敛实验；没有独立连续真值，且角向射线对盒壁/棱角的裁剪仍须单独验证。程序完成、对称、谱正性或分轴差异小，均不能独立证明 G1 PASS。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
