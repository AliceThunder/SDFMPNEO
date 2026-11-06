# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-06T02:25:28+00:00
- 结束（UTC）：2026-11-06T02:29:05+00:00
- 当时检出的代码提交：`c8e5aca`
- 执行：`python run.py --mode bfzi-p1-full-sea`
- 同批 pytest：110 passed in 52.95s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## P1-C 真实双盒全海水五个 curl 模式原核 L （仅数值对照，尚非物理验收）

- `p1_full_sea_focused: scope=real_5_curl_modes,domain=outer_ball_minus_two_actual_boxes,poses=both_identity,kernel=original_unsoftened_1/r,x_rule=shell_3_4_8_3,background_moment_tolerance=0.0025,local_cap_order=polar_6_azimuth_14,phase_pair=0_and_0.25,source_radial_windows=true,exact_positive_4pi_S_M_moment_fit=true,historical_diagnostics=available_with_--p1-full-history,gate=not_certified`
- `p1_full_sea: variant=source_cap_refined_both,method=radial,x_nodes=4336,inner_rule=inner_radial_4_polar_4_azimuth_10_scheme_uniform_exact_scalar_S_moment_adaptive_angle_source_window_radial_source_angular_caps_cap_polar_6_azimuth_14,norm_L=9.08234654e-08,trace_L=9.443626414e-08,raw_skew=0.0049932978,min_eig=7.294649248e-10,max_eig=9.080478834e-08`
- `p1_full_sea_self: variant=source_cap_refined_both,L00=8.735326826e-10,L11=9.42933046e-10,L22=8.735326826e-10,L33=9.42933046e-10,L44=9.080333268e-08`
- `p1_full_sea_mutual: variant=source_cap_refined_both,L01=-2.035666948e-24,L02=8.503606326e-12,L03=2.260479388e-25,L04=-6.200531365e-25,L12=-2.930110649e-25,L13=2.120124681e-10,L14=2.560432571e-10,L23=-2.646581958e-25,L24=-2.809164297e-25,L34=-2.560432571e-10`
- `p1_full_sea_focused_cap_quality: variant=source_cap_refined_both,azimuth_phase=0,x_points=4336,accepted=3500,fallback=0,unique_charts=3672,positive_cap_nodes=308448,max_angular_measure_rel=3.3306691e-16,max_scalar_moment_rel=0.0024911893,max_first_moment_scaled=0.00091115346,background_refined=1296,background_unresolved=0,gate=not_certified`
- `p1_full_sea_pair_reciprocity: variant=source_cap_refined_both,worst_pair=(1, 4),max_pair_self_normalized=0.024505613,global_raw_skew=0.0049932978,pre_symmetrization=true,gate=not_certified`
- `p1_full_sea_pair_reciprocity_details: variant=source_cap_refined_both,pair14=0.024505613,pair34=0.024505613,pair12=5.2522358e-16,pair23=4.6895796e-16,pair13=1.8397101e-16,pair03=1.7600965e-16,pair24=1.5206923e-16,pair01=9.3988907e-17,pair02=4.4992396e-17,pair04=2.1019812e-18,scope=raw_pair_antisymmetry_relative_to_self_energies`
- `p1_full_sea: variant=source_cap_phase_shift_both,method=radial,x_nodes=4336,inner_rule=inner_radial_4_polar_4_azimuth_10_scheme_uniform_exact_scalar_S_moment_adaptive_angle_source_window_radial_source_angular_caps_cap_polar_6_azimuth_14_cap_phase_0.25,norm_L=9.016805037e-08,trace_L=9.379790043e-08,raw_skew=0.0069170347,min_eig=7.530631293e-10,max_eig=9.014914474e-08`
- `p1_full_sea_self: variant=source_cap_phase_shift_both,L00=8.771049583e-10,L11=9.478698987e-10,L22=8.771049583e-10,L33=9.478698987e-10,L44=9.014795071e-08`
- `p1_full_sea_mutual: variant=source_cap_phase_shift_both,L01=5.612094551e-13,L02=-1.263133158e-11,L03=-2.082836446e-12,L04=-7.308256716e-12,L12=-2.082836446e-12,L13=1.93561315e-10,L14=2.309037132e-10,L23=5.612094551e-13,L24=7.308256716e-12,L34=-2.309037132e-10`
- `p1_full_sea_focused_cap_quality: variant=source_cap_phase_shift_both,azimuth_phase=0.25,x_points=4336,accepted=3500,fallback=0,unique_charts=3672,positive_cap_nodes=308448,max_angular_measure_rel=4.4408921e-16,max_scalar_moment_rel=0.0024911893,max_first_moment_scaled=0.00091115346,background_refined=1296,background_unresolved=0,gate=not_certified`
- `p1_full_sea_pair_reciprocity: variant=source_cap_phase_shift_both,worst_pair=(1, 4),max_pair_self_normalized=0.033695535,global_raw_skew=0.0069170347,pre_symmetrization=true,gate=not_certified`
- `p1_full_sea_pair_reciprocity_details: variant=source_cap_phase_shift_both,pair14=0.033695535,pair34=0.033695535,pair01=0.0020773386,pair23=0.0020773386,pair24=0.0017027954,pair04=0.0017027954,pair12=0.00013979994,pair03=0.00013979994,pair13=4.5753206e-16,pair02=5.2534867e-17,scope=raw_pair_antisymmetry_relative_to_self_energies`
- `p1_full_sea_source_cap_phase_cross: same_x_same_curl_modes_same_cap_order_same_radial_same_exact_4pi_S_M,relative_L=0.0072529373,self_relative=0.0072167762,raw_skew_unshifted=0.0049932978,raw_skew_shifted=0.0069170347,only_local_azimuthal_phase_changed=true,gate=not_certified`
- `p1_full_sea_mode_delta: comparison=source_cap_phase_shift_vs_refined,self0=0.0040894586,self1=0.0052356344,self2=0.0040894586,self3=0.0052356344,self4=0.0072175982,max_self=0.0072175982,max_mutual_self_normalized=0.024194788`
- `p1_full_sea_focused_weak_mutual: L02_unshifted=8.50360632608e-12,L02_shifted=-1.26313315766e-11,L13_unshifted=2.12012468121e-10,L13_shifted=1.93561315041e-10,L14_unshifted=2.56043257077e-10,L14_shifted=2.30903713248e-10,gate=not_certified`
- `p1_full_sea_note: focused_current_bottleneck_only,historical_full_diagnostics=--p1-full-history,independent_x_and_continuum_accuracy_still_unresolved,G1=FAIL`

此运行是实际双盒全海水 curl 模式及原核 L 的数值收敛实验；没有独立连续真值，且角向射线对盒壁/棱角的裁剪仍须单独验证。程序完成、对称、谱正性或分轴差异小，均不能独立证明 G1 PASS。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
