# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-05T03:10:21+00:00
- 结束（UTC）：2026-11-05T03:11:07+00:00
- 当时检出的代码提交：`a9f48db`
- 执行：`python run.py --mode bfzi-p1-full-sea`
- 同批 pytest：97 passed in 43.72s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## P1 原 Coulomb 磁能：球解析真值、双盒几何矩、两紧支涡流（尚非全域教师认证）

- `p1_full_sea_exact: physical_domain=outer_ball_minus_two_actual_boxes,modes=5,mode_indices=0-1-12-13-24,current=existing_box_following_curl_basis,domain_S=exact_true_rotated_geometry,method=symmetric_difference_unsoftened_1/r,comparison=global_affine_Taylor,gate=convergence_study_not_certified`
- `p1_full_sea: variant=base,method=difference,x_nodes=384,y_nodes=720,norm_L=7.907577091e-08,trace_L=8.00955972e-08,subtraction_ratio=0.9812443,min_eig=-5.137240687e-10,max_eig=7.906078106e-08`
- `p1_full_sea_self: variant=base,L00=-4.610254011e-10,L11=9.791743888e-10,L22=-4.610254011e-10,L33=9.791743888e-10,L44=7.905929922e-08`
- `p1_full_sea_mutual: variant=base,L01=-3.151505194e-25,L02=-5.26986676e-11,L03=1.003945801e-25,L04=-9.342985166e-25,L12=4.17587243e-26,L13=1.113643953e-10,L14=-2.406961825e-10,L23=3.560502994e-25,L24=-1.295621596e-24,L34=2.406961825e-10`
- `p1_full_sea: variant=x_only,method=difference,x_nodes=1320,y_nodes=720,norm_L=8.024264573e-08,trace_L=8.186248019e-08,subtraction_ratio=0.9817608,min_eig=-2.660792537e-11,max_eig=8.02353731e-08`
- `p1_full_sea_self: variant=x_only,L00=6.844157211e-11,L11=7.451334192e-10,L22=6.844157211e-11,L33=7.451334192e-10,L44=8.023533021e-08`
- `p1_full_sea_mutual: variant=x_only,L01=7.582825473e-25,L02=-9.504949748e-11,L03=-5.355498158e-26,L04=3.936585622e-26,L12=7.747042745e-27,L13=1.209675389e-10,L14=4.131848258e-11,L23=3.531174478e-25,L24=-1.666113807e-25,L34=-4.131848258e-11`
- `p1_full_sea: variant=y_only,method=difference,x_nodes=384,y_nodes=2352,norm_L=9.758095521e-08,trace_L=1.019730079e-07,subtraction_ratio=0.9834941,min_eig=9.17534729e-10,max_eig=9.75556298e-08`
- `p1_full_sea_self: variant=y_only,L00=1.11519793e-09,L11=1.094019155e-09,L22=1.11519793e-09,L33=1.094019155e-09,L44=9.755457375e-08`
- `p1_full_sea_mutual: variant=y_only,L01=8.265688091e-25,L02=2.46389114e-11,L03=4.098448325e-25,L04=1.396316608e-24,L12=-8.190146855e-26,L13=1.754283827e-10,L14=-2.258903553e-10,L23=6.005802409e-24,L24=-8.185726402e-25,L34=2.258903553e-10`
- `p1_full_sea: variant=both,method=difference,x_nodes=1320,y_nodes=2352,norm_L=1.013685079e-07,trace_L=1.072654497e-07,subtraction_ratio=0.9833488,min_eig=1.101663156e-09,max_eig=1.013236583e-07`
- `p1_full_sea_self: variant=both,L00=1.718585735e-09,L11=1.252326091e-09,L22=1.718585735e-09,L33=1.252326091e-09,L44=1.01323626e-07`
- `p1_full_sea_mutual: variant=both,L01=8.432108207e-24,L02=-2.863742892e-11,L03=7.850804281e-26,L04=-2.216010244e-25,L12=-7.28646858e-27,L13=1.506306707e-10,L14=4.020939877e-11,L23=3.085952785e-24,L24=5.217722955e-26,L34=-4.020939877e-11`
- `p1_full_sea_change: method=difference,x_only_vs_base=0.01948965,y_only_vs_base=0.2356018,both_vs_base=0.2843744,both_vs_x_only=0.264565,both_vs_y_only=0.04005104,self_x=0.01812127,self_y=0.2355996,self_both=0.2842904,mutual_x=0.00507344,mutual_y=0.001297302,mutual_both=0.005057438,pass_status=not_determined_without_independent_truth`
- `p1_full_sea_cross: variant=both,difference_vs_taylor_unsoftened=1/r,relative_L=0.8094096,self_relative=0.8094096,mutual_scaled=0.0006784406,taylor_raw_skew=0.01090924,taylor_min_eig=9.988316472e-10`
- `p1_full_sea_note: all five currents and both rotated insulating solids are unchanged across rules and formulations. The difference integral is symmetric by identity (raw_skew is not a gate); near-wall/edge angular x and y cubature, true reference L and infinite ocean remain unverified. Ball analytic tests alone cannot certify BFZI physical accuracy or teacher training.`

以上只检验不软化 1/r 的解析球域对照。原 BFZI 双绝缘盒的奇异自项、近邻块、R/K、外域及神经教师仍须各自通过独立验收。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
