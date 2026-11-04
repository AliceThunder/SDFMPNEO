# BFZI 最近一次本地运行摘要

- 运行状态：**PASS（程序完成；不代表物理收敛）**
- 开始（UTC）：2026-11-04T06:45:05+00:00
- 结束（UTC）：2026-11-04T06:45:58+00:00
- 当时检出的代码提交：`cd73a65`
- 执行：`python run.py --mode bfzi-g1-foundations`
- 同批 pytest：62 passed in 43.75s
- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。
- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。

## G1 三条基础路线的实验基准（尚非认证）

- `g1_structured_basis: base=27, level1=71, level2=211, compact_level1=44`
- `g1_coulomb_split: modes=5, x_nodes=4008, y_nodes=4584, local_angular_dL=4.51102e-07, ball_vs_split_dL=0.0778894, local_angular_dZ=9.46672e-10, ball_vs_split_dZ=0.00481719, local_angular_dJ_R=1.27697e-09, volume_x_nodes=4584, volume_y_nodes=6124, volume_dL=0.0298039, dimensionless_dL=0.00196389, refined_on_coarse_residual=0.00193254, volume_dZ=0.00182062, volume_probe_dq=0.00058998, min_L_split_eig=1.04838e-09`
- `g1_coulomb_note: this is a 5-mode fixed-domain, unsoftened singularity-split pilot; both x/y volume cubatures and local angles are compared, not yet converged. No 229-mode certificate.`
- `g1_exact_exterior: outer_radius=0.52, skin_depth=0.503292, dirichlet_rel_dJ=0.0934458, transparent_log_derivative=-4.66172-1.60492j`
- `g1_exact_exterior: outer_radius=0.85, skin_depth=0.503292, dirichlet_rel_dJ=0.0148812, transparent_log_derivative=-3.47714-1.78985j`
- `g1_exact_exterior: outer_radius=1.3, skin_depth=0.503292, dirichlet_rel_dJ=0.00199346, transparent_log_derivative=-2.89742-1.88508j`
- `g1_exterior_note: exact l=1 toroidal spherical-cavity MQS transparent boundary is not a general BFZI two-box DtN map.`

Green L 独立奇异拆分当前仅在小试探空间运行；紧支 curl 层级仅证明构造/嵌套，尚未证明逼近收敛；解析透明边界只适用于球腔偶极的 l=1 扇区。

## 收敛判定与诊断


**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，都不能单独构成连续物理参考解的精度证明。
