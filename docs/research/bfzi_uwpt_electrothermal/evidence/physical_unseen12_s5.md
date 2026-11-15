# BFZI V150 U01–U12 独立物理直接法四组合验收

当前结果：通过 2/12，失败 0/12，待完成 10/12；总验收 OPEN。

对每个工况进行 ultra/superfine 两档原生湿 FV × 1.875/1 s 真实热步，统一 15 s 阻抗采样，完整 600 s。

## 全部样本

| 样本 | 状态 | 时间 ultra | 时间 superfine | 空间 1.875 s | 空间 1 s | 铜/封装/海水最高温 (K) |
| --- | --- | --- | --- | --- | --- | --- |
| U01 | pass | 0.161% / 0.142 / 0.021 | 0.159% / 0.142 / 0.021 | 1.489% / 0.056 / 0.087 | 1.488% / 0.056 / 0.087 | 345.10/322.68/300.55 |
| U02 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U03 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U04 | pass | 0.154% / 0.400 / 0.048 | 0.155% / 0.401 / 0.048 | 1.032% / 0.623 / 0.892 | 1.032% / 0.622 / 0.891 | 435.67/372.97/322.89 |
| U05 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U06 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U07 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U08 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U09 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U10 | pending | 待运行 | 待运行 | 0.838% / 0.331 / 0.486 | 待运行 | 384.64/345.24/311.66 |
| U11 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| U12 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

表中四类误差分别为动态复阻抗相对 L2、铜最高逐时刻绝对差 K、封装最高逐时刻绝对差 K。
时间门槛：1%、0.5 K、0.5 K。空间门槛：2%、1 K、1.5 K。

## 已有四组合误差统计

### time_ultra（2 组可比较）
动态复阻抗：最大 0.1606%，均值 0.1574%，P95 0.1602%。
铜温度：最大 0.4003K，均值 0.2712K，P95 0.3874K。
封装温度：最大 0.0479K，均值 0.0342K，P95 0.0465K。

### time_superfine（2 组可比较）
动态复阻抗：最大 0.1586%，均值 0.1569%，P95 0.1584%。
铜温度：最大 0.4011K，均值 0.2717K，P95 0.3882K。
封装温度：最大 0.0479K，均值 0.0342K，P95 0.0466K。

### space_h2（3 组可比较）
动态复阻抗：最大 1.4887%，均值 1.1197%，P95 1.4431%。
铜温度：最大 0.6233K，均值 0.3369K，P95 0.5941K。
封装温度：最大 0.8916K，均值 0.4883K，P95 0.8511K。

### space_h1（2 组可比较）
动态复阻抗：最大 1.4884%，均值 1.2601%，P95 1.4655%。
铜温度：最大 0.6224K，均值 0.3392K，P95 0.5941K。
封装温度：最大 0.8908K，均值 0.4890K，P95 0.8507K。

## 物理有效域、功率与热能守恒

每一组合均需满足海水 271.15–373.15 K、铜低于 1357.77 K；端口电功率相对误差和海水积分焦耳损耗相对误差均不超过 1e-10，热节点及全局热预算不超过 1e-4 W，海水热容量守恒。

## 几何修订与可复现性

原冻结 U09 的 alpha2=0.9851324743684382、core2_mm=25 未通过原 BFZI 几何有效性检查，保留原始失败记录。修订 U09 仅改变 alpha2=0.86、core2_mm=20；其余样本与功率、热参数均保持冻结。

在仓库根目录运行（BFZI 依赖目录为相邻 BFZI）：

    python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U01 U02 U03 U04 U05 U06 U07 U08 U09 U10 U11 U12 --dt 15 --grids ultra superfine --inner-step 2 --source-order 5 --run-tag s5 --out runs/bfzi_unseen_all_s5_h2.json
    python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U01 U02 U03 U04 U05 U06 U07 U08 U09 U10 U11 U12 --dt 15 --grids ultra superfine --inner-step 1 --source-order 5 --run-tag s5 --out runs/bfzi_unseen_all_s5_h1.json
    python -m examples.bfzi_unseen_accept --run-tag s5 --out runs\bfzi_physical_unseen12_s5.json --report docs\research\bfzi_uwpt_electrothermal\evidence\physical_unseen12_s5.md

源数据保存在 runs/bfzi_phys_U*_s5.npz；所有样本的单组合功率、温度及热残差在 runs\bfzi_physical_unseen12_s5.json 中。

## 守恒指标（所有已完成组合的最大绝对值）

max_rel_port_power: 2.809063e-15
max_rel_q_power: 8.318804e-16
peak_thermal_residual_W: 1.022540e-05
peak_global_budget_W: 4.240395e-06
skin_water_capacity_error_J_K: 5.820766e-11

## 动态阻抗与温度峰值时刻

| 样本 | 最大动态 R (Ω) | 最大动态 X (Ω) | 铜峰时刻 (s) | 封装峰时刻 (s) |
| --- | ---: | ---: | ---: | ---: |
| U01 | 0.0149281 | 0.00879666 | 390 | 405 |
| U02 | 待运行 | 待运行 | 待运行 | 待运行 |
| U03 | 待运行 | 待运行 | 待运行 | 待运行 |
| U04 | 0.0391297 | 0.0223835 | 405 | 405 |
| U05 | 待运行 | 待运行 | 待运行 | 待运行 |
| U06 | 待运行 | 待运行 | 待运行 | 待运行 |
| U07 | 待运行 | 待运行 | 待运行 | 待运行 |
| U08 | 待运行 | 待运行 | 待运行 | 待运行 |
| U09 | 待运行 | 待运行 | 待运行 | 待运行 |
| U10 | 0.0272133 | 0.0149131 | 405 | 405 |
| U11 | 待运行 | 待运行 | 待运行 | 待运行 |
| U12 | 待运行 | 待运行 | 待运行 | 待运行 |

## 失败项

