# BFZI V150 U01–U12 独立物理直接法四组合验收

当前结果：通过 0/12，失败 0/12，待完成 12/12；总验收 OPEN。

对每个工况进行 ultra/superfine 两档原生湿 FV × 1.875/1 s 真实热步，统一 15 s 阻抗采样，完整 600 s。

## 全部样本

| 样本 | 状态 | 时间 ultra | 时间 superfine | 空间 1.875 s | 空间 1 s | 铜/封装/海水最高温 (K) |
| --- | --- | --- | --- | --- | --- | --- |
| U01 | pending | 0.163% / 0.142 / 0.021 | 待运行 | 待运行 | 待运行 | 345.16/322.71/300.58 |
| U02 | pending | 0.028% / 0.038 / 0.039 | 待运行 | 待运行 | 待运行 | 368.22/346.83/307.05 |
| U03 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 334.78/325.13/304.27 |
| U04 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 435.98/373.10/322.98 |
| U05 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 330.84/317.45/304.18 |
| U06 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 309.11/305.43/298.87 |
| U07 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 403.56/354.53/309.42 |
| U08 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 408.66/374.91/315.21 |
| U09 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 308.03/304.40/297.46 |
| U10 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 384.79/345.32/311.72 |
| U11 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 367.22/341.79/313.12 |
| U12 | pending | 待运行 | 待运行 | 待运行 | 待运行 | 345.91/333.79/311.10 |

表中四类误差分别为动态复阻抗相对 L2、铜最高逐时刻绝对差 K、封装最高逐时刻绝对差 K。
时间门槛：1%、0.5 K、0.5 K。空间门槛：2%、1 K、1.5 K。

## 已有四组合误差统计

### time_ultra（2 组可比较）
动态复阻抗：最大 0.1629%，均值 0.0956%，P95 0.1562%。
铜温度：最大 0.1423K，均值 0.0899K，P95 0.1371K。
封装温度：最大 0.0391K，均值 0.0298K，P95 0.0382K。

### time_superfine（0 组可比较）

### space_h2（0 组可比较）

### space_h1（0 组可比较）

## 物理有效域、功率与热能守恒

每一组合均需满足海水 271.15–373.15 K、铜低于 1357.77 K；端口电功率相对误差和海水积分焦耳损耗相对误差均不超过 1e-10，热节点及全局热预算不超过 1e-4 W，海水热容量守恒。

## 几何修订与可复现性

原冻结 U09 的 alpha2=0.9851324743684382、core2_mm=25 未通过原 BFZI 几何有效性检查，保留原始失败记录。修订 U09 仅改变 alpha2=0.86、core2_mm=20；其余样本与功率、热参数均保持冻结。

在 D:/work/bfzi-et-v2.1/SDFMPNEO 中运行（BFZI 依赖目录为相邻 BFZI）：

    python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U01 U02 U03 U04 U05 U06 U07 U08 U09 U10 U11 U12 --dt 15 --grids ultra superfine --inner-step 2 --out runs/bfzi_unseen_all_h2.json
    python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U01 U02 U03 U04 U05 U06 U07 U08 U09 U10 U11 U12 --dt 15 --grids ultra superfine --inner-step 1 --out runs/bfzi_unseen_all_h1.json
    python -m examples.bfzi_unseen_accept

源数据保存在 runs/bfzi_phys_U*_*.npz；所有样本的单组合功率、温度及热残差在 runs/bfzi_physical_unseen12_final.json 中。
