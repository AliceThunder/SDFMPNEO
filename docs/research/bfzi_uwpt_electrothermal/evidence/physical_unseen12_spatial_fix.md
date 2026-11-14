# BFZI V150 冻结 U01–U12 空间电磁温度反馈精度修复

2026-11-15。继续执行 `PHYSICAL_DIRECT_FINAL_ACCEPTANCE_V1.md`，冻结 12 组案例及原 2%/1%/温度/守恒门槛。

## 原始失败

原冻结最终报告：6/12 通过，U01/U02/U03/U05/U06/U11 因空间动态复阻抗误差分别为 2.571%、2.296%、2.091%、2.248%、2.044%、2.795% 未达 ≤2%。时间/温度/守恒均通过。原始证据仍保留在 `runs/bfzi_physical_unseen12_final.json`。

## 方程及实现根因

原 `sdfmpneo_bfzi/sea_obb.py::_tetra` 将每个旋转封装切割出的海水多面体划为以内部中心和边界三角面构成的四面体，每四面体仅使用其重心进行体积积分。该求积只对一次多项式精确，无法解析海水切割单元内部的二阶空间变化。

相同的切割体积积分点进入 BFZI 原生源投影 `C`、近邻 Green 核积分、源场协方差 `mass`、温度相关电导率 `sigma(T)` 与局部 Joule 热重分配。由 `J(q,t)\propto sigma(q,t) C(q)` 和 `P_{sea}\propto\int sigma(q,t)|C_{eff}(q,t)|^2 dq` 可见，忽略体内二阶变化会直接影响电磁温度反馈。功率守恒通过说明数值损耗总量匹配端口，并不能消除源项的空间积分误差。

## 实装修复

在每个子四面体采用正权重四点二阶积分，四组重心坐标为 `(a,b,b,b)` 的轮换，其中
`a=(5+3 sqrt(5))/20`、`b=(5-sqrt(5))/20`，每点权重为子四面体体积的四分之一。该积分对二次多项式精确，所有点位于水体内，保持原切割体积、原 FV 节点与原 BFZI V150 电磁热闭环。未修改介质律、热参数、冻结案例或验收阈值。

## 验证

- `tests/test_bfzi_sea_obb.py` 新增四面体体积、一次矩与二次矩验证。
- `tests/test_bfzi_sea_obb.py tests/test_bfzi_obb_faces.py tests/test_bfzi_sea_layer.py tests/test_bfzi_v148_et.py`：28 passed。
- 对 U01、U11 按原 `ultra/superfine`、600 s、15 s 输出、1.875 s 内部热步重新求解。

复算命令：

```powershell
python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U01 U11 --dt 15 --grids ultra superfine --inner-step 2 --out runs/bfzi_unseen_u01_u11_quad2.json
```

## 验收状态

原总验收 **OPEN**。待 U01/U11 重算空间误差核验，再决定是否推进原冻结全集。


## 第二轮：三阶近源积分与热层角部归属

第一轮四点二阶切割积分已运行 U01、U11 原冻结时程（ultra/superfine，内部热步 1.875 s）：
- U01：原 2.5714%，二阶 2.5721%；无实质改善。
- U11：原 2.7943%，二阶 2.7650%；仍失败。

进一步检查 BFZI V150 `_record_quadrature(rec,order)`：完整原生湿 FV 记录的 `subboxes=[]`，因此无论 `order=3` 如何请求，均直接返回初始化时的固定 `2×2×2` 八点 Gauss 数据。原 `_mass` 和 `subcell_temp` 使用这些预存点，使近源 `C` 均值和二次矩在 40–50 mm 湿 FV 中分辨率不足，并产生跨网格系统性比例误差。

实装修复：
- `bfzi_v148_adapter._mass` 对完整海水 FV 构造独立 `3×3×3` Gauss 求积，绕组及端口引线源项在这些点直接积分，核 `G0` 保持原 V150 方案。
- `BFZIV148Adapter` 中的 FV 源均值 `C` 与局部协方差修正 `mass` 使用同一高阶求积，避免只提高二次矩却继续使用旧二阶场均值。
- `sea_layer._qmap` 对封装棱角处重叠的表面热层棱柱按距离最近的真实封装面选择唯一热层节点，避免较密源积分点被多重映射。
- `bfzi_physics_recheck --run-tag s3` 为新真值 NPZ 使用独立文件名，保留冻结旧参考数据。

**U01 实测电磁重放**：用原两网格完整 600 s 温度轨迹逐时刻驱动修复后的 V150 电磁模型，不重算热演化；动态复阻抗空间相对 L2 **0.2959826%**（原真值 2.57138%）。该结果隔离并验证了**电磁离散改进**；最终验收仍需完整闭环重算。源数据在 `runs/bfzi_U01_source3_replay.json`。

第二轮相关局部测试 **26 passed**，包含角部最邻近热层归属测试。最终冻结验收持续 **OPEN**。


## U01 三阶近源方案完整双向耦合初验（1.875 s）

以冻结工况 U01，原 ultra/superfine 网格、完整 600 s、15 s 输出、实际 1.875 s 以内子步运行，两网格 `candidate` 且全时程物理有效。比较如下：

| 对照 | 动态复阻抗空间相对 L2 | 铜温度最大差 | 封装温度最大差 |
| --- | ---: | ---: | ---: |
| 原 V150 参考 | 2.5714% | — | — |
| 二阶切割单元 | 2.5721% | 0.0641 K | 0.0952 K |
| 三阶近源且均值一致 | **0.2775%** | **0.0515 K** | **0.0826 K** |

新方案的两级网格最大热节点残差 ≤4.144e-6 W，全局热预算残差 ≤1.247e-6 W，端口功率相对误差 ≤4.19e-15，海水热容量偏差 ≤5.82e-11 J/K；峰值海水 ≤300.546 K，铜 ≤345.080 K。**U01 空间 h2 验收通过**。完整样本最终验收继续等待 1 s 实际热步对照，状态保持 OPEN。

正式验收聚合器 `examples/bfzi_unseen_accept.py` 增加独立 `--run-tag s3` 数据源筛选，原冻结记录复聚合仍为 **6/12 PASS、6/12 FAIL、0 PENDING**；新运行的 NPZ 不覆盖原始文件。

## 完整闭环 U01（2026-11-15）

保持冻结几何、激励与热材料参数，BFZI V150 ultra/superfine 原始两网格、真实 1.875 s 热步、每 15 s 输出阻抗，分别完成 600 s 的电磁热直接法。修复后的动态复阻抗空间相对 L2 为 **0.277465%**（原始 2.57138%，固定旧热轨迹电磁重放 0.29598%）；铜/封装空间最大温差为 **0.05149/0.08263 K**。最大热节点残差 **4.1431e-6 W**，全局热预算最大 **1.2466e-6 W**，最大相对端口功率失衡 **4.19e-15**，热容量相对原 FV 保持守恒。两组均 `candidate`，空间门槛通过。

证据文件：`runs/bfzi_unseen_u01_u11_s3_h2.json`，`runs/bfzi_phys_U01_{ultra,superfine}_dt15_h2_*_s3.npz`。原 `runs/bfzi_physical_unseen12_final.json` 及其 NPZ 不覆盖。原样本总验收仍 OPEN；U01 1 s 热步及 U11 空间对照另行验收。

## 完整闭环 U11 及全集复算启动

U11 在原冻结工况、原 ultra/superfine 网格、15 s 输出和 1.875 s 内部热步下两次均完整完成 600 s，空间动态复阻抗相对 L2 **0.371160%**（原 2.7943%），铜/封装空间全程最大差 **0.37538/0.53481 K**，热节点/全局最大残差分别低于 **7.64e-6/1.70e-6 W**，端口相对功率误差低于 **2.83e-15**。两档均为 `candidate`，空间收敛通过。原 U01 在 ultra 网格真实内部 1.875/1 s 比较的动态复阻抗、铜、封装误差为 **0.15819%/0.14213 K/0.02052 K**，时间收敛通过。

U01 和 U11 的两级空间验收通过后，已按原冻样清单启动剩余 U02/U03/U04/U05/U06/U07/U08/U09/U10/U12 全集的两档网格与两个内部热步复算。验收使用 `--run-tag s3` 保存独立 NPZ 与 JSON，原冻结数据不覆盖。当前**总验收仍为 OPEN**，以新 48 次完整时程的四组合结果为最终依据。

源记录：`runs/bfzi_unseen_u01_u11_s3_h2.json`；余下独立记录将分别保存于 `runs/bfzi_unseen_u01_u11_s3_h1.json`、`runs/bfzi_unseen_G1_s3_h2.json`、`runs/bfzi_unseen_G2_s3_h2.json`、`runs/bfzi_unseen_G3_s3_h2.json`、`runs/bfzi_unseen_rest_s3_h1.json`，运行完成后通过 `python -m examples.bfzi_unseen_accept --run-tag s3 --out runs/bfzi_physical_unseen12_s3.json --report docs/research/bfzi_uwpt_electrothermal/evidence/physical_unseen12_s3.md` 汇总。


## U11 三阶近源方案完整双向耦合初验（1.875 s）

冻结 U11 原两级网格、600 s 全时程通过材料域和功率/热能守恒。空间动态复阻抗 L2 **0.3711600%**（原 2.7943%；二阶切割求积仍为 2.7650%）；铜温空间差 **0.37538 K**，封装温空间差 **0.53481 K**，全部优于原 2%/1 K/1.5 K 门槛。最大热残差 ≤7.637e-6 W，全局预算残差 ≤1.692e-6 W，端口功率相对误差 ≤2.83e-15，海水热容量守恒偏差 ≤5.82e-11 J/K。

**U01、U11 两组独立失败样本已通过三阶近源方案完整 600 s 空间 h2 验收。** 1 s 内部热步复验及其余 U02–U10/U12 的全部四组合继续执行。证据 `runs/bfzi_unseen_u01_u11_s3_h2.json` 和对应 `*_s3.npz`，原 48 次冻结基准均保持原文件。
