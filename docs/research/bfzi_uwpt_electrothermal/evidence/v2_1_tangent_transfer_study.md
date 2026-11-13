# V2.1：跨几何海水物理切线残差与 8/16 次方向求解同条件验收

日期：2026-11-12。研究分支：`research/bfzi-uwpt-electrothermal-surrogate`。

## 本轮交付

已新增：

- `sdfmpneo_bfzi/geom_tangent_transfer.py`：参考几何的真实三维海水热模态作为统一坐标，使用体积加权空间模态重叠及正交 Procrustes 配准每个 BFZI 几何的八维热模态，按相同 2×2 互易复端口构建48维 Jacobian 标签。在40维 BFZI 冷态复阻抗、线圈和 SE(3)/双封装几何上，通过邻域物理切线插值和小型神经网络残差迁移至新几何。神经校正权重仅使用训练几何的 leave-one-geometry-out 切线验证选择，`P03/P07` 未用于监督或模型选择。
- `examples/bfzi_tangent_study.py`：同一原 BFZI V150 几何、同一冷态电磁锚、同一8个热模态、同一温度扰动幅度下，分别计时 16 次双侧和 8 次单侧原始BFZI方向求解。对 P03/P07 使用完全留出的真实三维热场和原复 Z 评价海水单独热增量、完整端口热变化；再将各切线接入已训练的40维分数阶神经网络和五材料保守32阶热ROM，运行三维自主闭环，评价实际动态复 Z、海水/双封装/铜温度误差、海水 q 功率及五材料收支。P07 同时覆盖已有150V/15s双启停真值。
- `tests/test_bfzi_tangent_transfer.py`：热模态不同排列及符号配准、40维条件残差训练与保存/加载/零方向EM求解的合成验证。沙箱局部验证：`2 passed`。

## 物理一致性

原有 `SeaTangent` 使用原 BFZI 的物理热 ROM 前8海水空间模态，基于两端口复阻抗方向导数 `jac_j`。不同几何的SVD/POD模态存在符号翻转和子空间内旋转。对每个新几何，在原物理FV坐标下构造重叠矩阵 `C=B_target^T W_ref B_ref`，由 `C=U S V^T` 得到正交变换 `Q=U V^T`，使 `B_target Q` 在参考坐标中尽量对齐。对应的方向导数在训练标签中同步按 `Q^T jac_target` 变换，使一阶热阻抗泛函不变：

```text
Delta Zsea(theta) = sum_j <B_j,theta>_volume * jac_j
Aligned modes = B_native @ Q
Aligned jac   = Q.T @ jac_native
```

跨几何物理基插值在原 BFZI 海水 FV 中心进行。训练用 P01/P02/P04/P05/P06 原 BFZI 8模态双侧16次教师；新的 P03/P07 在预测阶段只需原冷态锚及32阶热ROM，不需要原 BFZI 的温度扰动 EM 方向求解。原三维 J/q、五材料导热和铜精确 Bessel 趋肤模型全部维持原样。预测后的切线通过已有 `SeaTangent` 对象接入完整动态闭环。

## 真实BFZI实验执行

在具备独立 BFZI V150 `bfzi-v150-turn-scan` 物理环境、原项目已有运行数据的条件下执行：

```powershell
python -m examples.bfzi_tangent_study --bfzi-root D:/work/bfzi-et-v2.1/BFZI --runs runs --amplitude .2 --out runs/bfzi_tangent_study.json
```

输入仍为已经归档的七个几何冷态锚、原P03/P07真实动态热场和复端口、原P03/P07 ROM、40维分数阶神经权重。程序通过各几何已存的原生线圈位置、圆角参数、yaw/pitch/roll及独立双封装尺寸重建 BFZI V150 算子，并检查 FV 网格和冷态海水复 Z 与档案一致，之后才进行方向求解。

每几何 **单侧8次 vs 双侧16次**原始电磁方向调用；训练集五几何双侧教师 **80次方向调用**，独立留出两几何的实验比较额外 **48次方向调用**，总计128次方向调用（不包括各几何原始海水Green/冷态锚、ROM建设或验证冷态求解），物理耗时以生成JSON中实际计时为准。文件 `runs/bfzi_pose_pXX_fd_central.npz`、`runs/bfzi_pose_pXX_fd_forward.npz`、`runs/bfzi_tangent_transfer.npz`、`runs/bfzi_pose_pXX_neural_tangent.npz` 可供后续使用。自动汇总 `offline_sec`、`training_sec`、`oracle_true_T`、`autonomous_3D` 等字段。

实验指标：
1. 原BFZI真温度上的单独海水 `Delta Zsea` 与完整动态 `Delta Ztotal` 相对 L2，比较两种差分与零方向神经预测。
2. 使用相同原分数阶网络、同一保守ROM自主闭环的真实复 `Z(t)`、海水/封装/铜三维温度误差、海水 q 端口积分误差和五材料能量预算。
3. 每几何冷态物理构建耗时、8/16次电磁方向求解耗时、神经切线训练耗时，以及每条动态历史的独立推理耗时；不混淆离线成本与原 BFZI 动态加速比。

## 当前验证边界与后续验收

本轮沙箱可进行局部 Python 合成测试，沙箱未获得独立 BFZI V150 环境及其存档二进制数据，因此尚未产生新一轮原始 BFZI 的 P03/P07 单侧/双侧精度或离线计时。原先已经实测的 P03/P07 双侧16次高精度结果继续作为物理基线，不以合成测试或理论方向求解次数替代真实结果。

下一次原 BFZI 环境实算后，直接依据 `runs/bfzi_tangent_study.json` 判断：若单侧满足完整端口及独立海水热变化精度目标，则可以用8次替代16次；神经零方向方案若整几何外推和三维闭环满足同一精度标准，再评估将其用于新几何的高精度默认方案。


## 补充：无需新的 BFZI 方向求解，直接利用既有真实时程

新增 `sdfmpneo_bfzi/geom_tangent_data.py` 与 `examples/bfzi_tangent_data_study.py`。每个训练几何已有原 BFZI V150 三维温度、完整复2×2端口及多电压/启停真实时程，通过原冷态锚的铜趋肤函数直接隔离海水热增量：

```text
Delta Zsea(t) = Ztotal,BFZI(t) - Ztotal,cold
               - diag[Zcu(Tcu(t))-Zcu(Tamb)]
beta_j(t) = sum_i V_i B_j(x_i)[Tsea(x_i,t)-Tamb]
Delta Zsea(t) ~ sum_j beta_j(t) J_j
```

对原训练几何 P01/P02/P04/P05/P06，以物理冷态ROM海水基投影为设计矩阵，采用分列归一化多输出岭回归，最后一条训练几何时程留出选择正则强度。记录8个模式的实际有效秩、训练海水复阻抗相对误差及内部留出历史误差。对于数据未激发的物理模态，算法将其方向导数正则化至零，并报告有效秩，避免把不可辨识方向称作真实 BFZI 电磁方向求解结果。

之后用同一物理热基 Procrustes 对齐及40维小网络学习五几何间的切线残差。P03/P07 仍完全独立，模型推理时保持原冷态物理锚、原三维J/q、精确铜趋肤和32阶五材料能量守恒热ROM。每个新几何的切线预测无需再调用原 BFZI 温度扰动电磁求解器。

此路径能直接使用仓库已经保存的 `.npz` 实验数据及神经权重；**不需要独立 BFZI V150 安装或 XEnergy**：

```powershell
python -m examples.bfzi_tangent_data_study --runs runs --out runs/bfzi_tangent_data_study.json
```

输出 `runs/bfzi_pose_p0{1,2,4,5,6}_data_tangent.npz`、`runs/bfzi_tangent_transfer_data.npz`、P03/P07物理切线预测及 `runs/bfzi_tangent_data_study.json`。JSON 同时给出P03/P07原已归档16次双侧切线与零方向物理历史教师/神经残差的**真实温度电磁评价**及**40维分数阶网络+32阶热ROM的自主三维闭环评价**，包括P07已有150V/15s时程、实际推理时间、完整动态Z/温度误差、q端口闭合和五材料能量收支。

数值边界：历史回归的切线属于**已激发温度子空间上的物理时程拟合**；与直接对每个热模态独立扰动求解的原 BFZI 16次方向Jacobian，分别报告和比较，不预设两者精度相同。新增合成验证 `tests/test_bfzi_tangent_data.py` 覆盖精确线性海水阻抗恢复、铜温变扣除、原几何与时程一致性。沙箱模拟物理小样本测试与上轮模态对齐测试联合 **4 passed**（依赖轻量替身环境，非完整BFZI集成验证）。

当前新方法的真实P03/P07精度与速度仍需上述完整历史数据运行，原双侧16次已实测的0.0019%～0.0027%为对照基线。
