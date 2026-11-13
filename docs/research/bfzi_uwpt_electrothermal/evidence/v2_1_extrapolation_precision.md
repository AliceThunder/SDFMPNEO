# V2.1：BFZI 神经代理跨几何外推与高精度动态复阻抗实算

实测日期：2026-11-13。分支 `research/bfzi-uwpt-electrothermal-surrogate`。所有真值均由独立原 `AliceThunder/BFZI` V150 `bfzi-v150-turn-scan` 直接计算；与三维守恒热 ROM、分数阶网络的真实留出对比在 XEnergy 执行，不使用 GitHub Actions。

## 研究目标和已确认的技术选择

主线仍为原 BFZI 的圆形—圆角方形双线圈及原生 SE(3) 相对位姿、双独立三维封装、多温度场、铜温度与海水复阻抗动态反作用，允许关断与再启动。优先提高独立新几何的外推精度及完整三维电磁热动态推理精度，而不牺牲原三维空间 J/q 或材料能量守恒。

本轮的实验分清 **纯神经复阻抗、闭环神经训练、扩大原 BFZI 几何教师、真实空间热状态的冷态 BFZI 复阻抗灵敏度** 四个来源。最终精度最高的路径为：**分数阶神经历史生成动态海水阻抗初值 → 原冷态电磁物理灵敏度按神经—ROM 预测的三维热状态校正动态海水复阻抗 → 原 BFZI 精确铜趋肤阻抗和空气电感组成完整复端口 → 原 3D FV 或五材料守恒32阶热 ROM 推进海水/两封装/两铜温度**。新几何首先计算冷态场、热算子和8模态阻抗灵敏度；随后任意动态时步不调用原 BFZI Green/GMRES。

## 1. 纯神经模型的闭环自训练和整几何留出

新增 `sdfmpneo_bfzi/geom_train.py`，实现与推理一致的微分闭环电磁热源。每时间步根据神经预测的海水增量 `Delta Zsea` 和铜温度，采用原 BFZI 冷态空气 L/铜 Bessel 趋肤阻抗对温度的局部 Jacobian 构建 `Ztotal`，重新求解两端口电流、海水功率和双铜加热率，再进入下一分数阶 SOE+指数记忆时步。训练逐渐从真值源推进到完全自主源，训练后真实推理仍使用原 BFZI 完整铜 Bessel 阻抗。损失为完整复 Z 的物理 Jacobian 误差、海水与封装平均/热点、铜温度，并通过几何均衡权重防止样本较多的形状主导。

正式训练入口 `examples/bfzi_geom_frac.py` 新增 `--closed-loss`、`--balanced`、`--geo-val`、`--closed-warmup` 和 `--closed-ramp`。 `--geo-val P02` 从训练集排除 **P02 整个几何**，而 P03 仍完全不用于训练或模型选择。实验显示在只学习 P01 作为新姿态教师的情况下，P02 整几何验证选择最早 epoch，P03 动态 Z 误差 **5.740%**：实际几何覆盖不足，不能依靠优化器对现有少量几何进行有效外推。此模型完整归档，未用作主结果。

从现有40维姿态模型微调的闭环模型则在 P03 完全留出的**标量自回归热摘要**中，使动态 Z 平均相对误差 **4.628%→4.039%**、铜最大温差平均 **0.925→0.828 K**，M4/M1/M2 留出误差也略降。进一步用同一 P03 原 BFZI **完整三维热 ROM**测试，其动态 Z 分别由此前 **1.609%/2.097% 变成 2.354%/2.707%**，标量提高未带来三维反馈精度提升；简单的旧/新模型等权平均得到 **1.963%/2.390%**。因此保留 `runs/bfzi_geom_closed.pt` 及 `runs/bfzi_geom_geo.pt`、完整误差记录作为消融，**没有把其替代为高精度三维主模型**。

## 2. 加大真实原 BFZI 几何教师覆盖：纯神经外推提升

新增原 BFZI V150 的三组几何 P04–P06 进行训练，真实数据包含原生双线圈形状、分别的封装尺寸、三维偏移、yaw/pitch/roll 与600s启停历史。新增一个完全留出的 P07 作为更大角度外推测试。

| 原 BFZI 样本 | 核心形状、姿态 | 作用 |
|---|---|---|
| P01 | 圆形+圆角，法向约 2°，210/226mm封装 | 训练 |
| P02 | 双圆角，法向约 10°，214/234mm封装 | 训练 |
| P04 | 圆形+圆角，原生roll 8°，偏移(10,0,-5)mm，208/226mm封装 | 新增训练 |
| P05 | 双圆角，原生roll 12°，偏移(-9,0,7)mm，216/232mm封装 | 新增训练 |
| P06 | 双圆角，原生roll 16°，偏移(4,0,-9)mm，220/230mm封装 | 新增训练 |
| P03 | 圆形+圆角，实际法向19.990°，偏移(8,0,5)mm，210/232mm封装 | 完整新几何留出 |
| P07 | 圆形+圆角，输入roll 27° + 额外yaw7°/pitch9°，偏移(13,0,9)mm，216/240mm封装 | **大角度外推且完全新几何留出** |

原 M1/M2/M4 继续作为圆角方形/圆形参考几何。全部保留原 BFZI 原生线圈、空气互感、海水 Green/源与三维 FV 热算子。P04–P06 各 6 条真实 600s 程序（5条可训练，1条独立启停留出）；联合训练实际 **51 训练/18 验证**历史，几何损失均衡，最佳 epoch 907。训练未使用 P03、P07 任何动态真值。

| 完整新几何留出的纯神经动态复 Z `Delta Z` 误差 | 原40维网络 | 新增真实几何训练 |
|---|---:|---:|
| P03 约20° | 4.628% | **3.468%** |
| P07 约27° | 5.968% | **3.870%** |

P07 的双封装热点温差平均 **2.877→2.803 K**，铜温度平均最大差 **1.237→0.711 K**。原M4/M1/M2动态Z留出误差在新模型分别为 **3.219%/3.883%/4.167%**。新模型为 `runs/bfzi_geom_diverse.pt`，报告 `runs/bfzi_geom_diverse_metrics.json`，P07三种网络独立对照 `runs/bfzi_geom_diverse_p07_compare.json`。几何教师文件和所有冷态锚已入库。

## 3. 原 BFZI 冷态八模态电磁热灵敏度：提升真正的外推精度

新增 `sdfmpneo_bfzi/geom_tangent.py`。只对每个新几何做**一次离线计算**：在原 BFZI 冷态3D热 FV 算子上，用海水弱源、两铜热源生成的物理热 ROM，抽取海水体积正交的前8个空间温度模态 `B_j(x)`；对每个模式使用原 BFZI V150 的 `em.solve(Tamb +/- delta_j*B_j)` 两次，计算 **原双端口复海水阻抗方向导数**：

```text
D_j Zsea = [ Zsea(Tamb + delta_j B_j) - Zsea(Tamb - delta_j B_j) ]/(2 delta_j)
beta_j(t) = sum_i [ Vi B_j(xi) (Tsea(xi,t)-Tamb) ]
Delta Zsea,phys(t) = sum_{j=1}^8 beta_j(t) D_j Zsea
Ztotal(t) = Zsea,cold + Delta Zsea,phys(t)
          + j omega Lair + diag[ZCu,original_BFZI(TCu(t))]
```

方向温度扰动每个模式约1K峰值；存档 `runs/bfzi_pose_p0X_tangent_anchor_sea_tangent.npz`，后续每步仅做小矩阵投影与双端口方程。各向同性空间热源仍以相同海水端口功率归一化，热 FV 保留原五材料能量预算。新几何离线阶段需要额外约 **16 次原 BFZI 电磁方向求解**；推理阶段无需再做这些求解。

**先隔离电磁线性化误差**：使用 P03/P07 原 BFZI 动态直接法的真实三维温度、真实铜温度，将其放入上述冷态线性化（此处真实温度仅用于评价，不参与8模态计算）。两条40V/115V时程的完整动态复 Z 误差：

| 原 BFZI 真实热场上的电磁灵敏度 | 40V | 115V |
|---|---:|---:|
| P03 完整动态复 Z 相对误差 | 0.00183% | 0.00228% |
| P07 完整动态复 Z 相对误差 | 0.00218% | 0.00273% |

对单独 **海水 `Delta Zsea`** 的相对误差，P03 为0.543%/0.575%、P07为1.268%/1.342%；完整复 Z 的绝对热变化同时含原 BFZI 精确铜趋肤阻抗，因此应将这两类指标区别理解。这也是本阶段只用8个空间热模态便可显著提高完整动态复 Z 精度的物理基础。数值见 `runs/bfzi_pose_p03_tangent_metrics.json` 与 `runs/bfzi_pose_p07_tangent_metrics.json`。

## 4. 真正自主三维热场+神经网络+灵敏度的完整闭环

进一步修改 `sdfmpneo_bfzi/geom_heat.py`、`geom_rom.py`、`examples/bfzi_geom_heat.py`，新增 `--sea-tangent` 和 `--tangent-weight`。每个时间步先用神经网络动态海水阻抗预测，执行一次保守三维热预测，**在预测的新三维温度场上评价上节原 BFZI 冷态海水灵敏度**，重组复阻抗，再执行保守的三维热校正步。所有原 BFZI 空间热传导、铜趋肤、端口功率热源和材料热收支保留。对于32阶 ROM，热预测步不记录重复材料收支，输出一个时间步只产生一组完整的五材料预算。

固定 `runs/bfzi_geom_diverse.pt` 的分数阶40维权重，用 P03/P07 原BFZI完全未见的直接法时程对比：

| 自主完整三维推理，八模态灵敏度权重1.0 | 40V | 115V |
|---|---:|---:|
| P03 动态复 Z 相对误差 | **0.001830%** | **0.002281%** |
| P07 动态复 Z 相对误差 | **0.002177%** | **0.002712%** |
| P03 海水三维最大温差 | 1.204e-5 K | 8.477e-5 K |
| P03 封装三维最大温差 | 9.629e-6 K | 7.688e-5 K |
| P07 海水三维最大温差 | 6.920e-5 K | 4.854e-4 K |
| P07 封装三维最大温差 | 7.726e-6 K | 6.139e-5 K |
| P07 铜温度最大差 | 3.260e-8 K | 1.102e-7 K |
| P07 五材料热预算最大差 | 3.446e-13 W | 4.950e-11 W |
| P07 时程神经+32阶热 ROM 推理耗时 | 0.0575s | 0.0615s |

`--tangent-weight .9` 保留10%的纯神经海水 Z 增量预测，P07相同40/115V轨迹完整动态 Z 误差为 **0.2331%/0.2374%**，其余材料功率收支仍闭合。各权重都独立实跑并存档；采用 `--tangent-weight 1` 作为**当前最高精度模式**，保留 `0` 作为纯神经快速基准，`.9` 作为物理—神经混合权重对照。

文件：`runs/bfzi_pose_p03_tangent_heat.json/.npz`、`runs/bfzi_pose_p07_tangent_heat.json/.npz`、`runs/bfzi_pose_p07_tangent_hybrid.json/.npz`。

## 5. 强电压和细时间网格的双重外推

为了增强真实升温及热反作用，新增 P07 完整未见几何的 **150V、15s时间步、600s含通断/重启**原 BFZI V150 直接法轨迹，原独立时序 ID16/ID18。训练仍只用30s原BFZI教师，且不含P07。

直接法实际最高升温：
- 海水：ID16/18 为 **0.19837/0.26078 K**。
- 两封装：**20.6194/22.8624 K**。
- 双铜：**61.7263/62.7645 K**。

原 BFZI 150V/15s/41时刻两条动态直接法合计耗时 **10.2245s**；8模态高精度代理两条时程合计 **0.2828s**，同条件动态约 **36.2倍加速**（不计每几何离线冷态建模、8模态灵敏度构建及网络训练）。

| P07 150V、15s、600s全三维，独立留出 | 程序ID16 | 程序ID18 |
|---|---:|---:|
| 纯神经动态复 Z 相对误差 | 2.5517% | 2.5130% |
| **完整灵敏度物理校正后动态复 Z 相对误差** | **0.001903%** | **0.002580%** |
| 预测海水最大空间温差 | 0.001054 K | 0.001181 K |
| 预测两封装最大空间温差 | 0.0000944 K | 0.0002024 K |
| 预测双铜最大温差 | 4.77e-7 K | 6.79e-7 K |
| 32阶 ROM 五材料热预算最大差 | 1.263e-10 W | 1.263e-10 W |
| 代理动态时程 | 0.1416s | 0.1412s |

两条原 BFZI 150V 15s 真实直接法轨迹、完整 T/J/q/Z 预测及误差分别保存在 `runs/bfzi_pose_p07_15s_holdout.npz`、`runs/bfzi_pose_p07_15s_tangent_heat.json/.npz`、`runs/bfzi_pose_p07_15s_neural_heat.json/.npz`。原直接法耗时保存在 `runs/bfzi_pose_p07_15s_anchor.json`。

## 6. 最短复现入口

现成 P07 冷态锚及原 BFZI 8模态灵敏度文件已入库，执行动态推理无需原 BFZI 安装：

```powershell
# 全新电压启停：150V/15s，40D分数阶记忆+8D原BFZI物理灵敏度+32D守恒热ROM
python -m examples.bfzi_geom_heat --standalone --voltage 150 --heat 210 --cool 210 --step 15 --horizon 600 --anchor runs/bfzi_pose_p07_anchor.npz --model runs/bfzi_geom_diverse.pt --rom-rank 32 --rom-file runs/bfzi_pose_p07_rom.npz --sea-tangent runs/bfzi_pose_p07_tangent_anchor_sea_tangent.npz --tangent-weight 1 --out runs/bfzi_pose_p07_user.json

# 同原BFZI真实150V 15s轨迹逐点比较温度和动态复Z
python -m examples.bfzi_geom_heat --anchor runs/bfzi_pose_p07_anchor.npz --teacher runs/bfzi_pose_p07_15s_holdout.npz --model runs/bfzi_geom_diverse.pt --rom-rank 32 --rom-file runs/bfzi_pose_p07_rom.npz --sea-tangent runs/bfzi_pose_p07_tangent_anchor_sea_tangent.npz --tangent-weight 1 --compare-full --out runs/bfzi_pose_p07_15s_tangent_heat.json
```

对另一种新几何，可使用 `python -m examples.bfzi_param` 直接调用原 BFZI V150 的 `--sea-tangent-rank 8` 预生成同样的冷态电磁灵敏度锚。原 BFZI 教师时间步通过 `--teacher-step 15` 控制。

针对性验证：本轮新训练与混合推理用实际 BFZI 多几何动态数据和独立场文件检验；沙箱闭环/集成测试 **5 passed**，原 BFZI 环境训练测试 **10 passed**，独立灵敏度模块 **2 passed**、三维 ROM/灵敏度相关 **8 passed**。这些是不同运行批次，未合并为一个一次性总套件数。

## 7. 下一主线

1. 将冷态电磁灵敏度与分数阶网络联合训练为**物理切线上的小残差代理**，以减少每个新几何离线的16次原 BFZI方向求解，并维持已经实测的全复阻抗精度。
2. 扩展独立高功率场、任意相对位姿、连续圆角系数、封装真实尺寸与介质范围；用**整几何留出和跨时间步长留出**检验完整三维热反馈，避免仅依据标量网络输出选择最终模型。
3. 继续提高真三维海水 J/q 空间场与原 BFZI 的逐时刻一致性，保留海水 q 体积积分与复阻抗损耗、双铜热源和五材料热量预算。
