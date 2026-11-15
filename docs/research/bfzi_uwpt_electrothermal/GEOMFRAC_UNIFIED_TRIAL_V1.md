# GeomFrac 统一电磁热神经代理：第一轮试验 V1

## 目标

只保留 **GeomFrac 作为神经主干**，让同一个 SOE 记忆和几何编码驱动两个分支：

1. **电磁分支**输出海水动态复阻抗的六个实分量，替代在线八模态温度切线；
2. **热学分支**输出下一时刻的 32 维热状态，替代热 ROM 的在线隐式时间步。

铜内部阻抗、双端口电路和海水弱式热源仍采用已有物理公式。保留五个热学分区（海水、双封装、双铜）的积分能量预算。

现阶段保留已有冷态 BFZI 空间场、32 阶热基及 FV 热算子，供三维 T/J/q 重构和守恒校正使用。**在线时间推进无需原 BFZI Green/GMRES、八模态电磁灵敏度和 32 阶热隐式方程。** 这属于神经热时间推进的第一轮试验；还未取代新几何的热基构造。

## 网络与物理流程

\`\`\`text
几何条件（22/40D）──────────────┐
三路功率历史 ──> 分数阶 SOE ──> GeomFrac 共享 MLP (128×2)
                                     │              │
                            海水复阻抗增量6D   热状态更新32D
                                     │              │
                               双端口物理电路   五分区守恒修正
                                     │              │
                              实际电流/海水热源  三维温度重构
                                     └───────┬──────┘
                                        下一时间步
\`\`\`

一个时间步由当前记忆、三路加热量、几何条件、上一步热状态和时间步长预测新的海水复阻抗及热状态；随后利用铜阻抗和双端口电路求电流、功率，以 **5×5 线性修正**使五个材料分区的积分热量收支一致。没有调用原来的 32×32 隐式热算子求解。这个约束保持了材料级能量预算；局部三维热场精度仍需通过物理真值对照验证。

## 新增代码

- \`sdfmpneo_bfzi/geom_neural.py\`：\`GeomFracET(GeomFrac)\` 共享记忆与隐藏层，新增神经热状态头、五材料守恒修正和自主时程 \`rollout\`。
- \`examples/bfzi_geom_neural.py\`：物理教师完整时程输入、热状态与海水动态阻抗监督训练、保存模型和归一化尺度。
- \`examples/bfzi_geom_neural_infer.py\`：加载冷态锚、空间热基与新网络，生成动态 Z/I/T/J/q 时程 NPZ。
- \`tests/test_bfzi_geom_neural.py\`：热分支可训练性、五分区能量预算及教师数据训练入口的局部测试。

训练数据要包含 \`time_s\`、\`temp_K\`、\`z6\`，并通过 \`voltage_v\` 或 \`current_A\` / \`currents_A\` 提供完整激励。温度的 FV 节点排列和冷态复阻抗必须与 \`--anchor\` 及 \`--rom-file\` 一致。训练和验证使用不同的完整电压历史，绝不使用验证时程来拟合参数。模型输出温度摘要与海水增量损失，并以 32 维热状态变化作为另一个监督目标；在训练后期增加自由热状态滚动误差。

## 运行入口

\`\`\`bash
python -m examples.bfzi_geom_neural \
  --anchor runs/pose_anchor.npz \
  --rom-file runs/pose_rom.npz \
  --train runs/pose_train_1.npz runs/pose_train_2.npz \
  --val runs/pose_val_1.npz \
  --init-model runs/bfzi_geom_diverse.pt \
  --epochs 400 --out runs/geomfrac_et.pt \
  --report runs/geomfrac_et.json

python -m examples.bfzi_geom_neural_infer \
  --anchor runs/pose_anchor.npz \
  --rom-file runs/pose_rom.npz \
  --model runs/geomfrac_et.pt \
  --voltage 100 --heat 240 --cool 180 \
  --step 15 --horizon 600 \
  --out runs/geomfrac_et_infer.npz
\`\`\`

这里的 \`pose_*\` 为同一具体几何已生成数据的文件名示例，并非仓库预置的准确文件名。应替换为真实存在、物理定义一致的文件。

## 已完成与待完成

**已完成：** 新增可训练共享网络热头；纯神经海水动态阻抗输出；神经热状态一步及自主时程；五分区能量修正；训练、保存、加载和推理入口。沙箱简化算例已确认前向、反向传播与热量收支运算可运行，材料收支残差约 \(3.4\times10^{-13}\,\mathrm W\)。该测试使用简化物理对象，用于验证实现路径。

**待完成：** 在原 BFZI V150/S5 真实冷态锚和物理教师时程上实际训练；全新几何/热参数的独立测试；与原 8 模态和 32 阶热 ROM 的端到端精度与耗时对比；热模态在不同几何之间的对齐；跨几何坐标连续温度解码器。此前高精度混合代理的数字不等于本轮神经化架构的新验收成绩。

按本轮实验重点，应先在一个已建立冷态锚的真实几何上进行“同几何新历史”验证，再扩展多几何训练与冻结未见几何，逐步减少几何专属的热 ROM 构基。


## 2026-11-15：P07 真实物理时程首次训练与在线测速

这一步已经使用真实原 BFZI V150 历史数据运行，并非合成样本。训练前修复了热基实际秩与测试网络秩不一致的问题，增加了同一 NPZ 内多条独立时程的 `#0` / `#1` 选择器；相关 **12 项测试全部通过**。

- 几何：P07，40 维几何条件，32 阶热基。
- 冷态文件：`runs/bfzi_pose_p07_tangent_anchor.npz`；热基文件：`runs/bfzi_pose_p07_rom.npz`。
- 教师：`runs/bfzi_pose_p07_15s_holdout.npz`，两条不同历史各 41 时刻、600 s、15 s 采样、450 个原热节点。
- 训练：ID16（`#0`）；留出检查：ID18（`#1`）。初始权重 `runs/bfzi_geom_diverse.pt`，最多 300 轮，Adam，固定种子 41，历史状态滚动损失权重 0.2。
- 两个模型分支仍来自单一 `GeomFracET(GeomFrac)` 主干；在线不使用八模态电磁方向校正或热 ROM 隐式求解。
- 实际测试与运行文件：`runs/geomfrac_et_p07_trial.pt`、`runs/geomfrac_et_p07_trial.json`、`runs/geomfrac_et_p07_trial_infer.npz`。

| 热方程训练损失 | 最佳轮次 | 留出动态 Z 相对 L2 | 封装最大温差 | 铜最大温差 | 五材料预算最大残差 | 全 FV 节点残差最大 |
|---|---:|---:|---:|---:|---:|---:|
| 不使用热方程项 | 163 | 0.4980% | 1.2088 K | 0.3967 K | 8.66e-9 W | 0.3427 W |
| 绝对 ROM 热方程残差，权重 0.03 | 164 | 2.6484% | 5.9322 K | 2.0692 K | 9.94e-9 W | 1.4538 W |
| **相对教师的热方程残差，权重 0.03** | **122** | **0.3205%** | **1.1088 K** | **0.2349 K** | **8.77e-9 W** | **0.6500 W** |

教师温度投影到 32 维空间和 15 s 采样本身存在误差。因此直接逼迫预测满足零绝对 ROM 残差，会使网络偏离教师。最新实现改为相对于教师的热算子残差，并按模态热幅度和时间尺度归一化；同一数据划分下的阻抗与温度精度明显改善。材料分区积分守恒与所有 FV 节点热方程残差是两个不同指标；后者仍需改善。

**在线测速**：使用训练权重、P07 冷态锚和 32 阶空间基，150 V、210 s 通电 + 210 s 冷却、总长 600 s、41 采样点的自主时程实测 **0.1132 s**，最高铜温 **352.068 K**，材料收支峰值 **8.77e-9 W**。统计的是在线 `rollout` 本身，不含加载模型与冷态锚；同波形直接法速度比需在后续对照中确定。

**研究位置**：上述训练仅包含同一几何的一条训练历史与一条留出历史，属于真实物理数据上的功能/精度试验。P07 的旧独立留出历史在本次试验后已成为训练/验证来源，不能再将其用于本模型的独立跨几何验收。冻结 U01--U12 不参与神经训练，S5 高阶源积分+含细海水热层的物理数据仍需要专门的热节点/空间映射适配，之后再进行跨几何和热参数泛化验收。

推荐的复现命令：

```powershell
python -m pytest tests/test_bfzi_geom_neural.py tests/test_bfzi_geom_heat.py tests/test_bfzi_geom_rom.py -q
python -m examples.bfzi_geom_neural --anchor runs/bfzi_pose_p07_tangent_anchor.npz --rom-file runs/bfzi_pose_p07_rom.npz --train "runs/bfzi_pose_p07_15s_holdout.npz#0" --val "runs/bfzi_pose_p07_15s_holdout.npz#1" --init-model runs/bfzi_geom_diverse.pt --epochs 300 --patience 75 --closed-weight .2 --therm-weight .03 --out runs/geomfrac_et_p07_trial.pt --report runs/geomfrac_et_p07_trial.json
python -m examples.bfzi_geom_neural_infer --anchor runs/bfzi_pose_p07_tangent_anchor.npz --rom-file runs/bfzi_pose_p07_rom.npz --model runs/geomfrac_et_p07_trial.pt --voltage 150 --heat 210 --cool 210 --step 15 --horizon 600 --out runs/geomfrac_et_p07_trial_infer.npz
```

