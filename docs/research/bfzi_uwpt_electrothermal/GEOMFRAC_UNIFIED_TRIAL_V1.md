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
