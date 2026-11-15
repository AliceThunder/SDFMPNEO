# BFZI S5 五阶物理教师到 GeomFracET 的薄层适配 V1

2026-11-15。目标：在现有 **GeomFrac 单主干** 与 32 阶热状态神经预测的基础上，接入含八层海水薄层和逐工况热参数的原 BFZI V150 五阶物理教师。原 S5 物理直接法和冻结验收结果不修改。

## 物理热态的统一顺序

原 S5 \`BFZIV148Skin3D\` 的温度状态顺序：

\`\`\`text
[海水 wet FV | 发射封装64 | 接收封装64 | 发射铜 | 接收铜 | 海水薄层1024]
\`\`\`

GeomFracET 的热降阶、材料功率与三维场采用：

\`\`\`text
[海水 wet FV + 海水薄层1024 | 发射封装64 | 接收封装64 | 发射铜 | 接收铜]
\`\`\`

\`s5_bridge.state_order\` 形成严格的一一对应排列，并将**热容量与整个稀疏传热矩阵同步重排**，不删除物理热节点。温度教师进入 \`examples/bfzi_geom_neural.read\` 时按保存的排列恢复。热 ROM 的五个守恒分区中，海水包括 wet FV 与全部薄层。

\`sea_layer.py\` 额外保存每个实际旋转封装表面薄层节点的三维中心 \`skin_xyz\`。原湿 FV 的单位电流场、弱式热源通过薄层体积分配矩阵 \`skin_mix\` 转移到薄层位置，保持积分源功率；在代理在线计算中继续执行海水总端口功率归一化。新锚保存 \`state_order\`、\`nwet\`、\`nskin\` 和热参数；原有 P03/P07 冷态锚仍兼容。

**电磁源分配精度的区别：** S5 物理教师的 \`skin_heat="source"\` 使用五阶近场求积点重建热源，在线神经代理使用冷态弱式热源在薄层上的体积一致映射。这是快速空间热源近似，必须单独比较三维热点与温度场，不能仅凭五分区能量收支判断空间精度已经通过。

## 几何与热参数输入

新 S5 条件向量由 **40 维**原 BFZI 几何/冷态双端口阻抗/位姿与封装参数，加上 **5 维**物理热参数构成：

\[
g_{45}=[g_{40},\,\alpha_\sigma,\,k_\mathrm{ins},\,t_\mathrm{ins},
           \,k_\mathrm{pkg},\,h_\mathrm{pkg/sea}].
\]

五个量的归一化尺度为 \`[0.02, 0.25, 0.001, 0.25, 20]\`。绝缘厚度没有人为指定时，读取原 S5 物理模型已根据导线几何计算的真实绝缘层厚度。这样材料变化由物理热算子和神经条件同时表示，而非只在数据文件名中出现。

训练/推理入口根据冷态锚的 \`thermal\` 信息自动使用 45 维网络；历史无热参数的 P07 模型仍按 40 维输入执行，不改变其权重结构。

## 冷态锚、32 阶空间热基和教师的构建

对于**冻结独立验收样本**，允许生成验证用冷态锚和热基；它们的动态轨迹不得加入模型训练：

\`\`\`bash
python -m examples.bfzi_s5_anchor \
  --bfzi-root ../BFZI \
  --spec-file runs/bfzi_physical_unseen12_corrected.json \
  --key U01 --role validation --grid ultra \
  --teacher runs/bfzi_phys_U01_ultra_dt15_h2_ik0.15_pk0.15_ph10_s5.npz
\`\`\`

该命令使用原 BFZI V150（五阶近源积分）构造相同的旋转线圈和双封装，进行**一次冷态电磁物理求解**，然后建立与薄层温度节点一致的 32 阶热基。默认输出：

- \`runs/bfzi_s5_U01_ultra_anchor.npz\`
- \`runs/bfzi_s5_U01_ultra_rom.npz\`
- \`runs/bfzi_s5_U01_ultra_metadata.json\`

教师数据验证自动检查冷态六分量复阻抗、温度长度、所有时间点和热状态投影维数。

用于训练的新几何应放在**另一个非冻结规格文件**中，使用 \`--role training\`。脚本对 U01–U12 强制禁止训练用途，避免在模型开发中泄露这 12 组独立验收样本。

\`\`\`bash
python -m examples.bfzi_geom_neural \
  --anchor runs/bfzi_s5_T01_ultra_anchor.npz \
  --rom-file runs/bfzi_s5_T01_ultra_rom.npz \
  --train runs/bfzi_phys_T01_train.npz \
  --val runs/bfzi_phys_T01_val.npz \
  --epochs 400 --closed-weight .2 --therm-weight .03 \
  --out runs/geomfrac_et_s5.pt \
  --report runs/geomfrac_et_s5.json
\`\`\`

示例 T01 是应另行生成的非冻结训练几何与历史，以上训练输入文件须先由原 BFZI V150 真实物理直接法生成。新 S5 模型训练后仍使用原 \`bfzi_geom_neural_infer.py\` 进行自主推理。

## 验证状态与下一阶段

沙箱中已通过 4 项针对性测试，包括状态排列可逆、薄层源的体积分一致性、真实热容量/热算子的逐元素保留，以及异常热节点维数拒绝。

本轮实现**教师—锚—热基—GeomFracET 训练接口**。S5 的原始 NPZ 为二进制物理场，未在当前沙箱取得整套源文件和原 BFZI 电磁依赖，因此本轮未执行新的真实 S5 训练，也未产生新的跨几何精度数值。此前已完成的 P07 真实神经热状态结果（动态 Z 0.3205%、在线 0.1132 s）仍是旧 40 维模型的实验数字。

下一任务：使用非冻结的 S5 五阶物理几何生成训练时程，并先对一只新几何进行完整空间/时间对照。进一步的**跨几何神经热态**还需要统一各几何的 32 阶热模态坐标，或改用几何条件化的空间温度解码器；当前几何专属热 ROM 的 32 维系数不应直接作为其他几何的同名监督标签。
