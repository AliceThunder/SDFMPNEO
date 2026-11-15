# GeomFracET 共享热坐标与跨几何训练 V2

日期：2026-11-15。

## 本轮解决的问题

旧 GeomFracET 在每个几何使用独立 32 阶热 POD/ROM 基。不同几何的模态顺序及空间形状不同，导致系数 \`a_1...a_32\` 缺少共同物理语义，不能直接当作跨几何神经监督标签。新模型需要一个可共享、材料能量守恒且可恢复空间温度的统一热坐标系。

新增 \`sdfmpneo_bfzi/geom_canon.py\`，直接从每个几何的冷态热网格中心、封装坐标、物理热容量生成固定语义的 32 个空间模式：

- **前 5 维**：海水（包含 S5 八层薄海水）、发射封装、接收封装、发射铜、接收铜的材料容量加权平均温升；
- **接下来 13 维**：海水几何空间的梯度、二阶与交叉形状、正弦形状和双封装近场分布；
- **接下来 7+7 维**：两个封装各自局部坐标下的一次、二次和交叉温度形状。

候选模式在各材料内部进行容量加权正交化，同时维持前五个材料均温基完全不变。封装模式针对原 S5 \`4×2×4\` 单元结构设计，避免厚度仅两层时二次项退化。新热基直接构造；**不需要逐几何运行温度脉冲解和 SVD 来生成 POD 热基**。

## 统一状态和守恒

每个几何保留其真实三维热容量、传热矩阵与材料坐标。令 \(\Phi_g\) 是该几何的容量正交基，\(C_{g,j}\) 是各模式所属材料的总热容量，则定义共享神经状态：

\[
a_g=\Phi_g^\mathsf{T}C_g(T-T_0),\qquad
s_{g,i}=\frac{a_{g,i}}{\sqrt{C_{g,\operatorname{mat}(i)}}}.
\]

于是前五个 \(s\) 系数**严格等于五种材料的容量平均温升（K）**。空间形状状态具有相同的温度量纲。对第 \(g\) 个几何，将其原 32 阶热传导方程变换到共享状态：

\[
\dot{s}_g + \operatorname{diag}(m_g)^{-1}K_g
\operatorname{diag}(m_g)s_g
=\operatorname{diag}(m_g)^{-1}q_g.
\]

这里 \(m_g\) 由各材料总热容量确定。训练对每个几何采用其自己的变换后热算子和真实教师源项；神经网络共享同一 GeomFrac 主干、SOE 时间记忆与热状态输出头。在线自主推理把神经状态变换回物理热基系数，继续使用原有 5×5 材料能量守恒校正与两端口电路。

## 新代码入口

- \`sdfmpneo_bfzi/geom_canon.py\`：确定性 32 维几何共享热基、材料容量尺度、物理状态投影、无脉冲 POD 的 ROM 建立。
- \`examples/bfzi_geom_multi.py\`：读入多个真实几何的 S5 冷态锚和完整教师时程、联合训练同一 GeomFracET、按照**未参与训练的几何**选择最佳网络并执行自主闭环验证。
- \`examples/bfzi_geom_neural.py\`：单几何版本保持兼容；追加逐几何共享状态、热算子与输入向量的数据路径。
- \`sdfmpneo_bfzi/geom_neural.py\`：通过检查点 \`canonical=True\` 切换共享热状态；使用原五分区物理守恒校正计算在线温度。
- \`examples/bfzi_geom_neural_infer.py\`：共享模式自动从冷态锚建立热基，推理无需逐几何 POD 文件；旧 40D/32阶 POD 模型仍可加载。
- \`tests/test_bfzi_geom_canon.py\`：模式正交性、材料平均温度不变性、两几何对齐、神经时间推进及多几何损失梯度。

## 数据清单与复现命令

使用**新非冻结训练几何**生成冷态锚和物理教师。示例 \`runs/geomfrac_multi.json\`：

\`\`\`json
{
  "train": [
    {"anchor": "runs/bfzi_s5_T01_ultra_anchor.npz",
     "teacher": ["runs/bfzi_phys_T01_train.npz", "runs/bfzi_phys_T01_train2.npz"]},
    {"anchor": "runs/bfzi_s5_T02_ultra_anchor.npz",
     "teacher": "runs/bfzi_phys_T02_train.npz"}
  ],
  "val": [
    {"anchor": "runs/bfzi_s5_T03_ultra_anchor.npz",
     "teacher": "runs/bfzi_phys_T03_val.npz"}
  ]
}
\`\`\`

示例文件名只是训练数据生成后所用的路径格式，并不代表这些教师时程已生成。模型自动识别 40/45 维输入，多个几何必须保持相同网络输入维数。冻结 U01–U12 不用作训练数据。

\`\`\`bash
python -m pytest tests/test_bfzi_geom_canon.py -q

python -m examples.bfzi_geom_multi \
  --manifest runs/geomfrac_multi.json \
  --init-model runs/bfzi_geom_diverse.pt \
  --epochs 500 --out runs/geomfrac_et_multi.pt \
  --report runs/geomfrac_et_multi.json

python -m examples.bfzi_geom_neural_infer \
  --anchor runs/bfzi_s5_T03_ultra_anchor.npz \
  --model runs/geomfrac_et_multi.pt \
  --voltage 100 --step 15 --horizon 600 \
  --out runs/geomfrac_et_multi_infer.npz
\`\`\`

共享热基需直接从同一物理冷态锚构建，不依赖 \`--rom-file\`。训练报告新增训练/验证几何数量、热基独立投影的三维温度最大误差、真实自主闭环的阻抗/铜温/封装温度/热预算误差，用于明确区分**热基自身表示误差**与**神经动态预测误差**。

## 当前完成状态与下一关口

本轮已经在沙箱以两组不同几何与约 5,000 节点合成热网格测试：共享材料均温准确、容量正交误差约 \(7.2\times10^{-15}\)、32 模态构造约 10 ms、神经在线时间推进和五材料守恒以及多几何反向传播通路通过。性能数据是合成网格局部测试，不代表 BFZI S5 实机加速。

**下一关口：** 真实非冻结 S5 几何及新激励教师数据，先测 32 维共享基对海水薄层热点和双封装三维温度的独立投影误差；若共享基表示误差限制了模型精度，再增强空间解码器或扩充专用薄层模式。随后完成联合训练，并在整组冻结 U01–U12 上进行独立电磁热代理验收。原 8 模态 EM 与 32 阶 POD 仍作为物理基线。
