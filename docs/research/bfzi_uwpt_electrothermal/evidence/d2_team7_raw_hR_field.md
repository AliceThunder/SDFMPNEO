# D2 TEAM-7 原始 Whitney 旋度 B 三维跨网格收敛

BFZI-PDR V1.1 · 2026-11-11 · **数值精度 OPEN**

## 实际计算与观察量

已在 XEnergy 上对 h10、gap06、gap05、gap04、rgap30、rgap40 六个独立真实 Elmer 200 Hz 工况取得原始有向边自由度：h10 复用已验证原生一步 `edge.result`；其余五个实际执行 `case_raw1.sif`（均 exit 0），导出 `raw1.result`。所有工况均在同一时间点 \(t=0.0003125\mathrm{s}\)，同一材料、源、6 mm 参考间隙/独立 h 轴和真空磁导率 \(\mu_0=1.256\times10^{-6}\mathrm{H/m}\)。

执行 `examples/bfzi_d2_team7_raw_conv.py` 时，在每个四面体上依据 **原始 Elmer A 边自由度直接计算常值 Whitney curl B**，再对三个相同的固定物理三维区域各 11³ 个位置进行所在四面体定位，直接取 **cell data**，按同位置向量 L² 差计算。整个流程没有调用节点化 Elmer B、DG 场重建或平滑。

每组原始旋度磁能量均与该组 Elmer 本机 `SaveScalars` 磁能量通过相对差 \(10^{-7}\) 的程序验收。制造解已有专项测试通过。

## 原始 B 场相邻差：一步实算

| 固定三维场区间与指标 | 较粗一档 | 较细一档 |
|---|---:|---:|
| 空气间隙 B，h：6→5→4 mm，相邻 L² 差 | 10.6299% | **8.3910%** |
| 上方空气 B，h：6→5→4 mm，相邻 L² 差 | 6.1932% | 6.0118% |
| 铝板内部 B，h：6→5→4 mm，相邻 L² 差 | 3.3284% | 4.1754% |
| 空气间隙 B，R：0.20→0.30→0.40 m，相邻 L² 差 | 11.5214% | **10.3119%** |
| 上方空气 B，R：0.20→0.30→0.40 m，相邻 L² 差 | 24.7520% | 23.5614% |
| 铝板内部 B，R：0.20→0.30→0.40 m，相邻 L² 差 | 3.4081% | 4.2540% |

h 加密在空气间隙原始 B 上确实降低相邻场差；R 相邻场差仍很大。部分铝板内部差未单调下降，因此保持 D2 OPEN。

作为同一数据的独立物理观察量，空气间隙 B 的 **RMS 幅值相邻差** 在 h 6→5、5→4 时仅为 0.267%、0.272%，R 0.20→0.30、0.30→0.40 时为 0.263%、0.260%。对应 **区域均值向量差除以参考 RMS** 为 0.146%、0.210% 和 0.553%、0.233%。这说明原始分片常值 B 的局部空间差异远大于整体能量/均值差，不能用 RMS 或平滑后的较低差值替代原始 B 的向量场收敛。

## 证据与复现

- [真实一步原始 B 场完整 JSON](d2_team7_raw_hR_200hz_1step.json)
- [原始 Whitney curl B 实现](../../../examples/bfzi_d2_team7_raw_b.py)
- [三维 h/R 同位置直接比较程序](../../../examples/bfzi_d2_team7_raw_conv.py)
- [同源原始磁能量一致性证据](d2_team7_whitney_raw_b.md)

复现：`python -m examples.bfzi_d2_team7_gap --root D:/work/BFZI-D2-verify/runs/team7_hr --elmerbin D:/work/ElmerFEM/ElmerFEM-nogui-nompi-Windows-AMD64/bin --case gap04 --raw-one --freq 200`，然后运行 `python -m examples.bfzi_d2_team7_raw_conv --root D:/work/BFZI-D2-verify/runs/team7_hr --steps 1 --n 11 --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_raw_hR_200hz_1step.json`。

后续继续真实**四周期终点**原始 A–φ 边自由度的 h/R 同位置对照，区分瞬态首步与最终物理场的收敛。当前 **TEAM-7 数值精度 OPEN、D2 OPEN、旧 G1 FAIL、神经代理教师冻结**。
