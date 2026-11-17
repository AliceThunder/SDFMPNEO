# V6 局部海水薄层分端口热切线 V1

状态：**开发集选模完成；全新 Z01–Z06 物理盲测待生成**。

## 1. 研究动机

V6 的动态阻抗由冷态端口、解析铜阻抗温度修正和海水动态阻抗切线组成。Y01–Y06 的历史独立验收显示，冷态阻抗误差很低，动态误差主要来自 RX 自阻抗分量。原切线主要使用 14 个全局海水热状态特征，不能直接表达 RX 邻近海水薄层的局部温度形状。

V7 已经提供 16 个局部海水薄层温度模态，TX/RX 各 8 个。本方法将这组已有温度表示接入动态阻抗预测，形成分端口局部热切线。

## 2. 模型结构

记共享 32 维热状态为 (a)，全局海水状态索引为 (I=[0,5,ldots,17])，V7 局部海水薄层系数为 (s=(s_{TX},s_{RX}))，每端口 8 维。几何与材料条件为 (g)。

保留原 V6 海水切线 (Z_{sea}^{base}(g,a))，只增加一个局部残差：

[
Delta Z_{sea} =
Delta Z_{sea}^{base} +
egin{bmatrix}
f_{TX}(g,a_I,s_{TX})\
f_M(g,a_I,s_{TX},s_{RX})\
f_{RX}(g,a_I,s_{RX})
end{bmatrix}.
]

输出顺序为 (R_{11},X_{11},R_{12},X_{12},R_{22},X_{22})。三个小型 MLP 分别学习 TX 自阻抗、互阻抗和 RX 自阻抗残差。每个分支减去同几何零热状态输出，确保冷态修正严格为零。铜导体温度阻抗继续使用原 BFZI 圆柱导体解析模型，不交由神经残差学习。

局部薄层特征由已训练的 `SkinFix` 解码器给出。它接收几何、32 维热状态、时刻和区间左端电压。推理端的阻抗切线因此需要时间和上一热步电压；`geom_free.rollout` 已把该因果信息传入时间条件化切线。冷态和传统两参数切线调用保持兼容。

## 3. 训练与开发集选择

训练入口：`examples/bfzi_geom_free_skintan.py`。

- 训练几何：原 V4 manifest 的 30 组、90 条历史。
- 开发几何：5 组、15 条历史。
- 只训练局部残差分支，冻结 V6 主网络、基础阻抗切线、V7 薄层解码器、冷态自/互阻抗代理和海水源代理。
- 训练状态输入来自 V6 自主预测的热状态；目标为 S5 教师海水动态阻抗减去 V6 基础切线输出。
- 闭环开发评分采用 `shape_mix=0.5`，以动态复阻抗相对 L2 选取最佳 epoch。
- 训练 manifest 中的 train/dev/test 几何集合会检查互斥；实际只读取 train 与 val，不读取 test。

开发集结果：

| 指标 | 原 V6 基础切线 | 局部薄层分端口切线 |
|---|---:|---:|
| 动态复阻抗误差 | 12.2046% | **10.8772%** |
| 匹配电流误差 | 6.9009% | 6.9021% |
| 五材料均温 RMSE | 0.35432 K | 0.35442 K |
| 五材料能量预算残差 | 1.01e-11 W | 1.01e-11 W |

最佳 epoch=80；共训练到第 140 轮后停止。开发集动态阻抗误差相对下降 10.88%。电流和温度变化很小，说明本次改进主要发生在阻抗输出支路。

代码编译和针对性测试通过：`tests/test_bfzi_geom_free.py`、`tests/test_bfzi_geom_skintan.py`、`tests/test_bfzi_geom_ztune.py`，共 5 passed。

训练报告：`runs/geomfrac_free_skin_tan_v1_dev.json`。状态：`runs/geomfrac_free_skin_tan_v1_status.json`。XEnergy 当前权重：`D:\\work\\geomfrac-train\\runs\\geomfrac_free_skin_tan_v1.pt`。

## 4. 新独立验收集合

Y01–Y06 的误差模式已用于确定本方法的研究方向，因此不再把它们当作本候选模型的盲测。新冻结 Z01–Z06：

- 6 组全新连续圆形/圆角方形几何；
- TX/RX 形状系数、内核尺寸、相对位姿、封装和热参数均有变化；
- 18 条 S5 物理时程，分别 60/300/600 秒；
- BFZI V150 S5、ultra 网格、1.875 秒内部热步；
- 不参与训练、开发评分、混合系数选择或模型筛选。

几何与驱动已冻结在 `runs/geomfrac_s5_free8_plan.json`。物理教师生成沿用具备逐历史失败隔离、即时保存和断点恢复的 `examples/bfzi_s5_buildset.py`：

```powershell
python -m examples.bfzi_s5_buildset --bfzi-root D:\\work\\geomfrac-s5-BFZI --plan runs/geomfrac_s5_free8_plan.json --tag free8 --grid ultra --inner-step 1.875
```

单条失败会记录失败原因并继续其他历史。完成后输出 `runs/geomfrac_s5_free8_manifest.json` 与 `runs/geomfrac_s5_free8_build_report.json`。

冻结候选只做一次完整评估：

```powershell
python -m examples.bfzi_geom_free_eval --plan runs/geomfrac_s5_free8_plan.json --manifest runs/geomfrac_s5_free8_manifest.json --split test --atlas runs/geomfrac_init_v1.npz --model runs/geomfrac_et_v4_8g_quality.pt --z-fix-model runs/geomfrac_et_v9_ztune.pt --field-atlas runs/geomfrac_field_v2.npz --skin-model runs/geomfrac_et_v7_skin_resid.pt --skin-tangent-model runs/geomfrac_free_skin_tan_v1.pt --port-self-model runs/geomfrac_portself_v5.npz --mutual-model runs/geomfrac_mutual_v6.npz --source-model runs/geomfrac_src_v6.npz --shape-mix 0.5 --out runs/geomfrac_free8_skin_tan_eval.json
```

## 5. 当前结论

局部薄层模态与分端口切线在既有开发集上带来明确改善，值得进入新几何验证。正式 V6 仍保持原始配置；局部薄层切线是候选版本，只有 Z01–Z06 的完整独立验收支持时才考虑替换。Z 集验收结果出来前，不继续用 Y 集选择参数。
