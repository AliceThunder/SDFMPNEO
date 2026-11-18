# BFZI V150 大规模冷态电磁数据与神经代理 V1

日期：2026-11-18

## 目标

将冷态电磁代理从电磁热历史样本中独立出来。在已有 V5 电磁热闭环完全冻结的基础上，生成新的冷态几何参考，用一个跨几何共享网络预测双端口冷态复阻抗。在线推理不调用 BFZI 冷态求解器。

第一批设计 500 组新几何，包含 400 组训练、50 组开发、50 组独立测试。首批固定 BFZI M4 200 kHz、原海水介质条件，覆盖两端口独立圆形/圆角方形参数、封装尺寸、XYZ 偏移和三个相对转角。材料变化随后在冷态数据主线上独立扩展。

## 当前实现

- `examples/bfzi_cold_plan.py`：确定性分层几何抽样、BFZI M4 十匝几何可行性约束，与已有 42G/5G 和 BB 诊断几何去重，固定三个集合及随机种子。运行后保存 `runs/geomfrac_cold500_plan.json`。
- `examples/bfzi_cold_generate.py`：复用 `examples.bfzi_param.create` 以及 `BFZIV148Adapter` 内部真正调用的 BFZI V150 原始源项、海水 Green/FV 和 Neumann/GMD 空气自感求解；在 293.15 K 进行一轮双端口电磁计算，不生成热算子、热 ROM、三维温度历史。每个样本单独子进程、有限时、成功原子保存为 `runs/geomfrac_cold500/samples/Cxxxx.npz`，失败写入 `progress.json` 并继续下一样本。断点恢复跳过已成功和已记录失败的样本；`--retry-failed` 可以重试失败项。
- 参考标签：`z0`（双端口 6 个 R/X）、`sea`（海水 6 个 R/X）、`air`（2×2 H）、`copper`（两端口复数电阻）、`omega`、`sigma`。末尾收集成按几何区分的 `data.json`。
- `sdfmpneo_bfzi/geom_coldnet.py`：独立几何冷态网络，输入 12 个几何/位姿/封装参数及四个绕组面积/周长趋势特征。输出双端口损耗电阻、去空气自感后的虚部和三个空气电感分量，组合成 `z0` 与 `air`。可复用 `apply(case,init)` 接口，替换现有冷态端口模型。
- `examples/bfzi_cold_train.py`：使用新增冷态样本与已有 42G 冷态训练参考拟合网络，在新增 50G 开发和原 T04/D01–D04 上选模，50G 独立测试只在选模结束后评估；不读取 BB 诊断样本进行训练或选模。
- `tests/test_bfzi_cold_data.py`：采样可重复/去重/几何有效性及冷态物理标签编解码，3 项通过。

## 实际执行与已有结果

XEnergy 使用路径：

- SDFMPNEO：`D:\work\geomfrac-v6`
- 原 BFZI V150：`D:\work\geomfrac-s5-BFZI`

固定计划已生成（seed 62026）：400 train、50 val、50 test，500 次候选采样；前三个真实 BFZI V150 样本 C0373、C0358、C0442 分别耗时 13.81、13.14、13.48 s，全部成功。完整 500 候选生产任务已启动（EXZA 任务 `job-2`，PID 32672），可通过远端 `progress.json` 和 `samples/` 检查已完成状态。

已观察到一组几何因旋转封装超出海水域而失败，失败已记录，后续样本继续成功执行。统计完成后，对缺少的有效几何按原 split 补充新的候选；不覆盖原计划中成功样本。

先用 42 个已有冷态训练样本加 3 个新参考验证了训练全链路：训练 45G、原 5G 开发；120 epoch 最佳检查点 120，冷态 Z 相对误差 2.8313%、空气自感相对误差 2.8493%、电流相对误差 23.6404%。这一结果只验证训练代码可运行，尚未达到 PortSelf V5 在相同开发集上的冷态 Z 约 0.83%、电流约 6.90% 的精度，不作为模型升级依据。报告见 `runs/geomfrac_coldnet_pilot.json`。

## 后续验收

1. 完成至少 400/50/50 有效且相互不重合的新增冷态训练、开发、测试几何；参考求解失败需逐样本记录并保留连续运行。
2. 使用独立冷态网络训练，分别比较原 5G 开发、新 50G 开发和新 50G 独立测试的自阻抗、互阻抗、空气自感及电路电流误差。
3. 固定冷态模型后接入现有 V5 电磁热闭环，保持 Atlas、热网络和 ZFix 不变，在相同样本、相同历史下衡量完整动态阻抗改进。
4. 只有在独立几何和闭环综合指标明显改善时才升级正式模型。

## 调用

```powershell
python -m examples.bfzi_cold_plan --samples 500 --seed 62026
python -m examples.bfzi_cold_generate --bfzi-root D:\work\geomfrac-s5-BFZI
python -m examples.bfzi_cold_train --data runs/geomfrac_cold500/data.json --old runs/geomfrac_cold_ports_v1.json
```

500 样本生产期间重启相同生成命令会跳过已保存的成功样本，并继续余下样本。全量完成前保留 V5 及 `geomfrac_portself_v5.npz` 不变。
