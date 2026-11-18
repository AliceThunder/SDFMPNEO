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

## 补齐有效样本与闭环验证

若首批候选存在封装越界等物理无效情况，初次任务结束后可运行：

```powershell
python -m examples.bfzi_cold_topup
python -m examples.bfzi_cold_generate --bfzi-root D:\work\geomfrac-s5-BFZI
```

`bfzi_cold_topup.py` 根据各 split 已完成的物理样本数量，为缺口增加新且不重复的候选并留出少量冗余。原 400/50/50 划分规则不变；新增样本继承原目标分组，保持已有成功数据不变。若仍有无效样本，可以重复补齐。

完成独立冷态模型训练后，用冻结的热状态 V5、现有 Atlas、热源代理和 ZFix 对比完整 BFZI-free 闭环：

```powershell
python -m examples.bfzi_cold_closed_eval --cold-model runs/geomfrac_coldnet_v1.pt
```

输出 `runs/geomfrac_coldnet_closed_v1.json`，要求原未见 5G 动态阻抗相对误差至少改善 0.1 个百分点、N 回放恶化最多 0.5 个百分点；BB 已打开的诊断集仅在选择完成后用于分析，不能据此选择模型。新增独立 50G 只进行冷态物理标签验收，除非为其另行生成动态参考。


## 2026-11-18 中途训练（不影响正在执行的数据生成）

当前生成器保持 EXZA `job-2` 原进程运行，最近一次状态快照为 292 成功、8 失败，物理无效样本已记录并跳过。为了研究样本规模效应，使用独立的 `bfzi_cold_snapshot.py` 从已有原子保存文件收集了 232 组新训练参考，并与原 42 组训练几何合并，在原 5 组开发几何上训练。该快照不修改原生产任务的 `data.json`，也不读取新 50G 独立测试集。

| 网络 | 原 5G 冷态 Z 误差 | 原 5G 电流误差 | 接入冻结 V5 动态 Z |
|---|---:|---:|---:|
| 原 PortSelf V5 | 0.8272% | 6.9017% | 12.3245% |
| 从头训练 274G ColdNet | 1.2167% | 10.1105% | 21.2558% |
| 274G 冷态残差 ColdRes | 0.7908% | 6.7245% | 12.6951% |

其中 `ColdRes` 保留原 PortSelf V5 的 TX/RX 自阻抗及冻结的互阻抗预测，由独立的小神经网络仅学习两端口 R/X 自阻抗残差；自感抗残差同时更新对应空气自感 `L_air`，以保持电路和海水分量物理一致。相关代码为 `sdfmpneo_bfzi/geom_coldres.py` 与 `examples/bfzi_cold_res_train.py`；零残差前向与自感抗/空气自感联动测试共 2 项，连同冷态数据测试合计 5 项通过。

这组中途结果说明数据规模扩张正在改善从头训练的冷态模型；残差学习也进一步改善冷态阻抗和电流误差。**原 5G 完整动态阻抗仍未改进，因此没有升级正式模型。** 新 50G 开发、50G 独立冷态测试尚未生成完毕，不报告它们的误差。

批次完成后的单入口：`python -m examples.bfzi_cold_finish --bfzi-root D:\work\geomfrac-s5-BFZI`。该入口检查原生产任务已结束，补齐各 split 有效样本至不少于 400/50/50，然后训练从头 ColdNet 和基于 V5 的 ColdRes，并保存新独立冷态测试结果；完整 V5 动态闭环评估继续通过 `bfzi_cold_closed_eval.py` 执行。

## 2026-11-19｜发现并修复参考几何有效性问题

初批 507 个可计算冷态参考在新 52G 独立测试上的同集指标：

| 模型 | 冷态复阻抗平均相对误差 | 旧 5G 完整动态阻抗误差 |
|---|---:|---:|
| V5 原始 PortSelf | 2.1847% | 12.3245% |
| 445G 冷态残差 ColdRes | 2.2221% | 13.1398% |
| 445G 从头训练 ColdNet | 3.0637% | 18.4140% |

误差最大的 C0459 在原 BFZI 12mm 引线错位下，两条 2.3mm 绝缘导线最近距仅 0.174mm，原生空气自感积分中 TX 绕组本体为 13.554μH，而引线项为 −7.749μH。对全部 517 个候选检查发现 201 个引线间距不合格；旧 47G 几何中有 27 个不合格，BB01/BB06 同样存在引线重叠。因此需要先修复数据的物理几何定义再对代理精度做判断，初批网络指标仅作记录。

已实现几何专用的 `coils_only` 快速入口、`bfzi_cold_lead_guard.py` 以及数据生成前的净距检查与自动补充有效几何。原始样本文件均保留。第一轮剔除后有 258/25/28 组有效成功参考；补样进程 EXZA `job-17` 已启动，新增候选 172/32/28，并按原 400/50/50 目标继续计算。理论分析与数字见 [冷态端部引线净距问题](GEOMFRAC_COLD_LEAD_CLEARANCE_V1.md)。

后续应使用净距合格的几何重训练；旧 42G 冷态标签中存在几何重叠的记录已通过 `bfzi_cold_finish.py` 的代码调整在下一次重训练时筛除。已有 V5 留作历史基线，真实物理独立验收需要重新建立没有引线冲突的动态样本。
