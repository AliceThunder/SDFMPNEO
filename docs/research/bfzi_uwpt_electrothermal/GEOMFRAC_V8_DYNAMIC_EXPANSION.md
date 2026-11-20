# V8 动态几何扩展与共享热网络更新

日期：2026-11-20
状态：实施中；正式V7保持冻结。物理数据与训练在 XEnergy 执行。

## 研究动机与已有证据

V8 冻结ColdNet主干+四个新增铜阻抗输出，使海水损耗功率历史
相对误差由V7的24.7713%降至1.9263%，冷态复阻抗精度仍为0.14668%。
动态复阻抗增量相对误差约10.31%，高于V7的9.92%。

仅调整原SkinTan六个阻抗通道增益，验证误差22.5327%降至22.382%；
六组历史核心几何约10.31%降至10.1736%，收益有限。
增益实验见 `examples/bfzi_v8_gain.py` 及XEnergy
`runs/geomfrac_v8_gain.json`。

物理教师热状态部分替换（直接重算对应阻抗，不修改推理模型）：
- V8自主热状态：历史6几何/12历史平均动态误差10.3109%。
- 仅替换真实五材料平均热状态：**7.0200%**。
- 仅替换真实16个海水表层模态：10.20398%。
- 替换全部热状态：12.2106%。

这说明优先改进五材料平均热状态的闭环预测具有更高潜在收益，
整场真实状态直接代入原切线并不能保证阻抗更准。
结果保存于XEnergy `runs/geomfrac_cold_v8_heat_oracle.json`。

## 新动态几何选择

新增 `examples/bfzi_v8_dyn_expand.py`。
以冻结 V7 冷态**训练**样本挑选12组新增训练几何，
以冻结 V7 冷态**验证**样本挑选4组新增验证几何。
使用已有几何归一化参数，按离已选样本最远的准则选择，
在核心域约束内增强覆盖。冻结 V7 冷态测试集及原
V7六组核心动态历史均不用于样本选择或网络训练。

冻结清单：`runs/geomfrac_v8_dyn_expand_plan.json`。
训练样本编号 V8T0001–V8T0012；验证样本编号
V8V0001–V8V0004。每组包括一条60s和一条300s脉冲历史。

物理参考采用BFZI V150五材料S5直接法，`ultra` 网格，
内部热时间步1.875s，输出间隔15s。
生成器原有的逐样本结果写盘和失败后继续逻辑保持不变。
物理样本路径保留在 XEnergy `D:\work\geomfrac-v6\runs`，
manifest `runs/geomfrac_s5_v8expand_manifest.json`，
日志报告 `runs/geomfrac_s5_v8expand_build_report.json`。

执行命令（可安全重复，已完成样本会按报告跳过）：

```powershell
cd D:\work\geomfrac-v6
$env:PYTHONPATH="D:\work\geomfrac-v8"
python D:\work\geomfrac-v8\examples\bfzi_s5_buildset.py --bfzi-root D:\work\geomfrac-s5-BFZI --plan runs\geomfrac_v8_dyn_expand_plan.json --runs runs --grid ultra --tag v8expand --duration 300
```

## 不新增补偿网络的热学优化

直接微调既有共享rank48热代理 `GeomFracET` 的热输出层，
冻结电磁SOE和主网络其它部分。
对 `examples/bfzi_geom_multi.py` 增加
`--free-cold-model` 以在模型选择时采用V8铜阻抗正确分解，
并增加 `--expand-manifest`、`--expand-plan` 复用已有
训练数据和新物理几何。

已启动未扩展数据上的热输出层试训，权重拟输出为
`runs/geomfrac_et_v8_heat_tune.pt`，开发报告为
`runs/geomfrac_et_v8_heat_tune.json`。
新动态参考完成后，采用新的训练/验证清单再次训练并与该
试训对照。所有数据划分和实验权重与V7正式模型隔离。

## 选择与验收

开发集选择时优先动态复阻抗增量误差，
同时检查端口电流、五材料温度、海水功率和能量预算。
六组V7核心历史只用于完成开发后回溯。
正式替换需要新冻结的未见几何动态测试，目标包括：

- 动态阻抗增量误差低于V7历史基线9.9217%。
- 海水功率误差维持约2%的水平。
- 材料平均温度RMSE低于原V7的0.2574 K。
- 保持零BFZI/网格在线推理成本。

在达到目标之前，V7仍为正式模型，V8为研究候选。

## 训练热源与V8推理一致性修复

在将现有共享热网络接入V8闭环时，发现旧训练代码仍用
`total_z - jωL_air - winding_copper` 估计海水发热，
其中包括应留在电路端口外部的静态引线阻抗。
这种训练输入与V8推理期间的显式铜阻抗分解不同。

新增的V8训练路径通过同一个冻结ColdNet取得每个参考几何的
`copper0`，计算
`lead0 = copper0 - port.copper(T0)`：
- 对历史数据的3通道教师热源输入重新调用
  `heat_drive(...,lead=lead0)`，并保留原参考温度和阻抗。
- 对训练闭环可微的 `_port_src` 在海水功率项中同步减去`lead0`。
- V7原训练入口在没有 `--free-cold-model` 时保持原先行为。
- 首次试训在发现训练/推理热源不一致后已中止，正式试训
  使用新的正确热源重新启动。

代码：`examples/bfzi_geom_multi.py`、
`examples/bfzi_geom_neural.py`。
在XEnergy完成语法检查与`tests/test_bfzi_cold_split.py`测试（2 passed）。

## 首批新动态参考与热算子图谱候选

首批 V8T0001–V8T0003 已完成6条动态历史。
使用正式V7共享热网络加冻结主干V8铜阻抗分解，3组
新训练几何上的独立动态历史平均结果：

| 指标 | 新训练几何 |
|---|---:|
| 冷态复阻抗相对L2 | 0.04794% |
| 动态阻抗增量相对L2 | 15.6646% |
| 电流相对L2 | 0.7019% |
| 五材料平均温度RMSE | 0.6787 K |
| 海水功率相对L2 | 0.6220% |
| 铜耗功率相对L2 | 1.1500% |

结果 `runs/geomfrac_v8_new_dyn_eval.json`（XEnergy）。

对新增的3个训练热算子锚点，复用现有`GeomInit.fit`重建
共享rank48冷/热图谱，沿用原先核带宽、正则化、
热参数缩放与独立线圈自阻抗插值。
实现 `examples/bfzi_v8_atlas_expand.py`。

在11组原开发几何上的热算子K相对F误差
3.8050% → 3.7952%；1组全新开发几何
V8V0001上的K误差5.2770% → 5.0991%。
该新增验证几何60s物理历史上的动态阻抗增量误差
5.4134% → 5.3853%，材料温度RMSE
0.04839K → 0.04541K。

这些收益来自离线追加几何锚点，在线仍是单一共享热算子。
`runs/geomfrac_init_v8_expand.npz` 为候选模型，
`runs/geomfrac_init_v8_expand.json` 记录图谱误差。
尚待V8V0001的300s长脉冲历史和其余几何验证，
不替换正式V7。

## 首批样本完成及长脉冲验证

V8T0001–V8T0003、V8V0001 共4组几何8条历史
均以BFZI V150参考生成成功，失败0条。
剩余V8T0004–V8T0012、V8V0002–V8V0004共12组
已由XEnergy本地连续生成队列启动（脚本`runs/geomfrac_v8_queue.py`）。
生成结果按每条历史保存，记录在
`runs/geomfrac_s5_v8expand_build_report.json`。

V8V0001两条完整动态历史的单几何均值对比：

| 指标 | 原共享热图谱 | 追加3锚点的热图谱 |
|---|---:|---:|
| 动态复阻抗增量相对L2 | **9.1372%** | 9.1487% |
| 材料平均温度RMSE | 0.10203 K | **0.09486 K** |
| 材料温度最大绝对误差 | 0.35671 K | **0.33658 K** |
| 海水功率相对L2 | 1.2272% | 1.2272% |

新图谱改善温度，动态阻抗增量基本持平且有极小退化，
因此仍作为实验候选。相关XEnergy文件：
`runs/geomfrac_v8_new_val_base.json`、
`runs/geomfrac_v8_new_val_atlas.json`。

新的共享热网络适配训练使用完整的108+6=114条
训练历史、33+2=35条验证历史；
输入海水功率已与V8铜阻抗分解一致。
通过 `--batch-hist 24` 对原 rank48 网络热输出层
进行小批量微调，按自主闭环动态阻抗选模。
输出文件待生成：
`runs/geomfrac_et_v8_heat_tune.pt`、
`runs/geomfrac_et_v8_heat_tune.json`。

## 冻结全新几何的最终验收集

通过`examples/bfzi_v8_final_plan.py`从冻结V7冷态**测试集**
选择4组未参与当前V8训练和模型选择的新几何，
并排除历史六组中曾使用过的V7D来源。
已将几何及工况冻结到仓库
`runs/geomfrac_v8_dyn_holdout_plan.json`：

- V8E0001 ← V7E0039
- V8E0002 ← V7E0078
- V8E0003 ← V7E0060
- V8E0004 ← V7E0023

这4组只用于开发结束后的最终独立BFZI动态测试，
生成标签之前不参与任何训练或参数选择。

## 持续任务（XEnergy）

首次4几何8条参考已经完成，成功8、失败0。
`runs/geomfrac_v8_queue.py`作为EXZA终端任务继续
生成余下12组；逐条历史保存，无需等待整个批次完成才落盘。

热网络训练采用首批3新增训练几何/6历史与1新增验证几何/2历史，
连同原数据总共114条训练历史、35条验证历史。
已验证教师热源修复生效并读取
`V8_THERM_EXPAND 3 1`、
`V8_THERM_CORRECTED_SOURCES 114 35`。
训练前开发闭环动态阻抗误差为16.6415%，正在通过现有
rank48热输出层微调选择候选。未完成的任务不作为成果验收。
