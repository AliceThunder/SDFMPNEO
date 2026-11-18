# 无 BFZI V5：跨姿态训练扩展与 AA 独立验收

状态（2026-11-18）：**V5 训练结果尚未取得，XEnergy 当前离线；AA 原 free9 教师计划作废，已冻结参数正确的 free10 计划。V6 保持正式主线。**

## 1. 为什么启动主网络 V5

局部海水薄层热阻抗残差 V1–V4 在原开发集误差下降约 10%–11%，但在 Z01–Z06（18 条 S5 历史）均未超过原 V6。V6 基线动态阻抗误差 40.0937%，V1–V4 分别 40.1525%、40.4128%、40.3501%、40.1966%。

原训练集对相对姿态特征覆盖不足。V5 将已参与研究分析的 Z01–Z06 正式加入训练，只保留原 5 组开发几何，微调基础 GeomFracET 主网络，重点改善 32D 热状态与动态阻抗的跨姿态表示。

- 训练：原 30 组几何 + Z01–Z06，共 **36 组、108 条历史**。
- 开发：T04、D01–D04，共 **5 组、15 条历史**。
- 训练 manifest：`runs/geomfrac_s5_v5_36g_manifest.json`。
- 初始化：`runs/geomfrac_et_v4_8g_quality.pt`；沿用原归一化。
- 300 轮上限、patience 60、学习率 0.0003，按原训练程序的 `dynamic_z` 指标选模。
- V6 的共享 32D 热物理坐标、SOE、五材料能量校正、冷态端口和全场解码架构保持不变。

已启动的 XEnergy 训练命令（连接恢复后检查进程和输出，不应重复启动相同训练）：

```powershell
python -u -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_v5_36g_manifest.json --init-model runs/geomfrac_et_v4_8g_quality.pt --reuse-scales --epochs 300 --patience 60 --lr 0.0003 --select dynamic_z --zrel-weight 1.0 --history-weight step --out runs/geomfrac_et_v5_pose_aug.pt --report runs/geomfrac_et_v5_pose_aug.json
```

训练日志位于 XEnergy `D:\\work\\geomfrac-train\\runs\\v5_pose_train.log/.err`。上次确认初始开发 `dynamic_z` 为 0.163145；该指标来自训练程序的神经 rollout，最终独立验收仍使用无 BFZI V6 完整闭环计算入口。当前无法确认训练是否结束。

## 2. AA 物理验收计划参数错误与处置

原始 `runs/geomfrac_s5_free9_plan.json` 在生成六组 AA 物理样本时，发生了材料数组索引错位：

- `alpha_per_K` 被错误填入绝缘导热率 `ins_k`。
- `thermal.ins_k` 被错误填入绝缘层厚度 `ins_t`。
- `thermal.ins_t` 则被统一设为 0.0004 m，原有逐样本绝缘厚度未采用。

这会显著改变铜/海水温度反馈和导热过程，原 `free9` 标签下生成的所有教师文件 **全部排除出独立验收**，包括已经成功计算的 AA01/60 s。旧计划被标记 `invalid_for_physics`，构建器和评估器均拒绝该计划。

纠正后的六组 AA 参数已冻结为 `runs/geomfrac_s5_free10_plan.json`，几何、位姿、封装和驱动保持不变：

| 几何 | alpha_per_K | ins_k (W/m/K) | ins_t (m) |
|---|---:|---:|---:|
| AA01 | 0.018 | 0.27 | 0.00041 |
| AA02 | 0.022 | 0.35 | 0.00034 |
| AA03 | 0.017 | 0.20 | 0.00046 |
| AA04 | 0.024 | 0.37 | 0.00031 |
| AA05 | 0.019 | 0.18 | 0.00049 |
| AA06 | 0.020 | 0.39 | 0.00033 |

AA01–AA06 未用于 V5 训练或开发集调参，修正后的 free10 是本轮独立验收唯一有效参考计划。务必使用 **新的 `--tag free10`**，避免复用旧 free9 错误材料参数产生的锚与时程。

## 3. 物理参考生成恢复

XEnergy 目前离线，上一轮 `free9` 进程的实际最终状态未知。连接恢复后先确认旧 `free9` 进程是否仍在，若在运行则终止它，保留 V5 训练进程与已有训练输出。然后取回本分支中的正确计划和最新生成器，在 `D:\\work\\geomfrac-train` 运行：

```powershell
git fetch origin research/bfzi-uwpt-electrothermal-surrogate
git restore --source=FETCH_HEAD --worktree -- runs/geomfrac_s5_free10_plan.json examples/bfzi_s5_buildset.py examples/bfzi_geom_free_eval.py

python -u -m examples.bfzi_s5_buildset --bfzi-root D:\\work\\geomfrac-s5-BFZI --plan runs/geomfrac_s5_free10_plan.json --tag free10 --grid ultra --inner-step 1.875
```

生成器对每条历史独立处理异常、逐条持久化，失败后继续后续样本；恢复运行会跳过已标记 `candidate` 且文件存在的成功历史。

输出 `runs/geomfrac_s5_free10_manifest.json`、`runs/geomfrac_s5_free10_build_report.json` 和 6 个新冷态锚、18 条 60/300/600 秒教师历史。仅 18/18 条真实 S5 物理历史状态均为 `candidate` 时启动完整验收。

## 4. V5 与 V6 的同集对照

V5 完成开发集训练且 free10 独立教师齐备后，使用完全相同的冷态自/互阻抗代理、海水热源代理、形状校正与 48 模态温度场重建，仅替换主神经网络 checkpoint。

```powershell
$common = @(
  '--plan','runs/geomfrac_s5_free10_plan.json',
  '--manifest','runs/geomfrac_s5_free10_manifest.json',
  '--split','test',
  '--atlas','runs/geomfrac_init_v1.npz',
  '--field-atlas','runs/geomfrac_field_v2.npz',
  '--z-fix-model','runs/geomfrac_et_v9_ztune.pt',
  '--skin-model','runs/geomfrac_et_v7_skin_resid.pt',
  '--port-self-model','runs/geomfrac_portself_v5.npz',
  '--mutual-model','runs/geomfrac_mutual_v6.npz',
  '--source-model','runs/geomfrac_src_v6.npz',
  '--shape-mix','0.5'
)
python -m examples.bfzi_geom_free_eval @common --model runs/geomfrac_et_v4_8g_quality.pt --out runs/geomfrac_free10_v6.json
python -m examples.bfzi_geom_free_eval @common --model runs/geomfrac_et_v5_pose_aug.pt --out runs/geomfrac_free10_v5.json
```

验收指标包括动态与冷态六端口复阻抗、匹配电流、海水/铜耗、五材料温度、完整三维温度场 RMSE、能量预算、新几何首次及重复推理耗时。V5 决策以 AA01–AA06 的全量结果和逐几何分布为依据；原 V6 与正式锚点 V9 均保留。

状态文件：`runs/geomfrac_et_v5_pose_aug_status.json`。

## 5. 2026-11-18：重新连线后的训练结果与闭环选模修复

XEnergy 恢复在线后确认：

- 原 V5 训练共完成 60 轮，最优 epoch=0，`best_score=0.16314533246528498`；
- 原 V5 保存的 `state` 与初始化 V6 的 `state` 逐参数最大绝对差 **0.0**，未得到新的有效检查点；
- 原 CLI 使用 `--closed-weight 0.2`、默认 warmup=20、ramp=60，选模应从第 80 轮开始；旧代码按 `patience=60` 提前退出，导致模型不可能进入有效选模阶段；
- 旧 `dynamic_z` 依据 GeomFracET 自身 SOE 输出端口阻抗选择检查点；无 BFZI V6 在线预测中的海水阻抗来自独立的 `geom_ztune` 热切线，Ge​​omFracET 的直接阻抗输出在线上被绕过，因而旧目标与实际部署不一致。

已修改 `examples/bfzi_geom_multi.py`：

1. 早停计数在闭环源 warmup+ramp 完成后才生效；
2. 新增 `--select free_dynamic_z`，直接在原 5 组开发几何上使用 `geom_free.rollout`，保留冷态自阻抗、互阻抗、热源、物理切线和 `shape_mix=0.5`，以真实无 BFZI 闭环动态 Z 选择最佳 epoch；
3. 新运行只训练 `heat_head`，冻结不参与当前在线端口预测的 SOE 输出头及其他网络参数。

已启动新的 `runs/geomfrac_et_v5_closed.pt` 训练，初始完整在线开发动态 Z 误差为 **0.122046**（12.2046%，与先前完整开发闭环基线一致）。训练仍使用 36 组、108 条历史，开发 5 组、15 条历史，AA 不参与训练或选模。计划 160 轮上限，batch_hist=36，patience=50，闭环源权重 0.2、warmup=10、ramp=20。

旧 `free9` 生成过程最终得到 15 条 `candidate`，AA06 冷态锚失败（封装越出 BFZI 海水域）。上述旧教师都属于错误热参数计划，全部无效。正确 `free10` 计划对 AA06 的位置、旋转和封装进行了物理可行性调整，其余几何保持不变；现已从头生成参考，起始 AA01/60s 与 300s 两条均已成功。

新状态见 `runs/geomfrac_et_v5_closed_status.json`。新训练与 free10 参考完成之前，继续保留正式 V6。
