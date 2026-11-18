# GeomFrac 独立热算子核宽度 V1

日期：2026-11-18

## 动机

GeomInit 原先用同一个 RBF 核宽度 `ell` 插值冷态输出、热容量、三维降阶热算子和弱热源。各输出具有不同的几何变化尺度：冷态端口/电感和热容量较平滑，32D 热算子中的局部导热耦合对几何变化更敏感。共用一个核宽度会让某一类输出的最优平滑程度约束另一类输出。

BB01–BB06 独立诊断中，42G 冷态/热算子候选使动态阻抗误差从 45.30% 降至 41.39%，但原开发集从 11.46% 升至 12.60%，因此不升级。用 BFZI 真实热容量与热算子替换 K/cap 后，开发动态 Z 从 12.60% 降至 11.96%；只替换热源则为 12.62%，全部替换为 11.99%。结果支持继续单独改进 K/cap 的跨几何表示。

## 方法

GeomInit 新增独立热算子核宽度 `kell`：

- `ell` 继续控制冷态输出、热容量和弱热源的 RBF 插值。
- `kell` 专门控制归一化 32D 热算子矩阵的凸权重。
- 热算子仍由训练锚点的归一化 SPD 矩阵作凸组合，再执行物理参数缩放；对称性和正定性保持。
- 正性保护所需的最近邻几何锚点也按热算子权重选择。
- 旧模型文件未包含 `kell` 时，自动使用 `ell`，保持已有检查点兼容。

`examples/bfzi_geom_init.py` 现在独立搜索 `kell` 与热物理缩放指数，并在原开发集上评估热算子 Frobenius 误差。冷态/热容量/热源插值仍由 `ell,lambda` 控制。

## 本轮相关实验

### 几何自由动态阻抗切线 V10

新增 `examples/bfzi_geom_free_ztune.py`，训练样本直接来自部署用的几何自由闭环：同一套冷态端口模型、32D 热算子、源项、SOE 热演化和基准海水切线。监督目标是 BFZI 完整动态阻抗与该闭环预测之间的差值。内部划分为 36 组训练、5 组原开发、6 组训练集内部留出。

| 指标 | 基准 V9 | V10 最佳联合选模 |
|---|---:|---:|
| 开发动态 Z | 11.4614% | 11.1790% |
| 内部留出动态 Z | 17.9126% | 18.1334% |
| 联合选模分数 | 14.6870% | 14.6562% |

V10 的开发集单项最佳为 10.9150%（epoch 30），联合选模最佳为 epoch 20。内部留出退化，联合分数仅改善 0.0308 个百分点，故不升级。

BB 已打开诊断集上，V10 动态 Z 为 45.1581%，V5 为 45.3022%；电流误差与温度误差基本不变，重复推理时间由约 0.1995 s 增至 0.2622 s。V10 不升级。

### 42G 热算子候选

采用 42G 热算子、原 V5 冷态端口/互感/热源模型，原开发集动态 Z 为 12.5990%，高于 V5 的 11.4614%。该候选不升级。

## 代码与验证状态

- `sdfmpneo_bfzi/geom_init.py`：增加 `kell`、独立热算子权重和旧模型回退。
- `examples/bfzi_geom_init.py`：增加 `kell` 与物理缩放指数联合搜索。
- `tests/test_bfzi_geom_init.py`：增加独立核宽度、热算子正定性、源项权重不变和保存/加载回归测试。
- 合成矩阵数值检查已通过：改变 `kell` 会改变热算子插值，源项插值保持不变，热算子仍为对称正定。
- 本轮新增的 pytest 回归测试尚未执行；上轮针对自由闭环入口的 6 项测试通过发生在本次 `kell` 改动之前。

## 当前决策与下一步

正式 V6 与 V5 均保持不变，V10 与 42G 热算子候选不升级。

下一步在 42G 真实 S5 锚点上重建独立 `kell` 热算子模型，先检查原开发集的 K/cap 和完整动态 Z，再决定是否值得进行新一轮 CC 独立物理验收。BB 只保留为已打开的误差诊断集，不参与模型选型。


## 36G 热状态头适配实验（待运行）

本实验从已验收的 `geomfrac_et_v5_closed.pt` 初始化，仅开放 `heat_head` 参数；冷态 TX/RX 自阻抗、互阻抗、热源模型和海水动态切线全部冻结。热算子由 36 组真实 S5 锚点重新拟合，独立核宽度由 `bfzi_geom_init.py` 在 11 组验证几何上搜索。训练使用 36 组几何/108 条历史，验证使用原 5 组开发几何加 6 组内部留出，共 11 组/33 条历史。六组 N 案例曾用于 V5 神经网络训练，本轮仅作为新 atlas 的插值回放验证，不属于独立神经网络留出；原 5 组开发几何仍是本轮未见几何。该 11 组集合仅用于训练过程选模，不作为最终独立验收。

训练入口：

```powershell
python -m examples.bfzi_geom_init --plan runs/geomfrac_pose42_plan.json --manifest runs/geomfrac_s5_v6_36g_hold6_manifest.json --rank 32 --out runs/geomfrac_init_pose42_kell_36g_v1.npz --report runs/geomfrac_init_pose42_kell_36g_v1_dev.json
python -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_v6_36g_hold6_manifest.json --init-model runs/geomfrac_et_v5_closed.pt --reuse-scales --heat-only --select free_dynamic_z --free-plan runs/geomfrac_pose42_plan.json --free-config runs/geomfrac_free_pose42_kell_v1.json --epochs 180 --patience 45 --val-every 5 --lr 0.0003 --batch-hist 36 --closed-weight 0.2 --closed-warmup 10 --closed-ramp 20 --out runs/geomfrac_et_pose42_kell_v1.pt --report runs/geomfrac_et_pose42_kell_v1.json
```

训练报告包含 `initial_selection_score` 和 `selection_improvement_pct`，用于候选与初始化 V5 在完全相同的 11 组/33 条历史上比较。原开发集 V5 的 11.4614% 是 5 组/15 条历史指标，只作历史参照，不能替代本轮同集基线。36G atlas 详细逐几何指标见 [atlas summary](../../../../runs/geomfrac_init_pose42_kell_36g_v1_summary.json)。当前状态：atlas 已完成，heat_head 训练正在 XEnergy 执行，模型升级验收待完成。


### 36G 热算子重建结果与闭环训练状态

36G/11V atlas 已在 XEnergy 重建完成：`ell=6.0`、`lambda=0.003`、独立热核宽度 `kell=0.6`，热算子缩放指数为 `(0.5, 0.25, 1.0)`。11 组验证几何的平均热算子 Frobenius 相对误差为 **3.791%**，热容量相对误差为 **0.00674%**，弱热源相对误差为 **9.279%**；验证热算子最小特征值均为正，平均约 `2.313e-5`。原始冷态 Z 平均相对误差为 2.154%。

`bfzi_geom_multi.py` 的 V5 初始化模型已在同一新 atlas、同一 11 组/33 条验证历史上完成闭环基线计算，`free_dynamic_z=16.5013%`。这是本轮 heat-head 训练的可比基线；此前 V5 的 11.4614% 对应原 5 组/15 条开发历史。仅开放 `heat_head` 的训练已启动，训练结果待完成后记录。


## 36G heat_head V1 训练结论（2026-11-18）

V1 训练已完成：36 组训练几何/108 条历史，11 组验证几何/33 条历史；最佳 epoch=55，实际运行 100 epoch。训练只改变 `heat_head.0.weight`、`heat_head.0.bias`、`heat_head.2.weight`、`heat_head.2.bias` 四个张量，其余网络参数与 V5 完全一致。

使用训练时相同的 `free_metric`、相同 atlas、端口自阻抗/互阻抗/源项修正及动态切线，重新对 V5 与候选模型进行 11 组/33 条 BFZI-free 闭环对照：总体 V5 16.50131%，候选 16.49222%，仅改善 0.00909 个百分点；原 5 组未见几何 V5 12.32451%，候选 12.62209%，恶化 0.29758 个百分点；6 组 N 几何回放 V5 19.98197%，候选 19.71732%，改善 0.26465 个百分点。逐几何结果见 `runs/geomfrac_et_pose42_kell_v1_free_eval.json`。

V1 的训练参数 `zrel_weight=0`、`tangent_loss_weight=0`，训练目标没有直接监督完整动态阻抗，BFZI-free 动态阻抗仅用于检查点选择。这解释了热头更新对闭环阻抗的作用很弱。候选不升级，继续使用 V5。下一轮固定相同数据划分、atlas 和其他参数，仅加入 `zrel_weight=0.5` 的动态阻抗监督，输出 V2 候选。


## 36G heat_head V2：加入动态阻抗监督（运行中）

V1 的 `zrel_weight=0`、`tangent_loss_weight=0`，使训练目标未直接约束完整动态阻抗；BFZI-free 动态阻抗只用于检查点选择。V2 在完全相同的 36G/108 历史训练集、11G/33 历史验证集、kell=0.6 atlas、V5 初始化权重和冻结冷态模型上，只增加 `--zrel-weight 0.5`，其余训练参数保持不变。`zrel` 项通过 `_zrel_full` 将预测海水阻抗变化与铜温引起的电阻变化共同纳入可微损失，训练仍仅开放 `heat_head`。基线为同一 BFZI-free `free_metric` 的 16.50131%。当前运行状态见 `runs/geomfrac_et_pose42_kell_v2_status.json`，候选待训练完成后按原 5 组未见几何与 6 组 N 回放分别评估。


## 36G heat_head V2 训练结论与目标错位（2026-11-18）

V2 训练完成 75 epoch，最佳 epoch=0，`zrel_weight=0.5`，选模误差与初始化同集基线均为 16.501306%。逐张量比较表明 V2 检查点与 V5 完全相同，说明所有训练更新均未通过部署闭环选模。相同 atlas、端口模型和 11 组/33 条历史的 `free_metric` 复核结果完全一致：总体 16.501306%，原 5 组未见几何 12.324512%，6 组 N 回放 19.981968%。V2 不升级。

根因已定位：V2 的 `zrel_weight` 作用于 `GeomFracET` 的 `zhat` 输出，而 `geom_free.rollout` 部署时忽略该输出，海水动态阻抗实际由 `geomfrac_et_v9_ztune.pt` 的 ZFix 切线模型根据热状态计算，再叠加铜温电阻变化。训练目标因此没有约束部署中使用的海水阻抗映射。另一个差异是当前 `score(closed=True)` 使用每个样本的 BFZI 热 ROM 做 `_heat_balance`，部署则使用跨几何 atlas 的 `shape_step/thermal_step` 和 `shape_mix=0.5`。

已更新训练接口：当传入部署 ZFix 模型时，`zrel` 现在可直接计算 `tangent(g,state)+copper(T)`；闭环源项与热平衡反馈也可调用同一 tangent。XEnergy `py_compile` 通过。单步实验中，闭环 tangent-zrel 在学习率 3e-4 下使原未见集误差增加 1.4912 个百分点；tangent 状态损失在学习率 1e-5 下增加 0.0892 个百分点。该结果确认仍需对齐 atlas 物理推进过程，当前不启动新的热头长训练。

下一步实现部署一致的可微训练推进：使用每个几何的 atlas 预测 `cap/K/src/z0/air/ext`，使用部署 ZFix tangent 计算海水阻抗，按部署相同的 `shape_mix`、`shape_step`、`thermal_step` 和电流反馈推进状态，再对完整动态 Z 误差反向传播。训练仍只开放 `heat_head`，V5 保持正式基线。V2 对照数据见 `runs/geomfrac_et_pose42_kell_v2_free_eval.json`。
