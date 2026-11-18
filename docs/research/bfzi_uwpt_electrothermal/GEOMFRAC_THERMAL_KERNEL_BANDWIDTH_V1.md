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

本实验从已验收的 `geomfrac_et_v5_closed.pt` 初始化，仅开放 `heat_head` 参数；冷态 TX/RX 自阻抗、互阻抗、热源模型和海水动态切线全部冻结。热算子由 36 组真实 S5 锚点重新拟合，独立核宽度由 `bfzi_geom_init.py` 在 11 组验证几何上搜索。训练使用 36 组几何/108 条历史，验证使用原 5 组开发几何加 6 组内部留出，共 11 组/33 条历史。六组内部留出只用于训练过程选模，不作为最终独立验收。

训练入口：

```powershell
python -m examples.bfzi_geom_init --plan runs/geomfrac_pose42_plan.json --manifest runs/geomfrac_s5_v6_36g_hold6_manifest.json --rank 32 --out runs/geomfrac_init_pose42_kell_36g_v1.npz --report runs/geomfrac_init_pose42_kell_36g_v1_dev.json
python -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_v6_36g_hold6_manifest.json --init-model runs/geomfrac_et_v5_closed.pt --reuse-scales --heat-only --select free_dynamic_z --free-plan runs/geomfrac_pose42_plan.json --free-config runs/geomfrac_free_pose42_kell_v1.json --epochs 180 --patience 45 --val-every 5 --lr 0.0003 --batch-hist 36 --closed-weight 0.2 --closed-warmup 10 --closed-ramp 20 --out runs/geomfrac_et_pose42_kell_v1.pt --report runs/geomfrac_et_pose42_kell_v1.json
```

训练报告包含 `initial_selection_score` 和 `selection_improvement_pct`，用于候选与初始化 V5 在完全相同的 11 组/33 条历史上比较。原开发集 V5 的 11.4614% 是 5 组/15 条历史指标，只作历史参照，不能替代本轮同集基线。当前状态：输入清单与配置已冻结，训练和模型升级验收待 XEnergy 执行。


### 36G 热算子重建结果与闭环训练状态

36G/11V atlas 已在 XEnergy 重建完成：`ell=6.0`、`lambda=0.003`、独立热核宽度 `kell=0.6`，热算子缩放指数为 `(0.5, 0.25, 1.0)`。11 组验证几何的平均热算子 Frobenius 相对误差为 **3.791%**，热容量相对误差为 **0.00674%**，弱热源相对误差为 **9.279%**；验证热算子最小特征值均为正，平均约 `2.313e-5`。原始冷态 Z 平均相对误差为 2.154%。

`bfzi_geom_multi.py` 的 V5 初始化模型已在同一新 atlas、同一 11 组/33 条验证历史上完成闭环基线计算，`free_dynamic_z=16.5013%`。这是本轮 heat-head 训练的可比基线；此前 V5 的 11.4614% 对应原 5 组/15 条开发历史。仅开放 `heat_head` 的训练已启动，训练结果待完成后记录。
