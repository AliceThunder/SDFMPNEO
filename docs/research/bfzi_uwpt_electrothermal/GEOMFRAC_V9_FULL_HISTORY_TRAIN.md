# V9：完整 BFZI 物理历史约束下的电磁热联合训练

## 数据

- 训练：`runs/geomfrac_s5_v8_full_manifest.json` 中 **48 组几何、132 条完整 BFZI V150 S5 ultra 电磁热历史**，每条保留采样时间、输入电压、三维温度场、48 维统一热状态、六分量复阻抗、电流与热功率。
- 开发：同一 manifest 中 **15 组互斥几何、41 条完整历史**。
- 初值：冻结 V8 ColdNet/GeomInit/SrcFix、SkinTanV3、ParZ48 与共享 rank48 SOE 热代理。原始 checkpoint 不覆盖。
- 之前1168个人工温度场电磁标签与2544个从真实历史拆分的热态标签只用于对照，不进入此阶段的主训练；避免用人工静态温度代替真实演化序列。
- 已冻结的 V8P/H/S 物理盲测不参与调参或候选选择。

## 核心实施

**A. 热演化优化。** 复用 `examples/bfzi_geom_multi.py` 的 SOE/热状态自回归训练和物理材料温度损失，`--heat-only` 仅更新现有热网络输出头；保留真实历史的电磁时间序列标签参与热状态对阻抗的影响监督。支持预测源反馈、温度、电流及阻抗联合损失。为避免线下验证与正式部署不一致，选择指标增加 `--free-parallel-model` 与 `--free-k-mix`，按冻结 V8 的 SkinTanV3+ParZ48+热算子混合0.25评估原15组几何的动态复阻抗误差；初值已核验为16.2321%。

**B. 电磁映射优化。** `examples/bfzi_v9_full_history_train.py` 读取完全相同的48/15几何完整历史。在冻结热网络的每条历史中自主 rollout，得到同时间戳的模型预测热状态 `a_pred(t)`；真实物理状态 `a_true(t)` 由完整三维温度场投影至共享48维空间。对同一物理复阻抗标签分别以 `a_true` 和 `a_pred` 训练现有 SkinTanV3，其他 ColdNet、ZFix、ParZ48 冻结，并保留到原始电磁映射的正则约束。验证使用完整真实激励历史自主闭环，与 V8 原模型在动态阻抗、TX/RX 单通道自阻抗、电流、材料温度和海水功率等指标下配对比较。仅在主要指标同步改善时保存候选。

**C. 耦合迭代。** 对从 A 选出的热网络及 B 选出的电磁模型，在冻结开发集上同条件对比。必要时以 A 的新热网络重新生成 `a_pred(t)`，再进行一次 B 优化。严格按实际开发集改善决定是否继续，而不无限扩充训练轮次或残差分支。

## 可复现命令

以下命令以 `D:\work\geomfrac-v6` 为当前目录，`PYTHONPATH=D:\work\geomfrac-v8`。

```bat
python D:\work\geomfrac-v8\examples\bfzi_geom_multi.py --manifest runs\geomfrac_s5_v8_full_manifest.json --rank 48 --init-model runs\geomfrac_et_v8_heat_tune.pt --reuse-scales --heat-only --epochs 30 --patience 12 --lr 0.00003 --batch-hist 16 --val-every 5 --closed-weight 0.2 --closed-warmup 0 --closed-ramp 5 --therm-weight 0.03 --cu-z-weight 0.02 --zrel-weight 0.08 --select free_dynamic_z --free-plan runs\geomfrac_v8_full_plan.json --free-atlas runs\geomfrac_init_v8_expand.npz --free-config runs\geomfrac_free_rank48_v8_selection.json --free-cold-model runs\geomfrac_cold_v8_frozen.pt --skin-tangent-model runs\geomfrac_rank48_skin_tan_v3.pt --skin-tangent-base runs\geomfrac_et_v9_ztune.pt --free-parallel-model runs\geomfrac_v8_par48.pt --free-k-mix 0.25 --out runs\geomfrac_v9_full_heat.pt --report runs\geomfrac_v9_full_heat.json

python D:\work\geomfrac-v8\examples\bfzi_v9_full_history_train.py --heat-model runs\geomfrac_et_v8_heat_tune.pt --epochs 25 --lr 0.00004 --pred-weight 0.5 --prior 0.4 --val-every 5 --out runs\geomfrac_v9_full_skin.pt --report runs\geomfrac_v9_full_skin.json
```

## 记录

2026-11-20：两个训练任务在XEnergy启动，热网络优化job-172，电磁映射优化job-173。初始自主闭环开发集均为 **16.2320984%**，原 TX R11 自身误差15.7848%、RX R22 自身误差20.7277%；主模型原始 V8 checkpoint 不修改。训练结果与是否晋级待完整开发集结果确定。
