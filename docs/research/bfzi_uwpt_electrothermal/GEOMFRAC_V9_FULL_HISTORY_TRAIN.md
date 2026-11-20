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

## 2026-11-20 完整历史时序训练结果

**完整历史热演化优化：** XEnergy job-172 完成20轮训练，开发集最优 epoch5。与冻结V8同条件比较，15几何41历史动态复阻抗误差16.23210%→**15.97331%**，TX R11自身15.78483%→**15.43288%**，RX R22自身20.72770%→**20.41097%**，X11自身13.94079%→**13.54133%**，X22自身15.45681%→**15.18780%**，材料平均温度RMSE0.452111→**0.445394K**，电流8.61544%→8.61519%，海水功率16.47941%→16.47964%，铜功率16.35969%→16.35675%。仅热网络输出头更新；ColdNet、共享基、SkinTanV3、ParZ48和铜模型均保持冻结。开发集达到候选标准；模型 `runs/geomfrac_v9_full_heat.pt`，报告 `runs/geomfrac_v9_full_heat.json`、`runs/geomfrac_v9_full_heat_val.json`。

**完整历史电磁校准：** 两次使用132条真实BFZI物理历史，先以原V8热网，再以V9新热网生成各2544个真实温度/自主预测温度配对，共每轮5088个状态，优化现有SkinTanV3，不增加新的网络头。第一次25轮无改善，动态Z16.2321%→16.3989%，RX自身20.7277%→22.5882%；第二次25轮无改善，动态Z15.9733%→16.0359%，RX自身20.4110%→22.0486%。两次均未保存替换checkpoint，主电磁映射保留V8；报告分别为 `runs/geomfrac_v9_full_skin.json` 和 `runs/geomfrac_v9_full_skin_heat.json`。

**独立物理验证：** 新 `runs/geomfrac_v9_full_blind_plan.json` 已在生成参考前提交冻结。V9B0001–4分别来自从未用于V8动态训练/开发或以往V8P/H/S/E验收的 V7E0070、V7E0072、V7E0082、V7E0141，4组全新几何，每组60s恒压与300s分段关断激励，共8条完整BFZI V150 S5 ultra物理历史。生成任务 XEnergy job-176 正在运行；独立验收只比较原V8与冻结开发候选，不使用盲测选择额外超参数。

具体机器可读开发集对比：`runs/geomfrac_v9_full_train_summary.json`。
