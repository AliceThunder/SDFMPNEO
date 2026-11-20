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

## 2026-11-21 全新几何8历史独立盲测完成

XEnergy BFZI V150 S5 ultra 物理直接法已生成V9B0001–0004的全部8条动态历史（每几何60s、300s），成功8、失败0，中断后的任务恢复跳过全部已成功历史。参考与结果已提交：

- `runs/geomfrac_s5_v9full_manifest.json`、`runs/geomfrac_s5_v9full_build_report.json`
- `runs/geomfrac_v9_full_blind_base.json`、`runs/geomfrac_v9_full_blind_heat.json`
- `runs/geomfrac_v9_full_blind_summary.json`

冻结V8与冻结开发集选出的V9仅热网络输出头候选同条件对比：

| 指标 | V8 | V9-heat |
|---|---:|---:|
| 动态复阻抗误差 | **9.7281%** | 10.1546% |
| TX R11自身动态误差 | **10.9924%** | 11.1986% |
| RX R22自身动态误差 | **11.4380%** | 12.0146% |
| X11自身动态误差 | **8.8300%** | 9.8625% |
| X22自身动态误差 | **10.8688%** | 10.9039% |
| 五材料均温RMSE | 0.183957K | **0.181775K** |
| 电流误差 | 3.31792% | 3.31768% |
| 海水功率误差 | 3.09140% | 3.09021% |
| 铜功率误差 | 2.35473% | 2.35567% |

按几何拆分，动态阻抗误差（V8→V9）：V9B0001 8.8833→8.7172%、V9B0002 **14.3162→16.3090%**、V9B0003 8.4073→8.4409%、V9B0004 7.3056→7.1514%，2/4改善。主要恶化集中V9B0002，该几何材料温度RMSE也0.31463→0.33895K，R22自身误差12.0351→14.5879%，X11自身误差19.7112→23.5792%。另三组温度RMSE均改善。

**结论：冻结V9热头仅为开发研究候选，未通过全新几何验收，不晋级正式V8。** 这次盲测数据封存，仅用于事后归因，不再用来选新模型超参数。下一轮把重点移到真实全时序物理标签约束的跨几何海水复阻抗温度敏感度，避免继续单独微调热头或采用高维附加静态残差。正式V8与V7权重不变。

## 2026-11-21 区域复阻抗敏感度实验

独立盲测未通过后，停止继续微调SOE热头。复用全部**48组训练几何/132条完整 BFZI V150 S5 ultra 物理历史**构造低维物理敏感度，按三组共享热坐标（海水均温模式0、低阶模式1–31、局部海水skin模式32–47）分解已有冷态BFZI焦耳热源`src`，拟合每组的海水复阻抗温度系数。实部三个阻抗分量R11/R12/R22共9个线性系数；虚部X11/X12/X22另9个线性系数，零温升严格为零。无新神经网络头，无在线BFZI求解。实部混合与虚部混合参数分别评价。代码：`examples/bfzi_v9_phys_region.py`，实现：`sdfmpneo_bfzi/geom_weakz.py`，全部结果：`runs/geomfrac_v9_phys_region.json`。

在原15组完全互斥开发几何、41条BFZI真实物理历史的**真实热状态输入**下，原冷态弱焦耳映射海水电阻相对误差R11/R12/R22为33.4286%/15.1759%/28.5994%，区域校准后分别为**10.9077%/2.8627%/12.5940%**。新虚部物理映射的X11/X12/X22分别为**6.6317%/7.9261%/8.9920%**。这证实完整物理参考能够监督海水阻抗温度响应的空间物理系数。

用同样的未见开发几何进行**完整自主电磁热闭环**：
- 原冻结V8：动态Z16.2321%，TX R11自身15.7848%，RX R22自身20.7277%，X11自身13.9408%，X22自身15.4568%。
- 校准实部并与原电磁头混合0.5：动态Z**15.4790%**，R11**14.4391%**，R22**19.3712%**，X11 13.9394%，X22 15.4641%；五材料温度RMSE0.45197K。
- 再将校准虚部以0.25混合：动态Z15.4604%，X11 13.7438%，X22 15.0312%，TX/RX实部和温度基本保持。相对于实部校准，仅额外改善0.0187个百分点，**未达到预先设置的0.1个百分点选择门槛**；因此虚部混合参数最终保持0。
- 两组系数中的局部skin实部系数出现训练拟合边界（2.0、0.5、2.0），说明区域线性模型的局部灵敏度仍受高相关模式/源模型误差影响。

**结论：** 通过完整物理历史能够显著提高“真实温度→海水复阻抗”的离线预测准确性。闭环收益目前有限，仍受自主温度空间状态偏差与现有电磁补偿头的系统误差耦合影响。该实部区域方案仅保留开发候选，未做新盲测、未晋级V8；此前未校准的WeakSea在开发集15.2604%而V8S盲测未通过，缺少充分理由立即为较弱的15.4790%候选重新开展大批物理盲测。下一阶段应将复数物理敏感度作为原电磁温度头的结构先验/训练约束，联合优化真实热态与自主热态，优先提高未见几何上R11/R22/X11/X22的同时泛化，并用全新独立几何完成最终验收。冻结V7/V8原权重保持不变。
