# 跨几何共享热状态扩展至 48 阶 V1

日期：2026-11-19

## 1. 研究动机

六组新独立动态验收显示 ColdRes 已改善动态阻抗，但 C0260 和 C0725 仍存在明显动态误差。对同一批 12 条 BFZI V150 动态物理历史进行 canonical 热场投影后发现，32 阶热坐标遗漏了海水皮肤层的局部空间结构。该结构是温度变化影响海水阻抗的关键区域。

本阶段将原有 32 阶共享热状态扩展为 48 阶：保留五个材料均值和原 27 个空间模式，新增 16 个 TX/RX 海水皮肤层局部模式。64 阶仅作为对照。

## 2. 物理参考投影证据

在 C0441、C0043、C0416、C0027、C0725、C0260 六组独立几何的 12 条 60s/300s BFZI V150 物理温度历史上，使用相同 canonical 基函数投影：

| 指标 | 32 阶 | 48 阶 | 64 阶 |
|---|---:|---:|---:|
| 平均全场 RMSE | 0.3384 K | **0.1239 K** | 0.1238 K |
| 最大全场误差 | 9.3129 K | **4.5817 K** | 4.5822 K |
| 平均海水皮肤层 RMSE | 0.6421 K | **0.2198 K** | 0.2194 K |

48 阶将平均全场投影 RMSE 降低 63.4%，海水皮肤层 RMSE 降低 65.8%。64 阶没有带来可观收益。原始结果见 `runs/geomfrac_cold_dyn6_basis.json`。

## 3. rank48 跨几何热算子 Atlas

使用原有 36 组训练几何和 11 组开发几何重建 rank48 Atlas：

- 主几何核宽度 `ell=6.0`，正则化 `lambda=0.003`
- 热算子核宽度 `kell=0.6`
- 热算子容量尺度指数 `[0.5, 0.25, 1.0]`

rank48 开发集热算子 Frobenius 相对误差为 3.8050%，平均最小特征值为 (2.3007\times10^{-5})，保持对称正定。rank32 对应误差为 3.7915%，平均最小特征值为 (2.3127\times10^{-5})。扩展阶数没有破坏共享热算子插值质量。

Atlas：`runs/geomfrac_init_pose42_kell_36g_r48_v1.npz`。报告：`runs/geomfrac_init_pose42_kell_36g_r48_v1_dev.json`。

## 4. 软件实现

- `geom_free.rollout` 的状态、容量缩放、隐式热步和守恒校正按 Atlas rank 运行，支持 32/48/64 阶。
- `geom_free_train` 的可微热步矩阵、材料尺度和状态初始化按算子维度运行。
- `GeomInit.balance_cu` 的铜—封装守恒校正支持 32/48/64 阶。
- rank32 `SrcFix` 更新 rank48 Atlas 的前 32 个嵌套 canonical 模式，新增 16 个模式保留 Atlas 物理源项。
- rank32 `ZFix` 继续作用于原 32 个全局热状态；新增 `geom_skintan.restore_rank48`，将 rank48 状态的后 16 个皮肤层坐标直接送入 TX/RX 分端口阻抗切线。
- 训练和开发评估入口按 checkpoint rank 读取教师历史。

canonical rank48 的前 32 个基函数与 rank32 完全一致，新增模式只附加在原有模式之后，因此旧冷态源项和基础阻抗切线可以按嵌套前缀继续使用。

## 5. 测试

XEnergy 上相关测试：

- rank48 自主神经热演化、隐式 ROM、能量预算、Atlas 铜—封装校正及 rank32 源项扩展：18 项通过。
- ZFix rank48 前缀兼容、SkinTan 直接 rank48 皮肤状态适配：10 项通过。
- 合计 28 项相关测试通过，修改文件通过 `py_compile`。

## 6. rank48 神经热头训练

当前 XEnergy 训练任务：

- 36 组训练几何、108 条动态历史
- 11 组开发几何、33 条动态历史
- 从冻结 V5 rank32 checkpoint 初始化并扩展至 rank48
- 只训练 `heat_head`，保持 SOE 电磁记忆主体不变
- 使用动态阻抗闭环开发指标选模
- AdamW，初始学习率 (3\times10^{-4})，每批 12 条历史，最多 120 epoch，patience 40

训练完成后，先在原开发集检查相对初始化的动态阻抗改善，再用冻结六组独立动态物理参考验收。随后比较原 ZFix 与直接 rank48 皮肤层切线，检验新增 16 个模式能否将温度场改进传递到海水动态阻抗。

## 7. 验收与升级规则

1. rank48 神经模型必须在相同 11 组开发集上优于 rank48 初始化模型，最小动态阻抗改善沿用 0.1 个百分点的选模门槛。
2. 六组独立动态物理参考用于最终比较，报告每组、每条激励和整体动态阻抗误差。
3. 同时报告五材料平均温度误差、海水功率误差、铜耗误差和能量预算残差。
4. 新增皮肤层切线必须在冻结六组上改善整体动态阻抗，并避免 C0260/C0725 的显著退化。
5. 候选未达到独立验收要求时，正式模型继续保留 V5。

## 当前结论

48 阶是目前最合适的共享热状态扩展：它大幅改善海水皮肤层温度场表示，且 64 阶没有额外收益。rank48 Atlas、神经热头和直接皮肤状态阻抗切线均已完成开发训练；当前进行阻抗切线驱动的热头联合微调。冻结六组独立验收通过前，正式模型继续保留 V5。


## 8. Training selection correction — 2026-11-19

The first rank48 heat-head run used `free_dynamic_z` selection with the frozen rank32 `ZFix` tangent. That tangent consumes only the first 32 global coordinates, so the appended 16 skin coordinates had no direct path into the selected impedance metric. The run completed 70 epochs but retained its initial checkpoint: 16.4181% initial and best dynamic-Z selection error, 0% improvement.

The corrected heat-head run uses `--select thermal`, which evaluates the reconstructed rank48 temperature field and directly measures whether the new skin coordinates are learned. At epoch 50, the maximum full-field temperature error on the 11-geometry development set improved from 2.3884 K to 1.9647 K (17.7% reduction); epoch 50 was the current best at the last progress update.

A rank48 direct-skin impedance tangent trainer is now available as `examples/bfzi_geom_rank48_skintan.py`. It fits the port-factorized TX/mutual/RX correction against the appended 16 predicted rank48 states and evaluates it in the same closed geometry-free loop. `examples/bfzi_geom_multi.py` now accepts `--skin-tangent-model` and can use the same tangent for training loss and free-loop checkpoint selection. After the thermal run finishes, train this tangent on the 36G/11G split, then fine-tune the thermal head with its tangent loss and compare against the frozen six-case holdout.


## 9. Rank48 thermal-head result — 2026-11-19

The corrected thermal-selection run completed successfully:

- Selected checkpoint: `runs/geomfrac_et_pose42_kell_r48_thermal_v1.pt`
- Best epoch: 50; early stop after 90 epochs.
- Development selection metric: maximum full-field temperature error averaged over 33 histories improved from 2.3884 K to 1.9647 K (17.74% reduction).
- Worst individual development-history maximum temperature error: 7.6799 K.
- Mean autonomous dynamic-Z relative error on the same 33 teacher histories: 16.3031%, compared with 15.9086% for the untrained rank48 initialization. Thermal-field improvement alone did not improve this impedance metric.

The next task is fitting the direct rank48 skin tangent against the 36G/11G split. The trainer uses the first 32 global states and appended 16 TX/RX skin coordinates from the predicted rank48 thermal rollout. Current process: XEnergy `job-35`.


## 10. Direct rank48 skin tangent result and joint fine-tuning — 2026-11-19

The rank48 direct-skin tangent fit completed on the 36G/11G split (108 train and 33 development histories). It uses the predicted first 32 global heat coordinates plus the appended 16 TX/RX seawater-skin coordinates. The geometry-free V4 port-factorized tangent reached its best development result at epoch 180:

- Frozen ZFix baseline: 17.1687% mean dynamic-Z relative error.
- Rank48 direct-skin tangent: 16.2292%.
- Relative improvement: 5.47% (0.9395 percentage points).
- Mean current relative error: 7.8807% versus 7.8755% baseline.
- Mean material temperature RMSE: 0.8230 K; mean maximum temperature error: 2.1326 K.
- Maximum material energy-budget residual: 2.38e-11 W.

The tangent therefore provides a measurable but still moderate impedance improvement. The model is stored at `runs/geomfrac_rank48_skin_tan_v1.pt`; its development report is `runs/geomfrac_rank48_skin_tan_v1.json`.

A rank48 heat-head fine-tune has now started using this tangent in both the differentiable tangent loss and the free geometry closed-loop selection. Only the appended 16 skin-mode output rows are trainable; the first 32 global thermal outputs and the SOE impedance branch remain frozen. It uses the same 36G/11G development split and does not access the sealed six-case holdout. The free-loop initialization score is 16.2292%, matching the selected tangent development result. Current XEnergy process: `job-40`.

The tangent restore path and multi-geometry training integration pass the focused tests (8 passed). After fine-tuning, compare the best checkpoint with the frozen six-case holdout and the V5 baseline before deciding whether rank48 can replace V5.

## 10. Geometry-conditioned V3 tangent comparison — 2026-11-19

A geometry-conditioned V3 tangent was trained with the same 36-geometry/108-history training split and 11-geometry/33-history development split used by V4. V3 uses compact physical descriptors (coil geometry, relative translation, package dimensions and thermal parameters) together with the 32 global and 16 appended skin states. TX self, mutual and RX corrections remain separate.

| Metric | V4 geometry-free | V3 geometry-conditioned |
|---|---:|---:|
| Best development dynamic-Z error | 16.2292% | **16.1513%** |
| Current error | 7.8807% | 7.8813% |
| Temperature RMSE | 0.8230 K | 0.8232 K |
| Maximum temperature error | 2.1326 K | 2.1337 K |
| Maximum energy-budget residual | 2.38e-11 W | 2.37e-11 W |

V3 reached its best development dynamic-Z error at epoch 120, reducing the common 17.1687% baseline by 5.93%. It is the preferred tangent for the next heat-head fine-tuning candidate. The V4 tangent fine-tune (job-40) is already running and will be retained as a comparison; the V3 fine-tune will use the same train/dev split, tangent loss and BFZI-free development selection. The frozen six-case/12-history dynamic set remains reserved for final independent evaluation.

Report: runs/geomfrac_rank48_skin_tan_v3.json; checkpoint: runs/geomfrac_rank48_skin_tan_v3.pt on XEnergy.


Implementation update: the BFZI-free evaluation entry now dispatches rank48 checkpoints to restore_rank48, while retaining the original rank32 time-conditioned adapter. The heat-head loss reuses the predicted tangent and batches teacher tangent targets to reduce repeated forward calls. Validation: 18 neural/multi/skintan tests passed; 11 free-evaluation/skintan tests passed.


## 11. Skin-output-only feedback fine-tuning — 2026-11-19

The first tangent-driven thermal fine-tune completed 70 epochs with the first 32 global heat outputs frozen and only the appended 16 skin-mode output rows trainable. Its free geometry closed-loop development selection remained at the initialization value of 16.2292% (best epoch 0; 0% improvement). The model-selection rule retained the original rank48 thermal checkpoint. This indicates that adapting only the appended skin outputs is too restrictive for the coupled impedance objective.

The next run unfreezes the complete thermal head while keeping the SOE impedance branch frozen. It retains the same rank48 skin tangent, thermal physics loss, full dynamic-Z loss and free-loop development selection. Current XEnergy process: `job-41`; output candidate: `runs/geomfrac_et_pose42_kell_r48_tangent_v2.pt`. The frozen six-case holdout remains untouched.

## 11. Heat-head fine-tuning with direct skin tangent

The V4 tangent fine-tune (job-40) trained only the appended 16 skin-mode output rows, with tangent-loss weight 1.0 and BFZI-free dynamic-Z selection. It completed 70 epochs; the best checkpoint remained epoch 0 at 16.2292% development dynamic-Z error, while the epoch-50 selection was 16.3941%. The trained checkpoint therefore retains the pre-finetune heat-head weights. This indicates that updating only the final 16 output rows is too restrictive for the current shared heat representation.

A one-epoch smoke test (job-43) passed with the V3 tangent and confirmed an initial BFZI-free development score of 0.161513. The full run (job-45) uses the better V3 geometry-conditioned tangent and trains the complete heat_head, while keeping the SOE dynamics frozen. It uses the same 36G/11G split, tangent loss 1.0, full dynamic-Z loss 0.5, thermal loss 0.03, closed-loop source warmup/ramp 10/20, batch size 12 and BFZI-free development selection. Learning rate is reduced to 1e-4. The training loss now reuses predicted tangent outputs and batches teacher-state tangent targets.

After job-42, the selected checkpoint will be evaluated on the frozen six-case/12-history dynamic set against the official V5 and the pre-finetune rank48 model. V5 remains the official model until the independent comparison passes.
