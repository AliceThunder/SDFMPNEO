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

48 阶是目前最合适的共享热状态扩展：它大幅改善海水皮肤层温度场表示，且 64 阶没有额外收益。rank48 Atlas 和跨几何推理代码已经建立，神经热头训练正在进行。正式模型暂不升级。


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
