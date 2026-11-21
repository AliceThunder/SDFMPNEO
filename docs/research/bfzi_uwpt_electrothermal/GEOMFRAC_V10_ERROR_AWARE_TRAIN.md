# GeomFrac V10：物理锚定的级联误差感知训练

日期：2026-11-21。主线采用原48组训练几何、132条完整BFZI V150 S5 ultra物理历史；开发集15组几何、41条历史；冷态训练额外使用原V7冷态覆盖集。原V8所有正式权重冻结，不引入新推理模块或网格法。研发原则是“模块预训练→接口误差适应→物理约束短窗口闭环→原部署闭环选模”。

## 1. 问题与训练目标

闭环中动态复阻抗误差近似为
```text
δZ ≈ (∂Zsea/∂a) δa + δZsea_model + δZcu + δZcold
```
已有V9独立诊断明确两类机制：(1) V9B0002冷态X22偏差+0.1194Ω，冷态电路电流误差10.65%，并放大铜温度；(2)其他几何主要是海水温度映射与热状态误差之间的抵消。V9旧训练把**模型预测热状态**送入电磁网络，却仍用**真实热状态阻抗**为标签，令下游学习补偿错误状态。V10修正该问题。

### A：冷态物理电路敏感度监督

保留原ColdNet双端口复阻抗9通道和可选4通道铜损输出。对各独立冷态工况、真实外电路族，在原`loss_Z + λ_I loss_I`上增加一阶电流敏感度误差：

```text
A* = Z* + Zext
I* = solve(A*, [1,0])
δI ≈ -solve(A*, (Zhat-Z*) I*)
Lsens = 1/2 mean(||δI||²/||I*||²)
      + 1/2 mean(top-quarter(||δI||²/||I*||²))
```

新增`--sens-weight`和`--init-model`，开发选模包含电流误差P90。优先让X11/X22等冷态敏感分量得到更充分监督。实现`examples/bfzi_cold_v7_train.py`，避免添加新网络层，原V8 cold checkpoint不覆盖。

首次XEnergy 80轮、最佳epoch55、同一原冷态数据集配对结果（V8→V10）：

| 指标 | 独立228开发几何 | 226冷态测试几何 |
|---|---|---|
| 冷态总复阻抗相对误差 | 0.13433→0.13083% | 0.14668→0.14329% |
| 平均电流相对误差 | 1.71040→1.63444% | 1.82225→1.74128% |
| 电流误差P90 | 3.48315→3.45261% | 3.74947→3.73625% |
| 最大电流误差 | 25.6109→23.2767% | 20.5911→20.5626% |

结果在`runs/geomfrac_v10_cold.json`；候选`runs/geomfrac_v10_cold.pt`。尚需在完整电磁热闭环确认冷态更改的影响。

### B：物理一致的电磁接口目标

使用既有完整BFZI热历史及原训练几何拟合的三片区复阻抗温度敏感度系数`runs/geomfrac_v9_phys_region.json`。

对真实热状态\(a^*\)与同时间冻结热网络产生的预测状态\(\hat a\)，预测状态的物理阻抗目标为：

```text
Z_target(g,ahat) = Z_BFZI(g,a*) +
                   η [Z_phys(g,ahat)-Z_phys(g,a*)]
```

`Z_phys`使用冷态BFZI弱式焦耳源在48个共享热坐标的投影及三区域训练拟合的实部/虚部系数，保持零温升时温度增量为零。原`examples/bfzi_v9_full_history_train.py`已新增`--phys-gain`、`--phys-weight`，从根本上避免“错误状态→真实状态阻抗标签”的错误配对，且仍用完整自主rollout开发集选模。参数、训练历史和模型保存在`runs/geomfrac_v10_phys_skin.json`与`runs/geomfrac_v10_phys_skin.pt`（成功晋级时）。

### C：短窗口可微SOE闭环

实现`sdfmpneo_bfzi/geom_window.py`和`examples/bfzi_v10_window_train.py`。

- **记忆保持：** 用完整BFZI真实时序热源推进SOE至窗口起点，前缀`no_grad`，只在窗口内反向传播；不清除分数阶历史记忆。
- **真实预测误差注入：** 初始热状态沿“真实状态→冻结V8自主热状态误差”方向逐渐增强扰动，模拟部署时上游的实际误差分布。
- **物理约束：** 48维温度状态；43维空间shape隐式传导，五材料守恒平衡迭代；双端口复数电路线性求解、海水与铜损、温度反馈通过PyTorch可微。铜绕组复阻抗保留原BFZI解析公式，在每个短窗口初态作局部切线近似以便微分，并保持零偏差锚点。
- **反向传播边界：** 固定ColdNet、SkinTanV3、ZFix、ParZ48，主要优化原`GeomFracET.heat_head`，避免电磁网络学会抵消上游热状态误差。
- **训练目标：** 48维热状态及局部skin模式误差、真实物理时间窗内的动态复阻抗增量、相同冷态基线下的电流增量、铜/海水功率增量，所有动态损失均以**同时间与同电压下的窗口初始状态**为基准，抑制模块之间的偏差转移。
- **课程训练：** 每次只保留2、4、8、16步的反向梯度计算图，逐渐增加来自冻结原V8的真实上游误差比例。选模时使用正式`geom_free.rollout`完整无梯度自主闭环，严格监测TX/RX自身误差、温度和电流。

实际一条完整BFZI历史的`examples/bfzi_v10_window_smoke.py`通过SOE前缀记忆与物理反馈的梯度检查（热网络输出头梯度范数非零、有限）。另在132训练/41开发历史执行了一次1-epoch/1-batch集成测试，训练损失有限、完整闭环输出正常；该单次更新未提升开发指标，未保存候选。16轮正式训练任务job-184继续评估。

## 2. 可复现实验

以下命令从`D:\work\geomfrac-v6`工作目录运行，并设置`PYTHONPATH=D:\work\geomfrac-v8`，使用已存在的训练数据、冻结初始权重与配置。代码可在研究分支获取，禁止通过GitHub Actions执行。

```bat
python D:\work\geomfrac-v8\examples\bfzi_cold_v7_train.py --data runs\geomfrac_cold_v7\data_merged.json --circuits runs\geomfrac_cold_ports_v1.json --init-model runs\geomfrac_cold_v8_frozen.pt --copper-weight 0.2 --sens-weight 0.25 --current-weight 0.5 --epochs 80 --batch 64 --out runs\geomfrac_v10_cold.pt --report runs\geomfrac_v10_cold.json
python D:\work\geomfrac-v8\examples\bfzi_v10_window_smoke.py
python D:\work\geomfrac-v8\examples\bfzi_v10_window_train.py --epochs 16 --updates 4 --batch 2 --val-every 4 --lr 0.00001 --out runs\geomfrac_v10_window_heat.pt --report runs\geomfrac_v10_window_heat.json
python D:\work\geomfrac-v8\examples\bfzi_v9_full_history_train.py --heat-model runs\geomfrac_et_v8_heat_tune.pt --epochs 20 --lr 0.00002 --pred-weight 0.5 --prior 0.5 --phys-weight 0.75 --phys-gain runs\geomfrac_v9_phys_region.json --val-every 5 --out runs\geomfrac_v10_phys_skin.pt --report runs\geomfrac_v10_phys_skin.json
```

## 3. 晋级门槛

候选在独立15几何、41条完整时序开发集上需要同步改善总体动态Z与主要自身阻抗精度，且电流、材料平均温度不退化。冻结候选后才用全新几何完整BFZI V150物理历史做独立验收；原V9B盲测只用于事后误差归因。每个子模块可以独立改善，正式整机晋级取决于完整闭环，不通过的候选仅保留研究结果。优先采用已有主网络与物理算子，不增加新推理残差分支。


## 4. 实际训练结果与重要实现改进（2026-11-21）

**ColdNet 冷态升级已成功形成开发候选。** `runs/geomfrac_v10_cold.pt`权重和`runs/geomfrac_v10_cold_summary.json`配对报告已提交。与原冻结V8相比，新ColdNet在226组冷态测试几何上冷态复阻抗相对误差0.14668%→0.14329%，平均电流误差1.82225%→1.74128%；与原V8热网络、SkinTanV3、ParZ48组合后，在15组开发几何、41条完整BFZI历史上的动态复阻抗16.23210%→**15.78604%**，R11自身15.78483%→15.28157%，R22自身20.72770%→20.33940%，端口电流8.61544%→**7.97102%**，材料均温RMSE0.45211→**0.43411 K**。模型所有其余网络保持冻结。

**错误接口标签修复。** `examples/bfzi_v9_full_history_train.py`已支持真实热态与自主热态使用BFZI敏感度一致的监督标签，共132历史/5088真实与预测配对状态。20轮实验`runs/geomfrac_v10_phys_skin.json`未提高开发集总体动态Z，未产生晋级权重。继续保留原SkinTanV3。

**短窗口状态分布适应。** 初版`geom_window.HeatWindow`已通过实际BFZI热源380.6W、SOE前缀、五材料可微反馈的有限非零反向梯度检查。直接仅以真实物理前缀记忆微调16轮后动态Z开发集16.23210%→17.51098%，未晋级。分析发现窗口起点自主预测状态与SOE真实前缀源不一致，因此`HeatWindow.warm`和`examples/bfzi_v10_window_train.py`已增加**同一比例融合真实/自主热状态及热源，并每隔若干轮刷新完整自主轨迹**的实现。第一轮组合新ColdNet的12轮刷新实验`runs/geomfrac_v10_window_heat.json`仍未晋级；其开发集基线为15.78604%，优化至第12轮反而16.09968%。冻结主权重未改变。

**物理敏感度指导反向梯度。** 为避免沿旧ZFix/SkinTanV3/ParZ48学习的非物理误差补偿梯度更新热网络，新增`HeatWindow.phys_z`与`--grad-mix`：前向继续使用冻结原模型海水阻抗`Znet(a)`，反向的海水阻抗雅可比按比例替换为训练集BFZI三区域复数物理敏感度`Jphys`：
```text
Ztrain(a) = (1-gamma)*Znet(a) + gamma*[stopgrad(Znet(a))+
                                     Zphys(a)-stopgrad(Zphys(a))]
```
前向数值严格等于`Znet(a)`；当`gamma=1`时反向`dZ/da=Jphys`。铜阻抗梯度仍由解析铜模型的局部温度切线给出。真实BFZI激励窗口的前向一致性、非零有限梯度已通过。物理梯度8轮阶段训练job-189运行中。

**正式独立几何计划已冻结。** `runs/geomfrac_v10_blind_plan.json`在物理计算开始前提交，V10B0001–4来自全新V7E0034/V7E0037/V7E0088/V7E0123；4组新几何×每组60s/300s两条完整BFZI V150 S5 ultra历史，共8条。物理参考生产job-188进行中。只比较训练前冻结的原V8和V10 ColdNet候选，以及先前开发集通过的其他候选；禁止基于此盲测选择额外超参数。


## 5. 2026-11-21 新增实验闭合

V10物理一致电磁训练job-185结束，20轮均未通过冻结开发集门槛，`runs/geomfrac_v10_phys_skin.json`已提交，未输出候选电磁checkpoint。

短窗口两轮开发实验：原SOE热头直接梯度优化（job-184）从冻结V8动态阻抗16.232%退化至17.511%；加入同源的SOE历史记忆/预测热状态混合、每3轮刷新自主轨迹且替换成改善后的V10冷态（job-187），开发基线15.7860%，最后一轮16.0997%，未晋级；`runs/geomfrac_v10_window_heat.json`已入库。

BFZI三区域物理复数敏感度用于训练时的反向雅可比，严格保留V8原电磁前向值（`--grad-mix=1`，job-189）8轮完成：开发基线15.7860%，epoch2 15.7273%、epoch4 15.7576%、epoch6 15.8328%、epoch8 15.9158%，RX自阻抗在early window轻微退化，按已设置的总阻抗改善门槛0.1百分点及端口约束，**无一轮晋级**。报告已提交`runs/geomfrac_v10_phys_window_heat.json`。结论是需从学习到的错误EM补偿与SOE热状态时间演化的不同响应进一步解决，即便使用更物理的反向雅可比，延长到16步后仍不能保证自主闭环改善。已经验证的冷态上游敏感度优化应保留为唯一开发候选。

**独立物理参考**：冻结于训练完成前的`runs/geomfrac_v10_blind_plan.json`，XEnergy `job-188`逐历史生成4组全新BFZI几何共8条60/300s轨迹；已完成V10B0001两条及V10B0002 60s。待完成后直接运行`examples/bfzi_v10_blind_accept.py`成对比较冻结V8及V10-ColdNet，不根据盲测选择超参数。
