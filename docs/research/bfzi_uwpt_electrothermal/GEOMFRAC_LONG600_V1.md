# GeomFracET 真实 S5 600 秒时程与空间模态稳定性

日期：2026-11-15。源代码：`AliceThunder/SDFMPNEO` 分支 `research/bfzi-uwpt-electrothermal-surrogate`。物理教师：原 BFZI V150 分支 `bfzi-v150-turn-scan`。计算平台：XEnergy。

## 1. 新增原 BFZI 五阶 S5 长时程教师

在先前 T01–T04 八条 60 s 短历史之外，新产生两条 600 s、15 s 输出（41 个时刻）、热积分上限 1.875 s、海水八层薄层的物理闭环时程：

| 教师 | 用途 | 电压 | 通断电 | 物理状态 | 单几何建模及长时程耗时 | 32D 热基空间误差上限 |
|---|---|---:|---|---|---:|---:|
| T01 | 训练 | 65 V | 30–180 s、210–360 s、390–600 s 断电 | candidate | 182.1 s | 4.325 K |
| T04 | 整几何留出 | 75 V | 30–180 s、210–360 s、390–600 s 断电 | candidate | 153.8 s | 0.779 K |

冷态热锚、完整物理教师和生成报告存于 `runs/bfzi_s5_{T01,T04}_ultra_long600_*.npz` 和 `runs/geomfrac_s5_long600_build_report.json`。冻结 U01–U12 未用于任何网络训练。

## 2. 发现长期空间热模态非物理漂移

对仅用 60 s 历史训练的 GeomFracET，T04 600 s 物理参考最高温仅 298.54 K。旧网络预测的局部温度最低 192.69 K，600 s 末最高 366.03 K，产生明显非物理空间温差。五材料总体热预算残差仍约 1e-9 W；预算守恒只约束五个材料加权平均温度，而 27 个空间温差模态此前允许无界积累。

## 3. 保守的空间模态可行域投影

在现有热基上写成 `theta=Phi_mean*a_mean+Phi_dev*a_dev`，前 5 个模式对应五材料容量均温，后 27 个描述材料内部空间温差。热基容量正交，因此对任意常数 `beta`，只将后 27 个模态缩放成 `beta*a_dev`，保持五种材料的加权平均温度**完全不变**。

设各有限体积节点的均温部分 `base_i`、空间偏差 `delta_i`，选择：

```text
beta = min(1, min_{delta_i<0} max(base_i,0)/(-delta_i))
a_dev <- 0.999 * beta * a_dev  (beta<1 时)
```

使热输入仅来自非负 Joule 损耗、初温与海水环境温度相同的热场满足 `theta_i >= 0`。每次调整高模态后再次执行五材料 `5×5` 守恒求解；最多五次。无需在线 BFZI 电磁方向求解或 32D 隐式热 ROM 求解。

同时将相同的材料校正与非负热场投影加入可微分闭环训练 `_heat_balance`，训练在校正后的状态上计算真实封装和铜温误差。新增单测检查非负性及材料加权均温不变性，原有相关测试保持通过。

## 4. 真实 600 s 自主推理前后对照（固定原权重，仅修改物理投影）

| 指标 | T01 原网络 | T01 加投影 | T04 原网络 | T04 加投影 |
|---|---:|---:|---:|---:|
| 动态 Z 相对 L2 | 32.895% | **25.638%** | 280.925% | **29.830%** |
| 三维温度最大误差 (K) | 82.088 | **7.414** | 102.370 | **1.460** |
| 铜温最大误差 (K) | 22.580 | **2.157** | 30.147 | **0.838** |
| 封装最大误差 (K) | 82.088 | **7.414** | 102.370 | **1.405** |

投影后的 T04 全场最低温度为 293.15 K，最高为 297.76 K；五材料预算残差最大 4.952e-9 W。仅加入物理可行域投影就显著缓解了长期漂移，证明原有长期失效的重要原因是高阶空间模态缺少物理约束。

## 5. 真实长时程训练与留出方案

在保持 GeomFracET 45 维几何/热输入、32D 热状态与分数阶 SOE 主干不变的情况下，训练清单 `runs/geomfrac_s5_long_manifest.json`：

- T01：2 条 60 s 及 1 条 600 s 真实历史；
- T02/T03：各 2 条 60 s 真实历史；
- T04：2 条 60 s 和 1 条 600 s 历史，**整几何留出，只用于模型选择和误差评估**。

因此总计 7 条训练历史、3 条留出历史。为控制长序列训练成本，`examples/bfzi_geom_multi.py` 新增 `--closed-warmup`、`--closed-ramp`，可自定义教师监督到可微分自主闭环功率的渐进切换，默认行为保持 20/60 轮。

600 s 训练命令：

```powershell
python -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_long_manifest.json --init-model runs/geomfrac_et_s5_soe.pt --soe-quad trap --heat-dt 1.875 --epochs 140 --patience 45 --closed-warmup 10 --closed-ramp 25 --lr .0002 --closed-weight .08 --therm-weight .05 --out runs/geomfrac_et_s5_long.pt --report runs/geomfrac_et_s5_long.json
```

新增 `tests/test_bfzi_geom_canon.py` 的热模态可行域测试，且原有 SOE／S5 物理教师／跨几何训练相关测试通过。

**验收范围**：本阶段先验证非冻结 T01/T04 的长时程热场稳定性与跨几何可用性。完整 T01–T04 八条 600 s 时程、多工况覆盖及冻结 U01–U12 的神经独立验收仍需继续。


## 6. 2026-11-15 续推：长短时程混合训练损失的时间权重

发现此前 `score()` 对每条教师历史分别计算时间平均损失，再直接对历史取等权平均。7 条训练历史分别包含 6×4=24 个短历史时间区间、1×40=40 个长历史时间区间；因此真实 600 s 训练只得到 1/7（14.29%）的总权重，虽然它占 40/64（62.50%）的观测区间。T04 留出历史中，600 s 数据也仅占 1/3 权重，而包含 40/48（83.33%）的观测区间。原长时记忆、冷却段在模型选择中受到低权重。

新增 `score(...,hist_weight="step")`，把每条历史内原损失保持不变，以该历史时间区间数量加权汇总。跨几何训练脚本新增 `--history-weight {equal,step}`，默认选 `step`；旧等权结果可以显式使用 `equal` 复现。教师与留出长短历史一并采用同一种权重，热源、热预算、空间基与 SOE 计算均不改变。

同时把教师历史和锚路径里的 Windows 反斜杠规范化为路径分隔符，使已提交的混合时程 manifest 可在 Linux/Python 沙箱使用。新增历史权重和路径转换测试；沙箱两项独立局部检查通过。

推荐在真实 S5 数据齐全的计算环境重训，并比较 600 s T04、两条 60 s T04 的动态阻抗相对 L2、温度场最大误差及五材料能量预算。可直接在原长时训练命令中附加 `--history-weight step --zrel-weight .2`，同时将 `--out` 和 `--report` 维持同一个训练版本以方便前后比较。建议把 `--zrel-weight 0` 作为仅改变时间权重的独立对照。

**本轮已交付代码及局部检查，尚未重跑完整 BFZI S5 600 秒神经训练。** 因此该权重改动对留出精度的实际提升仍需真实训练报告确认。


## 7. 2026-11-15 续推：动态阻抗训练目标与物理验收统一

进一步检查发现，旧的可选 `--zrel-weight` 只计算六维**海水**动态阻抗相对误差；模型最终验收的 `dynamic_Z_rel_L2` 则比较**海水加双铜绕组**的完整端口阻抗。这会让神经热头影响的铜阻抗变化只间接受温度损失监督，无法直接通过相对端口阻抗损失调整。

现在保留原有六维海水监督，增加与验收量对应的可微分完整动态阻抗损失：

```text
Z_pred(t)-Z0 = ΔZ_sea_pred(t) + diag(dZcu/dT|ambient * ΔTcu_pred(t))
L_zrel = Σ_{t>0} || pack(Z_pred(t)-Z0) - (z6_true(t)-z6_true(0)) ||_2^2
         / max(Σ_{t>0} || z6_true(t)-z6_true(0) ||_2^2, 1e-12)
```

其中双铜阻抗的温度切线复用原物理模块 `PortModel.copper()` 在环境温度附近的中心差分，与现有可微分自主热源使用同一近似。训练梯度能够同时作用于海水阻抗预测和 32D 热状态。现有保存模型的结构、SOE、空间热基和在线 BFZI 原始铜阻抗求解方式均不改变；`--zrel-weight 0` 继续对应此前未启用相对阻抗损失的行为。

新增单测 `test_dynamic_z_loss_matches_full_copper_and_seawater` 检验：教师完整阻抗同时包含指定海水变化与铜温变化时，目标损失为零；忽略铜温时产生非零误差，并正确向铜温变量反向传播。该核心计算已在 Python/PyTorch CPU 沙箱独立运行并通过数值与梯度检查。

推荐的下一次真实 S5 长时程跨几何训练采用同一冻结划分（T01–T03 训练，T04 全留出），单一新版本输出：

```powershell
python -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_long_manifest.json --init-model runs/geomfrac_et_s5_soe.pt --soe-quad trap --heat-dt 1.875 --epochs 140 --patience 45 --closed-warmup 10 --closed-ramp 25 --lr .0002 --closed-weight .08 --therm-weight .05 --history-weight step --zrel-weight 1 --out runs/geomfrac_et_s5_long_v2.pt --report runs/geomfrac_et_s5_long_v2.json
```

验收时与现有 `runs/geomfrac_et_s5_long.json`（T04 60s：21.59%、17.49%；600s：27.98% 动态 Z 相对 L2）逐历史比较，关注温度最大误差、五材料预算是否同时维持；再以独立全新 600s 工况进行最终泛化检查。当前仅完成代码、测试及方案更新，真实 V2 训练尚未执行，也尚无 V2 精度数值。
