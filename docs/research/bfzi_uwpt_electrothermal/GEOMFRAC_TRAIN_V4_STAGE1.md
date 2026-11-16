# GeomFracET V4 第一批数据与训练记录

日期：2026-11-15。目标分支：`research/bfzi-uwpt-electrothermal-surrogate`。

## 已完成的物理教师

沿用原 BFZI V150、S5 五阶近源积分、ultra 网格和 1.875 s 最大热积分步长。
已恢复并核验 T02、T03 各一条 600 s、41 个观测时刻的原始电磁热耦合教师：

| 几何 | 物理状态 | 最终物理时刻 | 32 阶热基最大投影误差 | 物理建模与时程耗时 |
|---|---|---:|---:|---:|
| T02 | candidate | 600 s | 0.2682 K | 150.0 s |
| T03 | candidate | 600 s | 1.4152 K | 157.4 s |

原始文件见 `runs/bfzi_s5_T02_ultra_train600_*.npz` 和
`runs/bfzi_s5_T03_ultra_train600_*.npz`。
当前 V4 训练集 T01/T02/T03 共九条教师历史，包括三条 600 s 历史；
T04 仍是完整几何开发验证集，冻结 U01–U12 保持独立。

## 数据扩展与训练程序

新增 `examples/bfzi_s5_v4_plan.py` 和 `runs/geomfrac_s5_v4_plan.json`：
固定随机种子 1504，划分 30 个训练几何、5 个开发验证几何、
5 个最终独立验收几何。新几何覆盖连续圆形—圆角方形参数、
空间位姿、封装尺寸及五个温度和热交换输入。
同一几何包含连续通电、阶跃/间歇通电三种激励，历时 60/300/600 s。

`examples/bfzi_s5_buildset.py` 已增加按几何选择和按激励定义时程长度的功能，
使用原 V150 物理直接法，没有更换电磁或热求解器。
`examples/bfzi_geom_v4.py` 根据真实教师文件生成无几何交叉的 3/5/10/20/30
规模训练清单，T04 保持在验证集。
首批新增 N01、N02 的物理教师生成已启动；3 几何 V4 对照训练已启动。
后续训练、验收数值以保存的报告为准，未完成的实验不计入结果。

## 执行入口

```powershell
python -m examples.bfzi_s5_v4_plan
python -m examples.bfzi_s5_buildset --bfzi-root D:\work\geomfrac-s5-BFZI --plan runs/geomfrac_s5_v4_plan.json --inner-step 1.875 --cases N01 N02 --tag v4
python -m examples.bfzi_geom_v4 --train 3 --val 1
python -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_v4_3g_manifest.json --init-model runs/geomfrac_et_s5_long_v3.pt --epochs 500 --patience 70 --lr .0002 --weight-decay .0001 --lr-patience 20 --lr-factor .5 --closed-warmup 10 --closed-ramp 40 --closed-weight .08 --therm-weight .05 --history-weight step --zrel-weight 1 --soe-quad trap --heat-dt 1.875 --out runs/geomfrac_et_v4_3g.pt --report runs/geomfrac_et_v4_3g.json
```

[新增物理结果] N01 已完成 60/300/600s 三条 V150 S5 教师；600s 41 点全场最高温度 312.892 K。N02 已进入物理数据生成。数据位于 runs/bfzi_s5_N01_ultra_v4_*.npz。

## V4 三几何真实训练完成

九条 S5 教师（T01/T02/T03 各 2 条 60s + 1 条 600s），T04 整几何留出三条。
AdamW、初始学习率 2e-4、耐心 70、最长 500 轮，实际 202 轮，
最佳第 132 轮，最佳验证损失 0.461930。
模型权重 runs/geomfrac_et_v4_3g.pt；报告 runs/geomfrac_et_v4_3g.json。

| T04 动态完整端口 Z 相对 L2 | V3 | V4 3几何 |
|---|---:|---:|
| 60s drive1 | 14.149% | 19.325% |
| 60s drive2 | 7.922% | 12.489% |
| 600s drive1 | 25.458% | 15.508% |

T04 600s 三维温度最大误差 1.667K，铜温最大误差 0.395K，
封装温度最大误差 1.667K，五材料预算最大残差 4.56402915460785E-10W。
600s 动态 Z 改善，短时程误差需在五几何及后续增量训练中降低。

## ������ V4 ��ɽ��
ѵ�� 202 �֣���ѵ� 132 �֡�T04 �����ջ���̬���迹��60s ����1 19.325%��60s ����2 12.489%��600s ����1 15.508%��V3 ��ӦΪ 14.149%��7.922%��25.458%��600s ������Լ 39.1%��60s ��������Ż���ģ�ͼ� runs/geomfrac_et_v4_3g.pt��ָ��� runs/geomfrac_et_v4_3g.json���������ļ��ε���ѵ�����弸��ѵ���ȴ� N02 ��ɡ�

## 新增五几何首批真实教师齐备

N01/N02 各完成 60s、300s、600s 三条原 BFZI V150 S5 物理时程，
五阶源积分、ultra 原网格、1.875s 热步保持不变，六条均为 candidate。
两个几何最大 32D 热基投影误差分别为 3.2145 K、
2.1822 K。
原始文件及逐历史电热守恒和热基误差见 runs/geomfrac_s5_v4_build_report.json。
五几何代理训练清单采用 T01/T02/T03/N01/N02，T04 保持整几何留出。

追加三几何已保存权重的独立推理计时：T04 60s 两条 CPU
0.0553s / 0.0607s，600s 0.3929s。600s 动态 Z 分段误差：
0–60s 19.325%，60–300s 9.868%，300–600s 16.802%。

V4 四几何增量对照（T01/T02/T03/N01，T04 整几何留出）：真实训练135轮，最佳第65轮；T04动态Z，60s两激励19.901%/20.609%，600s 17.755%。模型runs/geomfrac_et_v4_4g.pt和结果runs/geomfrac_et_v4_4g.json。

## V4 五几何真实训练完成

训练几何 T01/T02/T03/N01/N02，共15条真实 S5 教师，T04整几何留出。
训练运行 225 轮，最佳第 155 轮，验证损失 0.269888。
采用 AdamW、分数阶 SOE、32维共享热状态、每轮抽6条教师、每5轮验证。
最佳权重和完整报告：runs/geomfrac_et_v4_5g.pt / runs/geomfrac_et_v4_5g.json。

| T04 动态复阻抗误差 | V3 | V4 三几何 | V4 五几何 |
|---|---:|---:|---:|
| 60s drive1 | 14.149% | 19.325% | 20.796% |
| 60s drive2 | 7.922% | 12.489% | 18.478% |
| 600s drive1 | 25.458% | 15.508% | 16.184% |

600s 三维最高温度误差 0.9825 K，铜温
0.2914 K，封装温度
0.9825 K，材料预算残差 6.37101038591936E-10 W，
CPU 推理耗时 0.5585s。

V4 五几何动态Z损失权重对照：同训练数据、同三几何初始化、同AdamW配置。zrel=3，最佳轮次50，T04 60s1/60s2/600s误差20.074/18.540/15.948%；zrel=0，最佳轮次100，对应19.393/17.300/15.813%。原 zrel=1 数据见 geomfrac_et_v4_5g.json。

## V4 eight-geometry cohort and geometry repair (2026-11-15)

The original BFZI V150 S5 teachers for N03/N04/N05 are complete, with
three excitation histories of 60/300/600 seconds per geometry. The
eight-geometry training cohort T01/T02/T03/N01-N05 now contains
24 real physical histories. Its manifest is
`runs/geomfrac_s5_v4_8g_manifest.json`.

The 32-mode thermal projection maximum error for the 600s histories
is 1.514 K (N03), 2.251 K (N04), and 9.528 K (N05). N05's
package-region maximum error is 1.110 K. These are physical teacher
projection statistics, not neural inference results.

The 5g baseline held-out T04 dynamic impedance relative L2 errors were
20.796%/18.478%/16.184% for 60s-1/60s-2/600s. The zrel=0
ablation improved these to 19.393%/17.300%/15.813%, so the
8g training was initialized from the 5g zrel=0 checkpoint using the
same 2e-4 learning rate and 0.08/0.05 closed-loop/thermal weights.

The deterministic V4 geometry sampler previously generated
N06 alpha1=0.743, core1=12 mm: the ten-turn M4 coil needs at least
12.85 mm for the inner straight-core constraint. The sampler now
enforces a 1 mm reserve, setting N06 core1 to 14 mm. All existing
T01-T04 and N01-N05 case definitions remain unchanged. V4 plan tests
pass (2/2). N06/N07/D01 S5 teachers and the 8g training are running;
their numerical results must be recorded after completion.

## V4 eight-geometry first result and reproducible repair (2026-11-15)

The 8g training on T01/T02/T03/N01-N05 (24 physics histories)
completed in 120 epochs; best checkpoint at epoch 50.
Held-out T04 dynamic impedance relative L2 errors were 51.076%,
54.780% and 39.681% for 60s-1, 60s-2 and 600s-1. This model does not
improve accuracy over the 5g baseline (19.393%/17.300%/15.813%).

The 8g fit reused 5g neural weights but recomputed all normalization
scales: the first 6 impedance-output scales increased by factors
1.94 to 3.10 after adding N03-N05. Thus the inherited weights were
applied in a changed coordinate system, preventing a controlled
warm-start comparison. Training now supports `--reuse-scales`, which
retains the prior checkpoint's source, output and thermal-mode scales.
The 8g v2 model is being trained from 5g zrel=0 with these scales,
dynamic-Z loss weight 1, learning rate 1e-4 and the original
closed-loop/thermal coefficients 0.08/0.05.

N06 completed its 60/300/600s physical teachers, all status candidate.
Its 600s maximum 32-mode projection error was 8.066 K.
N07 was initially rejected by the native BFZI minimum inner-bend
constraint. BFZI computes its effective turn pitch as
(outer_radius-inner_radius)/turns. The V4 sampler now imposes both
the straight-core lower bound and a native inner-bend upper bound,
with a 1mm clearance margin. Only N07, N19 and N24 generated
geometry definitions changed; T01-T04/N01-N06 remain unmodified.
N07 and D01 passed the native geometry mesh preflight; their
600s S5 teacher generation has been relaunched.
