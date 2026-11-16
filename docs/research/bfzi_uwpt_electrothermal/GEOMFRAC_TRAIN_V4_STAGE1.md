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

## Independent checkpoint evaluation

`examples/bfzi_geom_eval.py` now evaluates any checkpoint on an
explicit train/val/test manifest split without retraining or changing
normalization. This permits D01 and future E01-E05 geometries to be
evaluated as autonomous physics-coupled, never-trained cases.

Example:
```powershell
python -m examples.bfzi_geom_eval --model runs/geomfrac_et_v4_8g.pt --manifest runs/geomfrac_s5_v4_8g_manifest.json --split val --out runs/geomfrac_et_v4_8g_eval.json
```

The 8g checkpoint evaluated on T04 reproduces the original three
dynamic impedance errors exactly (maximum difference 0.0).

## Eight-geometry independent-validation comparison (2026-11-15)

T04 and D01 are disjoint development geometries, never included in any of these training sets.

| Model | T04 60a | T04 60b | T04 600 | D01 60 | D01 300 | D01 600 |
|---|---:|---:|---:|---:|---:|---:|
| 5g | 19.393% | 17.300% | 15.813% | 16.407% | 14.084% | 18.172% |
| 8g | 51.076% | 54.780% | 39.681% | 29.017% | 30.039% | 32.303% |
| 8g_v2 | 25.844% | 24.237% | 18.104% | 19.095% | 17.104% | 20.555% |

The 8g v2 correction retained the 5g normalization and trained 145 epochs, best at epoch 55;
it improved over the original 8g, but 5g remained better on all six held-out histories.
The first larger-cohort experiments optimized a composite teacher/thermal objective;
checkpoint selection was not aligned with autonomous dynamic-Z relative L2.

New training option `--select dynamic_z` selects the checkpoint by mean
autonomous dynamic-Z relative L2 over all validation histories, with early stopping
and the learning-rate scheduler using this same metric. With a warm start, epoch 0
is a retained baseline checkpoint. The 10g experiment uses T01-T03/N01-N07,
30 teacher histories, and T04/D01 for development validation. The final
E01-E05 geometry cohort remains reserved for final independent testing.

New physical teachers N07, N08, N09 and independent D01 are completed, all
three excitation histories per geometry (60/300/600 s). Their 600s maximum
32-mode spatial projection errors are: N07=10.812 K, D01=1.499 K, N08=4.665 K, N09=2.255 K.
N10-N12 physical data generation continues independently.

## Controlled spatial-projection cohort

The long-time 32-mode temperature projection errors are up to
9.528 K (N05), 8.066 K (N06), and 10.810 K (N07). A controlled
8-geometry training manifest uses T01/T02/T03/N01/N02/N03/N04/N09
and the same T04/D01 development validation. Its purpose is to
test whether high thermal-ROM projection error in N05-N07 is
driving the poor large-cohort generalization. The neural architecture,
warm-start checkpoint, fixed scales, training loss weights, and
autonomous model-selection criterion match the 10g experiment.
Results from this cohort must be compared with both the 10g model
and the 5g baseline; the final E01-E05 test geometries remain held out.

N10 has completed its 60/300/600 s real BFZI S5 teacher histories; the 600 s 32-mode maximum temperature projection error is 0.800 K.

## V4 complete physics-cohort generation (2026-11-16)

All 36 new planned geometries (N01-N27, D01-D04 and E01-E05)
have completed BFZI V150 S5 physical histories of 60s, 300s and
600s (108 new histories), and are marked candidate by the physical
teacher. Together with T01-T04, all planned 40 geometry IDs now
have teacher data: 30 training, 5 development, 5 strictly held-out
final testing. Source JSON: `runs/geomfrac_s5_v4_teacher_summary.json`;
full physical run metrics: `runs/geomfrac_s5_v4_build_report.json`.
The 32-mode heat basis has geometry-dependent spatial projection
limits, including N26 20.502 K, N24 14.477 K and N07 10.812 K
on 600s histories. These are projection limitations, not neural
temperature errors. Training on these cases requires careful
independent evaluation.

Five-development-geometry (T04 and D01-D04) autonomous dynamic-Z
relative L2 mean across 15 trajectories:
5g model: 16.870%; 8g low-projection cohort: 16.315%.
The 10g model with dynamic-Z checkpoint selection retained the
5g initialization, while the low-projection 8g model improved
mean dynamic-Z on the first two development geometries.
A 30g model has been launched, initialized from the low-projection
8g checkpoint and selecting by all five development geometries.
Final E01-E05 histories remain unused in training/model selection.

## Thirty-geometry projection-aware sampling experiment

The all-geometry baseline trains the entire 30-geometry/90-history
training set with the same 8g low-projection warm start, fixed scales,
closed-loop loss and dynamic-Z selection over all five development
geometries. A matched controlled experiment sets `--proj-tau 5`.
Each teacher history is sampled into the training mini-batch with
weight `1/(1+(projection_max_K/5)^2)`. This reduces the influence
of 600s thermal histories whose true fields cannot be accurately
represented by the fixed canonical 32-mode basis, while preserving
every geometry in the available training pool. The validation metric
always uses every T04/D01-D04 history with equal importance; E01-E05
are not examined until the final checkpoint selection is frozen.
The experiment does not increase thermal ROM rank or change neural
architecture; its purpose is to test data-quality weighting as an
interim solution. The baseline and weighted outcomes must be
reported separately.

## Dynamic-impedance error source probe

On six T04/D01 physics histories, the 8g-quality checkpoint was
evaluated by feeding the teacher's true source inputs and true
previous-time thermal coordinates into the neural SOE/impedance
predictor, and comparing its seawater dynamic impedance prediction
against the teacher's seawater impedance, normalized by the total
port dynamic impedance magnitude. The teacher-forced result is
16.660% mean error; autonomous closed-loop error on the same histories is 16.385%.
Because oracle teacher inputs alone do not remove the dynamic-Z error,
the learned seawater impedance mapping itself is also a priority
for subsequent architecture/loss refinement. Probe numeric results:
`runs/geomfrac_et_v4_open_loop_probe.json`.

## Frozen V4 independent final-test acceptance

Checkpoint selection was performed exclusively on the 15 autonomous development histories across T04/D01-D04.

- 5g: development mean dynamic-Z relative L2 = 16.870%
- 8g_quality: development mean dynamic-Z relative L2 = 16.315%
- 30g: development mean dynamic-Z relative L2 = 16.315%
- 30g_quality: development mean dynamic-Z relative L2 = 16.315%

Frozen checkpoint: `runs/geomfrac_et_v4_8g_quality.pt`.
Final 5-geometry E01-E05 mean dynamic-Z relative L2 = 18.464%; maximum trajectory error = 33.653%.

| Unseen physics teacher | dynamic Z rel L2 | max temperature error |
|---|---:|---:|
| bfzi_s5_E01_ultra_v4_drive1 | 13.559% | 0.517 K |
| bfzi_s5_E01_ultra_v4_drive2 | 10.236% | 5.135 K |
| bfzi_s5_E01_ultra_v4_drive3 | 16.200% | 9.230 K |
| bfzi_s5_E02_ultra_v4_drive1 | 14.260% | 0.235 K |
| bfzi_s5_E02_ultra_v4_drive2 | 16.474% | 0.897 K |
| bfzi_s5_E02_ultra_v4_drive3 | 25.380% | 1.633 K |
| bfzi_s5_E03_ultra_v4_drive1 | 13.885% | 0.411 K |
| bfzi_s5_E03_ultra_v4_drive2 | 15.316% | 1.193 K |
| bfzi_s5_E03_ultra_v4_drive3 | 24.175% | 1.679 K |
| bfzi_s5_E04_ultra_v4_drive1 | 13.304% | 2.700 K |
| bfzi_s5_E04_ultra_v4_drive2 | 19.444% | 6.748 K |
| bfzi_s5_E04_ultra_v4_drive3 | 25.153% | 20.331 K |
| bfzi_s5_E05_ultra_v4_drive1 | 13.470% | 0.573 K |
| bfzi_s5_E05_ultra_v4_drive2 | 22.460% | 1.897 K |
| bfzi_s5_E05_ultra_v4_drive3 | 33.653% | 3.446 K |

The final test geometries were excluded from training, checkpoint choice and all prior model-development comparisons.
