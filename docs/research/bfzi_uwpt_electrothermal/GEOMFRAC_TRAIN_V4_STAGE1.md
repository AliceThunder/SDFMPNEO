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
