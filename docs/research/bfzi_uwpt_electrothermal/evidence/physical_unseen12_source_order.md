# BFZI V150 冻结未见样本的源积分阶数精度复核

2026-11-15。依据 PHYSICAL_DIRECT_FINAL_ACCEPTANCE_V1.md，保持 U01–U12 工况、原 ultra/superfine 网格、真实 1.875/1 s 热步和各项门槛不变。

## S3 全耦合复算发现

三阶源积分修复原 BFZI 固定八点近源误差后，U01/U02/U03/U11 已完整通过四组合。原本通过的高功率 U04 与 U10 在新的三阶方案上出现空间精度回退：

| 案例 | 原二阶源积分完整时程空间 ΔZ | S3 三阶源积分完整时程空间 ΔZ | 主要差异 |
| --- | ---: | ---: | --- |
| U04 | 0.9252% | 2.6103% | 第 1 端口动态 R |
| U10 | 0.8763% | 2.4853% | 第 1 端口动态 R |

两组海水及铜温度处于规定物性域、空间温差满足阈值，功率与热预算守恒，故主导问题是近源电磁空间求积稳定性。

## 独立温度场的阶数对照

以 S3 已完成的两网格热时程作固定温度输入，分别重建同一 BFZI V150 电磁求解器的三、四、五阶源积分。以下仅比较 **t=390 s 与 t=0 s 的复阻抗变化**，是单时刻诊断，尚未代替完整时程四组合验收。

| 案例 | 4 阶源积分，390 s 空间 ΔZ | 5 阶源积分，390 s 空间 ΔZ |
| --- | ---: | ---: |
| U01 | 0.7018% | 待复算 |
| U04 | 1.2508% | 0.6305% |
| U10 | 1.1625% | 0.5129% |

U04 从四阶到五阶误差继续下降，表明三阶近源 Gauss 规则尚未达到稳定的网格独立性。

## 代码与复验

- `BFZIV148Adapter` 新增 `source_order`，用于控制完整湿 FV 的近源 Gauss 积分阶数，三阶保留作为 S3 已记录的默认基线。
- `examples/bfzi_physics_recheck.py` 新增 `--source-order` 并写入运行 JSON 的 settings。
- 局部测试 `tests/test_bfzi_v148_et.py tests/test_bfzi_sea_layer.py`：24 passed。
- 启动五阶方案 U04/U10 的两网格、真实热步 1.875 s、600 s 全耦合计算，独立保存为 `_s5`，不覆盖 `_s3` 和原冻结基线。

复现命令：

```powershell
python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys U04 U10 --dt 15 --grids ultra superfine --inner-step 2 --source-order 5 --run-tag s5 --out runs/bfzi_unseen_u04_u10_s5_h2.json
```

最终 12/12 状态：**OPEN**。五阶完整闭环与全集四组合复验未完成前不锁定物理参考版本。
