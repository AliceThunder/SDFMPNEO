# V2.1 原 BFZI 近源电流 J(q,t) 与守恒弱式焦耳热 q(q,t)

日期：2026-11-14。使用原 BFZI V150 P03/P07 真旋转OBB、572/1312 湿FV、8层2mm保守海水热状态。复用已归档原 BFZI 600s动态物理时程，原温变Green/GMRES、端口电路与热轨迹不重算。

## 物理重建方法

原 BFZI 近源二阶矩积分沿用中心线及z-lead产生的双端口源场 C_p(q)。现在同时保留原湿FV高斯点的实际源矢量和体积均值 Cbar_i。每个湿FV原BFZI的复电流密度 J_i(t)保持不变：

```text
J(q,t) = J_i(t) - j*omega*mu*sigma_i(t)*sum_p I_p(t)*(C_p(q)-Cbar_i,p)
```

该恢复保留原FV平均J，补充原BFZI已有近源源积分能够分辨的空间变化。sigma_i沿用守恒FV体积均温，单位端口原BFZI Green/GMRES未增加求解。

用此源场构造局部正Joule原始形状 `g(q)=0.5*|J(q)|²/sigma_i`，并按原热求解采用的FV弱式损耗 `q_i` 逐单元重新分配：

```text
q(q,t) = q_i(t)*V_i*g(q,t) / sum_{r in FV_i} w_r*g(r,t)
```

由此精确保持每个湿FV的总热功率 q_i V_i、所有海水端口损耗以及原系统热容量；三维J/q分布可以在任意真实海水物理点查询。

## P03/P07 实际空间误差（共同真实湿FV求积点）

场误差为572→1312物理共同空间点的体积加权相对L2，原单元常值场与新源恢复场在完全相同位置比较。P03 600s处断电，相对场误差按零功率处理。

| 几何 | 时间 s | 原J跨网格差 | 新J(q)跨网格差 | 原q跨网格差 | 新q(q)跨网格差 |
|---|---:|---:|---:|---:|---:|
| P03 | 0 | 51.3323% | 10.2515% | 45.0374% | 3.9096% |
| P03 | 60 | 51.3918% | 10.2514% | 45.1130% | 3.8708% |
| P03 | 300 | 51.5316% | 10.2461% | 45.3342% | 3.7472% |
| P03 | 600 | — | — | — | — |
| P07 | 0 | 51.9483% | 10.7237% | 49.1405% | 4.3855% |
| P07 | 60 | 52.1596% | 10.7203% | 49.2894% | 4.3319% |
| P07 | 300 | 52.8227% | 10.7141% | 50.1756% | 4.3146% |
| P07 | 600 | 53.4380% | 10.8316% | 51.1211% | 4.8852% |

## 守恒数值结果

P03：共同求积点 18131，平均J最大残差 2.491e-16，弱式热源功率最大残差 3.391e-16。

P07：共同求积点 18261，平均J最大残差 2.863e-16，弱式热源功率最大残差 3.472e-16。

相关局部数值测试13项通过。重建无需额外温变电磁求解，热弱式源、端口复Z和所有原热节点温度均保留。实际误差下降主要来自把原BFZI未解开的近源几何变化恢复到原湿FV求积点，空间场仍有进一步改进空间。

## 文件、复算与下一项

核心代码：`sdfmpneo_bfzi/near_source.py`、`sdfmpneo_bfzi/bfzi_v148_adapter.py`。
完整实算：`runs/bfzi_obb_source_field.json`。
入口：`examples/bfzi_obb_source_field.py`。
测试：`tests/test_bfzi_near_source.py`、`tests/test_bfzi_subcell_temp.py`。

```powershell
python -m pytest -q tests/test_bfzi_near_source.py tests/test_bfzi_subcell_temp.py tests/test_bfzi_field_transfer.py
python -m examples.bfzi_obb_source_field --bfzi-root D:/work/bfzi-et-v2.1/BFZI --runs runs --poses P03 P07
```

下一项将局部8层海水温度T(q,t)、对应sigma(q,t)接入高阶电流与同源局部q恢复，进一步细化原湿FV的热源在近表面层与粗FV间的保守分配，再用独立物理600s时程和高阶空间J/q评价。
