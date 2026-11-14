# V2.1 八层海水近源温变 J(q,t) 与弱式焦耳热守恒分配

日期：2026-11-14。原 BFZI V150，P03/P07 真实非相交旋转 OBB，572/1312 原湿 FV，八层2mm海水热网络，600s完整高功率电磁热直接法。

## 已实现的闭环

新增可选 `skin_heat="source"`，原 `skin_heat="volume"` 继续作为默认。对原BFZI全部真实湿FV源高斯点以及八层海水热棱柱采样点预计算中心线与z-lead源场 C(q)。每个非线性热步从八层状态重建真实求积点温度，再将单元求积温度重新中心化到Green使用的守恒湿FV均温，形成局部电导率 sigma(q,t)。

以原BFZI电磁求解的每FV平均电流密度 Jbar_i 保留为基础，源场内恢复：

```text
mean_i[sigma*C_p] = sum_r(w_r*sigma_r*C_p(r))/sum_r w_r
J(q,t) = Jbar_i(t)-j*omega*mu*sum_p I_p(t)*
          (sigma(q,t)*C_p(q)-mean_i[sigma*C_p])
```

按原湿FV高斯点积分，上式的平均J恰等于Jbar_i。局部热密度初值为 g(q)=0.5*|J(q)|²/sigma(q)，热层使用面片物理棱柱求积点自身温度。对每个原湿FV i，以原弱式焦耳热总功率 P_i=q_i*V_i为固定目标，构造体水权重 B_i=Vbulk_i*mean_native(g_i)，层源权重 S_r=w_r*g_r；所有热源按 P_i/(B_i+sum_{r in i}S_r) 正比例缩放，分配给体水和八层节点。保持每原湿FV热源功率、端口电损和热容量不变，实际进入隐式热方程的非均匀源项随时间更新。

## P03/P07完整600s两网格对照

动态阻抗差以每网格自身冷态Z(t)-Z(0)的六个实分量全时程相对L2计算。仅改变八层海水局部热源分配，subcell温变电磁积分及物理参数一致。

| 几何 | 热源分配 | 动态ΔZ跨网格差 | 全Z跨网格差 | 封装最大跨网格温差 | 铜最大跨网格温差 | 海水体积均温最大差 |
|---|---|---:|---:|---:|---:|---:|
| P03 | 原均匀体积分配 | 1.172898% | 0.006884% | 0.169397 K | 0.038452 K | 0.000376 K |
| P03 | 新source分配 | 1.111566% | 0.006858% | 0.276988 K | 0.081556 K | 0.000534 K |
| P07 | 原均匀体积分配 | 0.289792% | 0.011025% | 0.715277 K | 0.424227 K | 0.002469 K |
| P07 | 新source分配 | 0.210567% | 0.010843% | 1.129048 K | 0.661981 K | 0.003001 K |

P03/P07动态复阻抗温变增量跨网格差分别下降5.23%和27.34%。封装/铜温度跨网格差增加，反映局部源在原湿FV与细热层之间仍存在离散敏感性。后续优先使用不同原BFZI网格共享的同一封装面片物理求积基准提高层功率空间收敛，保留两种模式可供严格对照。

## 8层热源占比及守恒

| 几何 | 原湿FV | 层源峰值功率 | 最大层源/海水电损 | 新方案对均匀方案封装温度最大差 | 节点热残差峰值 | 端口/弱式电损最大相对差 |
|---|---:|---:|---:|---:|---:|---:|
| P03 | 572 | 2.9510 W | 7.1103% | 0.2204 K | 5.078e-7 W | 5.149e-16 |
| P03 | 1312 | 2.9352 W | 7.0037% | 0.1848 K | 5.074e-7 W | 5.100e-16 |
| P07 | 572 | 11.5704 W | 9.3085% | 1.1192 K | 1.982e-6 W | 4.669e-16 |
| P07 | 1312 | 11.1701 W | 8.8662% | 0.8616 K | 1.975e-6 W | 5.770e-16 |

14项局部测试通过，含温度非均匀sigma(q)加权源场的原湿FV平均J约束、逐湿FV热功率精确分配及均匀场极限。真实四条600s时程完毕。

## 实现与数据

- `sdfmpneo_bfzi/near_source.py`：温变局部J与非负热源分配。
- `sdfmpneo_bfzi/sea_layer.py`：8层本地T(q)、sigma(q)、近源J(q)与保守热源进入每一个电磁热步。
- `examples/bfzi_obb_subcell_study.py`：可选 `--skin-heat source` 的真实P03/P07两网格600s实验。
- `tests/test_bfzi_skin_source.py`：局部守恒验证。
- `runs/bfzi_obb_skin_source.json`：动态阻抗、温度、源占比与热残差结果。
- `runs/bfzi_pose_p0{3,7}_skin_source_subcell_{medium,extra}.npz`：全部四条物理直接法轨迹。

复现：

```powershell
python -m pytest -q tests/test_bfzi_skin_source.py tests/test_bfzi_near_source.py tests/test_bfzi_subcell_temp.py tests/test_bfzi_sea_layer.py
python -m examples.bfzi_obb_subcell_study --bfzi-root D:/work/bfzi-et-v2.1/BFZI --runs runs --poses P03 P07 --levels medium extra --source-masks subcell --skin-heat source --out runs/bfzi_obb_skin_source.json
```

## 下一步

使用封装面片共用的高阶物理热源采样与原湿FV的严格体积交集，降低细层功率的跨网格差；在此基础上减少封装和铜的跨网格温差，再扩展新的物理真值到五材料32阶热ROM及几何分数阶神经代理。
