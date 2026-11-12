# V2.1 原 BFZI 三维热 ROM：五材料守恒、M3 新几何真实验证

日期：2026-11-12。代码分支 `research/bfzi-uwpt-electrothermal-surrogate`。前提是既有的 BFZI V150 原始 M3 冷态场锚 `runs/m3_bfzi_anchor.npz` 与已训练、M3 动态完全留出的几何分数阶记忆 `runs/bfzi_geom_frac_v5.pt`。

## 本轮新增

`sdfmpneo_bfzi/geom_rom.py` 实现共享 BFZI 冷态海水弱形式焦耳源、铜源与热导算子的**容量正交热 POD**；`examples/bfzi_geom_heat.py` 增加 `--rom-rank`、`--rom-file`、`--compare-full`，保留原全三维物理法以随时比较和读取温度/三维 J/q。

构基只利用预存的冷态 BFZI 热矩阵、5 种端口空间加热模式与对应的纯物理隐式热扩散响应；**不读取 M3 实测动态教师温度作为降阶训练数据**。构基时间尺度覆盖 1–3600 s 与关闭热源后的冷却段，涵盖启动、关断、再启动和长时间传热。

五个精确常数模式依次代表：
1. 海水全部 FV 单元
2. 发射线圈三维封装
3. 接收线圈三维封装
4. 发射铜线圈
5. 接收铜线圈

再从去掉上述材料平均模式的物理源热响应中提取空间梯度 POD，采用

```text
Phi^T diag(C) Phi = I
K_rom = Phi^T H Phi
(I + dt K_rom) a_next = a_prev + dt Phi^T Q(theta_next, I_next)
theta = Phi a
```

每个时间步的 Joule 源与完整原 BFZI 冷态锚代理保持同一端口海水功率、铜趋肤损耗、双向电路反馈，并通过迭代完成热源/铜温度同步。基中包含五个材料常数模式，因而保证每种材料**积分储热—传导—功率收支**在数值精度内闭合。局部逐 FV 节点能量残差包含截断误差，不应把投影材料收支与逐节点精确满足混同。

ROM 一次构建约 **0.217 s**；保存为 `runs/m3_bfzi_rom.npz` 后，重新加载约 **0.0039 s**。本次保留 32 阶热空间（原 M3 共 2316 个热态，32/2316=1.38%），可输出原所有 FV 温度、J/q 场及任意 XYZ 查询。已有新几何 BFZI 冷态计算不算入 ROM 轨迹时间。

## 原 BFZI 完整留出 M3 时程实测

M3 原始几何：圆角方形 240/240 mm、13/13 匝、100 mm 间距；原海水 2250 个 FV，双封装 64 个 FV，双铜节点，总 2316 状态。V5 分数阶电磁网络只用 M1/M2/M4 动态数据训练，M3 动态始终留出。

下表的**温度/Z/q 附加误差分母是同一个冷态 BFZI 锚定的完整 2316 维热代理**，也就是说衡量的是热 ROM 带来的误差；与完整原 BFZI 动态直接法之间已有独立 M3 参考对照。

| 指标 | 40 V / 30 s / 21 点 | 115 V / 30 s / 21 点 | 100 V / 15 s / 41 点 |
|---|---:|---:|---:|
| 32 阶 ROM 动态计算 | 0.07267 s | 0.07833 s | 0.16435 s |
| 2316 态完整热代理 | 0.10498 s | 0.10534 s | 0.22461 s |
| 热求解/全流程实际加速 | 1.445× | 1.345× | 1.367× |
| **海水全空间最大温差** | 3.548e-5 K | 1.766e-3 K | 1.319e-3 K |
| **双封装全空间最大温差** | 1.641e-5 K | 7.654e-4 K | 6.183e-4 K |
| 铜温度最大差 | 9.737e-11 K | 8.159e-8 K | 2.523e-8 K |
| 复端口动态 ΔZ 附加相对 L2 | 1.752e-11 | 1.213e-9 | 4.541e-10 |
| 海水 q 全空间附加相对 L2 | 1.110e-7 | 5.520e-6 | 4.021e-6 |
| **五种材料最大能量预算差** | 5.240e-13 W | 7.726e-11 W | 2.056e-11 W |
| 海水 q 体积积分与端口功率差 | 7.105e-15 W | 5.684e-14 W | 4.263e-14 W |
| ROM 逐 FV 节点最大残差 | 1.909e-5 W | 1.104e-3 W | 7.396e-4 W |

ROM 在热输运上进行了投影近似，所以局部 FV 方程残差包含降阶截断；所列五种材料**总体能量闭合**保持在 1e-10 W 内。

### M3 原 BFZI 直接法参照精度

对 40 V/115 V 真实教师，以本 ROM 替换完整热代理后，其相对于原 BFZI 直接法的全空间最大误差分别为：

| 材料 | 40 V | 115 V |
|---|---:|---:|
| 海水最大误差 | 3.695e-5 K | 1.679e-3 K |
| 双封装最大误差 | 1.814e-5 K | 8.143e-4 K |
| 铜最大误差 | 2.852e-5 K | 1.682e-3 K |
| 动态复阻抗变化相对误差 | 5.827% | 7.913% |

完整版三维物理代理此前相对于原 BFZI 直接法已经真实测得 158×/209× 的时程加速。增加 32 阶 ROM 后，M3 本机运行进一步减少三维热推进成本；速比按本轮同机器各自重新实测记录，避免混用不同实验时间。

## 不依赖原 BFZI 的可执行入口

首次从同一 M3 冷态锚生成物理热 ROM 并和 2316 态完整热模型比较，自动保存基：

```powershell
python -m examples.bfzi_geom_heat --anchor runs/m3_bfzi_anchor.npz --teacher runs/m3_frac_holdout.npz --model runs/bfzi_geom_frac_v5.pt --rom-rank 32 --rom-file runs/m3_bfzi_rom.npz --compare-full --out runs/m3_bfzi_rom_metrics.json
```

后续任意电压、启停历史和时间步复用已保存的 ROM：

```powershell
python -m examples.bfzi_geom_heat --standalone --voltage 100 --heat 210 --cool 210 --step 15 --horizon 600 --anchor runs/m3_bfzi_anchor.npz --model runs/bfzi_geom_frac_v5.pt --rom-rank 32 --rom-file runs/m3_bfzi_rom.npz --out runs/m3_bfzi_rom_user.json
```

输入已有原 BFZI 冷态锚，而每时刻不会运行 BFZI Green/GMRES 电磁求解。输出仍含所有三材料三维温度、海水 J/q、动态复阻抗及材料能量预算。需做 XYZ 查询时可沿用 `--query runs/m3_bfzi_query.npy --query-time 300`。

数值记录：`runs/m3_bfzi_rom_metrics.json/.npz`、`runs/m3_bfzi_rom_15s.json/.npz`、已存的 `runs/m3_bfzi_rom.npz`。新增测试 `tests/test_bfzi_geom_rom.py`，与已有三维保守和分数阶测试合计 **9 passed**。

## 下一步

- 直接在原 BFZI 的连续圆角参数、线圈相对位姿和封装尺寸变化下重建冷态电磁及热算子，为每个几何生成物理热基；跨几何分数阶电磁模型与三维热 ROM 联动验证。
- 逐步建立参数条件化、具有物理材料预算的空间基，让临近几何的 ROM 共享，减少新几何预计算。
- 在进一步提高动态海水复阻抗预测精度时，保留原 BFZI 铜趋肤方程及每材料热预算，不通过牺牲空间物理来降低标量误差。
