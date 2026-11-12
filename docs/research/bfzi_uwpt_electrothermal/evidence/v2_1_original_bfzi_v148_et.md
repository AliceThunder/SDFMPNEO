# V2.1：独立 BFZI V148 电磁计算与温度动态反作用适配

**目标：** 复用独立仓库 `AliceThunder/BFZI` 原始 V148 算法，连接海水空间温度、铜温度与随时间演化的 2×2 复阻抗；在原 BFZI 空间体单元上求热，不再额外构造另一套端口阻抗求解器。

**实现状态（2026-11-12）：** 适配层、原 BFZI 有限体积热导构造、温度时程 CLI 与专项测试代码已提交；沙箱 2D 四节点加权投影局部验证通过。**完整 BFZI V148 私有依赖环境下的双线圈运行及真实复阻抗误差尚未获得，本轮不可将其记为数值 PASS。**

## 2026-11-12 最小实际运行推进

首次在沙箱执行原 V148 热积分器同逻辑的小型测试时发现并修复错误：`BFZIV148ET.em` 曾将 `heat` 调用在求解结果 `V148State` 上，正确对象为原 BFZI 适配器 `self.model.heat(state,I)`。修复后，双端口通电—关断冷却—再次启动的 **3 项沙箱测试全部通过**；对应项目测试文件同步加入再启动历史和单步热量收支。

为了让强耦合热步的能量收支足够接近原隐式方程，热步固定点默认允许最多 **8 次迭代**，当相邻温度最大差小于 `1e-5 K` 时提前结束。沙箱 4 FV 海水单元、2 铜节点、2 封装节点的小型验证中，`Δt=1s` 的热输入 `0.159527762686 W`，储热增量 `0.020124695960 J`、向环境散热 `0.139403056680 W`，闭合相对差 **6.30×10⁻⁸**。这属于合成耦合算例的数值闭合，真实 BFZI 的热学参数与三维场精度继续 OPEN。

此外直接读取独立 BFZI 仓库 `outputs_v148/bfzi_v148_all_samples_matrix.csv`，取得 M4 原始 200 kHz、σ=5 S/m 的海水复阻抗回归目标：

```text
Zsea11 = 0.12781381449951726 - 0.00385452694387080j Ω
Zsea12 = 0.10805885891894453 - 0.00373162961285344j Ω
Zsea22 = 0.12781381449951723 - 0.00385452694387080j Ω
```

动态示例代码现同时输出：1）与同场景原 `calc.formal_with_local_mass` 直接重求值的相对误差；2）原 BFZI 存档 M4 复阻抗相对差；3）每个时刻海水端口功率与空间 Joule 热积分相对差。**这些指标属于待实际运行填充的检查，尚无真实 M4 动态数值结果。**

## 已有算法与本轮新增

原 BFZI 保留的实现：

- `bfzi_v148_results.py`：M1/M2/M3/M4 的真实圆角方形与圆形极限几何、实际端口来源；`compute_air_matrix_direct_zlead` 原空气自感互感；`formal_with_local_mass` 原海水复阻抗与源近场质量修正。
- `run_v130_local_source_support_zortho.py`：原有限体积标量势投影、Green/resolvent 方程、阻抗双线性读出。
- `bfzi_fast_sea.py`：原有源矩与 FFT 相关工具（独立后续优化路径）。
- SDFMPNEO `BFZISources.conductor_internal_impedance`：铜线皮肤效应复阻抗与铜温度材料律。

本轮增量代码：

| 文件 | 功能 |
| --- | --- |
| `sdfmpneo_bfzi/bfzi_v148_adapter.py` | `WeightedFV`、`BFZIV148Adapter`；原 BFZI V148 海水 Green/resolvent 和源局部质量修正的空间温度驱动 |
| `sdfmpneo_bfzi/v148_et.py` | `BFZIV148ET`；原 BFZI 有限体积海水面导热、两个封装/两个铜温度节点、动态复阻抗电路热反馈 |
| `examples/bfzi_v148_dynamic.py` | `run.py --mode bfzi-et-v148`，连接已有原 BFZI checkout，运行通电—关断冷却—再启动 |
| `tests/test_bfzi_v148_et.py` | 加权有限体积投影与小规模升温/停机冷却专项；本轮仓库完整 pytest 仍未实跑 |

### 计算原理

对原 BFZI 的每个海水有限体积单元 `k`，保留原物理坐标、体积、导体源与 Green 算子，更新温度：

\[
\sigma_k(t)=\sigma_0\left[1+\alpha_\sigma(T_k(t)-T_0)\right],
\qquad
\gamma_k^2=j\omega\mu\sigma_k.
\]

从原 BFZI 的面相邻关系中使用相邻 `σ` 的调和平均作为面通量系数，求解 `∇·(σ∇φ)=∇·(σ F)` 的有限体积投影，从而在材料不均匀时保持电荷守恒路径。这比仅修改原 Green 算子的统一 `σ` 常数更进一步。

采用原本的 `(I+G_0 V P_\phi)X=C` 算子、原闭合源 `C` 与原双线性端口读出，局部质量校正仍按 V148 的近源几何掩码和源积分构造。不同之处是先缓存每个近源体单元的积分差，再与当前 `γ_k^2(T)` 分别相乘，避免以一个全域平均电导率代表空间变温。

得到完整 `Z_{\mathrm{sea}}(t)`，与原始 BFZI Neumann/GMD `L_{\mathrm{air}}` 和已有铜内部 `Z_{\mathrm{cu}}(T)` 相加。海水单位端口电流从原 `P_\phi X` 响应重构，使用同一电流计算海水体积分 Joule 热。**保留海水端口损耗与空间 Joule 积分两个独立读出以供实测差异判断；不人为强制二者相等。**

热方程在原 BFZI 的海水有限体积邻接面上使用 `kA/d` 导热，外海面换热；封装与铜分别为热节点，使用热容、导热连接和铜 Joule 热。每个热步根据空间温度更新 `σ_k`、铜复阻抗和 2×2 复阻抗，通过所选被动外电路求新电流并推进热态，包含负载和开关历史。

固定几何的空气自感互感保持原值，海水温度引起的复反作用阻抗虚部使**总有效自感与互感随时间变化**。

## 直接运行原始 BFZI M4 双圆形极限算例

首先已有本地 BFZI checkout（路径由用户自行指定，程序仅读取，不修改原 BFZI 仓库）：

```bash
python run.py --mode bfzi-et-v148 \
  --bfzi-root /path/to/BFZI \
  --sample M4 \
  --voltage 3 \
  --step 60 --heat 240 --cool 180 --restart 180 \
  --out runs/bfzi_v148_dynamic.json
```

`M1`、`M2`、`M3` 同样可通过 `--sample` 选择。单次场景生成并缓存原始 BFZI 体积 Green 算子、源矩与局部质量矩；每个热步在更新的温度材料分布下重算电磁及电路响应。

输出：

- `runs/bfzi_v148_dynamic.json`：时间、电压阶段、复 `Z11,Z12,Z22`、海水复阻抗、等效电感、端口电流、海水体积分/端口功率、铜 Joule 功率、储热、最大温度和原 BFZI 等温海水阻抗比较误差；
- `runs/bfzi_v148_dynamic.npz`：每个时间步的**原 BFZI 海水 3D FV 中心、体积、T(x,t)、q(x,t)**，封装与铜温度历史。

默认比较同一等温条件下新接口与原 `bfzi_v148_results.formal_with_local_mass` 计算的海水阻抗，相对偏差直接写入 JSON，待真实运行后据此修复。

## 本轮验证与后续

- **沙箱实际通过：** 四节点二维有限体积闭环上的 `σ` 加权投影检查。对统一 `σ=1` 与 `σ=5`，投影结果数值一致；空间 `σ=[1,2,4,8]` 产生 **12.35%** 相对变化，复值有限体积投影正常。
- **代码已实现，未在完整依赖环境实跑：** 原 BFZI V148 双线圈 M1–M4、同一算例等温 Z 的数值回归、完整动态 T/Z/I/q 时程及其能量收支。测试文件已提交，不能把其存在当作测试通过。
- **精度阶段：** 当前热模块在海水用原 BFZI 的 FV 热传导，在两个封装和两根铜线用集中热状态。原 BFZI v148 的几何范围以现有 M1–M4 案例为起点；任意双 SE(3) 的原算子扩展、铜/封装完整三维温度以及独立热/电磁高保真参考均继续推进。

**下一关键项：** 在具有独立 BFZI checkout 的 Python 环境中执行上面的真实 M4 时程，核对等温相对误差、动态复阻抗、真实热源及升温/关断冷却。发现问题直接修改实现，随后扩展跨几何与空间热 ROM。


## 2026-11-12 下一轮：等温回归提前执行和动态传输效率

原始 BFZI 等温复阻抗现在是**进入长时热计算的前置实际数值检查**。新增 `--check-only` 只构造原 BFZI M4 电磁数据、求初始全复海水阻抗，并对照两项数值：
- 原 BFZI `calc.formal_with_local_mass(data)` 的等温 `Zsea`；同方法数值回归相对阈值设为 `1e-5`。若未达到则中止完整热积分，并保留独立检查结果 JSON。
- 直接读取独立 BFZI checkout 中 `outputs_v148/bfzi_v148_all_samples_matrix.csv` 的 M4 原始海水复阻抗矩阵。由存档列 `R_ohm` 与 `L_sea_reaction_uH` 按 `R+jωLsea` 重建，不再硬编码存档阻抗常数。

```bash
python run.py --mode bfzi-et-v148 --bfzi-root /path/to/BFZI --sample M4 --check-only --out runs/m4_check.json
python run.py --mode bfzi-et-v148 --bfzi-root /path/to/BFZI --sample M4 --voltage 3 --source-r 0.5 --load-r 0.5 --step 60 --heat 240 --cool 180 --restart 180 --out runs/m4_dynamic.json
```

真实时间记录新增 `Pinput_W`、`Pload_W`、`Psource_loss_W` 与 `efficiency`。电路采用完整温度相关 `ZEM(T)`，发射侧串联源电阻 `Rsrc`，接收侧电阻负载 `Rload` 和固定串联容性补偿：

[
P_{\mathrm{in}}(t)=\tfrac12\operatorname{Re}[\mathbf V(t)^\dagger\mathbf I(t)],\quad
P_{\mathrm{load}}(t)=\tfrac12 R_{\mathrm{load}}\lvert I_2(t)\rvert^2,\quad
\eta(t)=P_{\mathrm{load}}(t)/P_{\mathrm{in}}(t).
]

沙箱合成两端口电路 `V=(1,0)V`、`Rsrc=Rload=0.5Ω` 的输入功率 `0.7575755978738048W`、接收负载功率 `0.00013175411099177107W`、效率 `0.00017391546317166142`，完整代数功率分账残差 `5.55e−17W`；动态热时程局部测试仍 **3 passed**。上述结果为合成物理耦合验证，**真实 M4 等温回归结果、温度时程及效率仍待执行**。
