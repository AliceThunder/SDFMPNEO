# D0 → D1 独立物理直接法：Shafiei 三层 Helmholtz 复现记录（2026-11-06）

> **状态：D0 受限场景量级核验完成；D1 谱域核心实现和论文表 2 的部分复现完成，D1 总关口仍 OPEN；D2 尚未实现。** 旧 G1/P1-C 不变为 FAIL。不得以本文结果启动神经教师。
>
> 权威物理范围参见 [锁定规范](PHYSICAL_DIRECT_METHOD_LOCKED.md) 与 [验收表](PHYSICAL_DIRECT_METHOD_EVIDENCE_AND_GATES.md)。
> 实现：`sdfmpneo_bfzi/layered_helmholtz.py`；精简测试：`tests/test_bfzi_d1_layered_helmholtz.py`。

## 1. D0：定义与模型量级（基线限定，非全参数域批准）

- 峰值电流相量 `e^{+iωt}`；正向定义沿圆形源环固定方向；`Z_pq = V_p / I_q`（只激励 q），单位为欧姆。两个端口均可独立激励，`Z_12=Z_21`。
- 基线真实 BFZI 应为圆形／圆角方形**单根连续螺旋 + 真实闭合馈回流**，两独立 SE(3) 实心绝缘盒，均匀海水导电外域。**以下 Shafiei 标准件不是此 BFZI 几何**：它使用分离的、共轴同向圆环叠加与横向无限平面层。不得把它转换成两个独立 OBB 或声称端口铜自阻抗已知。
- `μ=μ0`、各层 `σ>=0`、`εr>0`、静止无磁芯。普通 MQS 用 `u=sqrt(k²+iωμ0σ)`；严格复现 Shafiei 论文时用 `u=sqrt(k²+iωμ0σ-ω²μ0ε0εr)`。根支取 `Re u>=0` 并采用与 `e^{+iωt}` 相容的外向分支。
- 表 1 海水 `σ=4 S/m, εr=81, μr≈1`。`f=400 kHz` 时海水 `ωε/σ≈4.51e-4`，`δ=sqrt(2/(ωμ0σ))≈0.398 m`；`f=460 kHz` 则分别为 `5.18e-4`、`0.371 m`。海水中小的位移电流比支持受限 MQS 近似，并不排除真实装置的其他介电层、全波效应或自感截面误差；必须同时检查装置尺寸及材料参数。
- 细丝零半径自感发散，不能由本文 `ΔZ` 还原铜总自阻抗；分离保存场路外接阻容、铜域和海水增量。当前 D0 的 BFZI P0 源场只在已测探针集有合格证据；其全几何域不做笼统 PASS。

## 2. 已核对的 Shafiei 原始论文具体依据

Sadjad Shafiei, Seyed Saeid Heidari Yazdi, Nurtas Adaikhan, Mehdi Bagheri, *A Detailed Analytical Modeling of Eddy Current Losses in an Underwater Medium for Wireless Power Transfer*, **IEEE Access** 13 (2025), 28988–28999. DOI: https://doi.org/10.1109/ACCESS.2025.3539925

可读正式 PDF（CC BY）：https://research.nu.edu.kz/files/89363636/A%20Detailed%20Analytical%20Modeling%20of%20Eddy%20Current%20Losses%20in%20an%20Underwater%20Medium%20for%20Wireless%20Power%20Transfer%20%20ac7dd746.pdf

- 第 28990 页式 (1)–(6)：Maxwell 含位移电流、Bessel J1 径向分离、复传播常数；
- 第 28991 页式 (10)–(17)：界面切向 E 连续与环电流源导数跳跃、线圈开路感应电压；
- 第 28992–28993 页式 (20)–(31)：三个平面层场景的附加自阻和互阻积分；
- 第 28993 页表 1：`Nt/Nr=15/17`、内径 170/165 mm、净匝距 1 mm、导线直径 1.5 mm、`σ=4 S/m`、`εr=81`；
- 第 28994 页：线圈间距 10 cm；第 28997 页表 2：85 和 400 kHz 的有限元、解析、测量参考表。
- 无非共轴偏移、没有电线截面场解或两个有限尺寸绝缘盒；论文分层介质在 xy 平面无限。真实 BFZI 后续不得拿其轴对称解验收非共轴几何。

### 关于匝距的重要复现细节

表 1 的 `space turn=1 mm` 是**导线之间净距**，另有 `wire diameter=1.5 mm`。采用径向中心距 2.5 mm、内径首环中心半径近似为 85/82.5 mm，可准确再现无限与半无限结果。若错误地取中心距 1 mm，则所有电阻显著偏小。尽管论文解析模型忽略铜内部截面电磁效应，参数表仍用导线直径确定环中心距。此离散环集合**不等于 BFZI 连续绕线**。

## 3. 实现原理（同类文献的独立 Dirichlet-to-Neumann 重写）

令 `z=0,d` 为发射、接收平面，层域 `z<0, 0<z<d, z>d`；`k` 为 Hankel 波数，
`u_j = sqrt(k²+iωμ0σ_j-ω²μ0ε0εr,j)`（MQS 去除末项），`q_j=u_j/μ0`，`t=exp(-u_2 d)`。两个接口的标量环向矢势 Hankel 核满足源跳跃，消去层内解后

```text
D   = (q2+q1)(q2+q3) - (q2-q1)(q2-q3)t²
G00 = [(q2+q3)+(q2-q3)t²]/D
Gdd = [(q2+q1)+(q2-q1)t²]/D
G0d = 2 q2 t/D
```

它等价于界面 Dirichlet-to-Neumann 矩阵求逆，且不包含指数增长项 `exp(+u2*d)`。对于相同介质，`G00=Gdd=μ0/(2u)`, `G0d=μ0 exp(-ud)/(2u)`。对于两个端口每匝半径 `a_p,n`，定义 `S_p(k)=Σ_n a_p,n J1(k a_p,n)`，完整环路场造成的复端口相互作用为 `Z_pq=iω·2π∫ k S_p(k)G_pq(k)S_q(k) dk`。没有有限截面自感正则化时，`Z_pp` 的真空磁自项发散；实现返回同一薄环集合相对真空的**有限差值 `ΔZ`**，同频、同半径、同端口方向逐核相减。

MQS 的空气参照纯虚，`Re ΔZ` 为总海水诱导损耗的端口表示。保留位移电流的全波形式同时包含真空辐射差值，不能未经附加判据就以 `Herm(ΔZ)` 当作纯海水 Joule 功率。积分 `k_max` 显式有限，必须外推或加倍上限并报告差异；`quad_vec` 提供积分容差。此求解器**仅计算复端口增量**，尚未实现 `Eφ(r,z), B_r, B_z, q` 空间输出及海水体积功率闭合，不构成 D1 PASS。

### 复阻抗与场—路接口（Wang 2025）

Shimin Wang, Wangqiang Niu, Xianwen Zhou, *Modeling wireless power transfer in marine environment via integrated electromagnetic field and circuit analysis*, **Measurement** 251 (2025), 117224, DOI: https://doi.org/10.1016/j.measurement.2025.117224。出版商摘要证实将复自感、复互感与外部电路方程联立，实验使用 22.5 cm 线圈、460 kHz。无官方可用全参数数值表时，不虚构其 `Z` 数值；本模块只给电磁增量端口矩阵。电路串补容抗、负载和逆变器/整流器各自独立组装，避免双重计入水损或铜损。

## 4. 原论文表 2 独立数值复现

使用公开表 1 参数、`d=0.10 m`、径向中心距 `2.5 mm`、`include_displacement=True`、`k_max=5000 m^-1`。每格为 `R_tx/R_rx/R_mutual`（Ω，实部）；左列为论文正式表 2 的 **analytical** 栏，右列为本实现值（不是测量）。源代码中电流幅度为单位峰值相量，输出电阻与激励幅值无关。

| 介质 | f/kHz | 论文解析 / Ω | 本次计算 / Ω | 判定 |
|---|---:|---|---|---|
| 全无限海水 | 85 | .261 / .334 / .209 | .260612 / .333992 / .208871 | 符合表格精度 |
| 半无限海水（下方空气） | 85 | .133 / .301 / .152 | .133048 / .301094 / .151702 | 符合表格精度 |
| 10 cm 有限海水层（两侧空气） | 85 | .104 / .133 / .090 | .1038 / .1328 / .0902 | 符合表格精度 |
| 全无限海水 | 400 | 5.082 / 6.505 / 3.854 | 5.081307 / 6.504973 / 3.854289 | 符合表格精度 |
| 半无限海水（下方空气） | 400 | 2.668 / 6.127 / 2.974 | 2.667397 / 6.127074 / 2.973612 | 符合表格精度 |
| 10 cm 有限海水层（两侧空气） | 400 | 2.386 / 3.054 / 2.056 | 2.272970 / 2.907733 / 1.968514 | **差异约 4–5%，未消除** |

对有限层 400 kHz，额外用论文 **(29)–(31) 的显式积分**而非 2×2 Green 矩阵重算，得到相同结果；增加波数截断 1500→25000 /m，实阻变化远小于该差异。此发现只说明在已冻结参数下两条等价公式给出一致数字，**不意味着论文错了或 D1 已完全通过**。待核对作者原始复现数据、材料/几何细节或表格中未说明的设定。不得后验调整电导率、频率或验收阈值使其凑合。

在 400 kHz 对全无限、半无限和有限三种场景，将 `include_displacement=True` 改成 `False`（仅 MQS）时，各 `Re ΔZ` 差异约 `5e-5~2e-4` 的相对量级。这是**本例的数值观测**，不推广到任意高频、介电材料。

### 运行

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d1_layered_helmholtz.py -q
python -c "from sdfmpneo_bfzi.layered_helmholtz import paper_case,incremental_impedance; c=paper_case('infinite',frequency_hz=400000); print(incremental_impedance(c,k_max=5000,include_displacement=True))"
```

**执行证据边界：** 当前会话在独立 Python 计算环境实跑核心函数的四项检查和表 2 四组无限/半无限对照；GitHub 仓库版本已提交，但尚未收到用户工作站对上述 pytest 的日志。完整代码在远端可按命令复测，不能伪称 3D 本机 FEM 已完成。

## 5. D1 剩余及 D2 固定实施路线

1. **先完成 D1**：解决有限海水层 400 kHz 的表格差异；实现同一谱核在三个空间层的 `Aφ, Eφ=-iω Aφ, B_r=-∂_z Aφ, B_z=(1/r)∂_r(rAφ)`，验证层界切向场、源导数跳跃、导电区体积 `0.5 σ|E|²` 与端口功率的一致性。谱域全波需把辐射功率与耗散分开；真正做 MQS 端口功率闭合时使用 MQS 基线。用加倍波数截断+空间积分半径做误差预算，并对论文图 11–14、表 2 的 `E_max` 做相同定义的对照。
2. **再进入 D2**：以 TEAM Problem 7 作为非轴对称三维基准：`σ_Al = 3.526e7 S/m`、`μr=1`、线圈 `2742 turns × 1 A/turn`、`f=50/200 Hz`。由原始 benchmark 几何构建不对称带孔铝板与激励；不要用本项目 BFZI 线圈偷换基准。使用 `H(curl)` 保形边元 / `A–φ` 一致弱式、`φ` gauge、铜激励与非导电域约束，消除梯度零空间。
3. D2 预先冻结探针及误差口径：磁通 `B_z` 沿 `y=72 mm,z=34 mm,0≤x≤288 mm`；涡流 `J_y` 沿 `y=72 mm,z=19 mm,0≤x≤288 mm`，逐频对官方/实验数据。每个物理工况至少三个空间网格细化等级和三个外域半径（或两类独立开域处理），分别记录探针复场、铜/铝损耗、端口和计算时间；人工边界不称无限域。
4. D2 的独立 3D 解达到标准基准后，才将相同 `A–φ` 机制映射到真实 BFZI 的单连续闭合回路、单盒、双盒两独立 SE(3)；导线有限截面与细丝外部场分账，端口复 `Z`、空间 `J/B/E` 和 Joule 体热不缺项。BEM/FEM–BEM、旧 Green–Galerkin 仅在后续 D5 比较；禁止 G1 角帽试错和 GitHub Actions。

**TEAM-7 资料：** https://doc.comsol.com/6.3/doc/com.comsol.help.models.acdc.multiturn_coil_asymmetric_conductor/multiturn_coil_asymmetric_conductor.html （标准模型使用 50/200 Hz 及上述两条探针）；https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/lsdyna_vm/Hlp_V_VM_L_EMAG010.html 。两者是独立 benchmark 的外部依据，不意味着本仓库已有 D2 参考解。
