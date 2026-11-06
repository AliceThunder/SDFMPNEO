# D2 主算法纠偏：一致源加载的闭合端口三维 A–φ 边元有限元
> **紧接的独立空间场核查（2026-11-06）：** 配对的离散端口/体积功率精确相等仅代表本算法的**Galerkin 代数守恒**，不代表真实 Maxwell 场准确。将已知线源的解析直线段矢势 `A_src_exact` 加回 `A_medium_h−A_vacuum_h`，得到新的空间物理重构 `E_phys=−iω[A_src_exact+A_medium_h−A_vacuum_h]−gradφ_h`，其 50Hz TEAM7 同样四档材料区 Joule 为 **46.993822/23.557745/17.446595/15.162941 W**，远高于原有限元自身的 **6.932518/6.833806/7.091349/6.737204 W**。这说明低阶总场直接离散的源表示仍未被充分解析；增加网格后差异缩小，但仍没有达到可靠空间收敛。新 [代码 `analytic_source_corrected_joule`](../../../sdfmpneo_bfzi/direct3d_galerkin.py) 和 TEAM-7/BFZI CLI 同时输出两种功率，**不让任何一个靠事后拟合“PASS”**。TEAM-7 B 探针现输出解析闭合线源 Biot–Savart 加上数值材料反作用，J 表面探针目前仍只是低阶单元值，不能冒称已获取连续单侧迹。D2 高精度 OPEN。
>


**2026-11-06｜最高优先级：真正掌握成熟物理方法，不追论文表格数值；不恢复旧 G1。**

> **当前结果：一个真实求解三维 Maxwell–MQS 的初步直接参考内核已经修正了闭合端口源加载不一致。离散互易及端口–体积功率恢复；TEAM-7 连续高精度、外域、材料趋肤、源截面及 Elmer 独立实测对照仍待验收，**D2 OPEN**。本文里的数值来自本轮隔离 Python 科学计算环境对与提交代码同结构的原型运行，仓库新增 tests/ 与 examples/ 在用户工作站的完整运行日志仍需独立保存。

## 为什么此前“端口路径精确积分后仍有大功率偏差”？

旧 D2 `solve_aphi` 使用解析入射矢势 `A_src` 的**四面体体积积分**作为涡流受迫项：
`-iω∫σ A_src·v_h`，而后处理通过闭合真实线圈的 `iω∮ A_react·dl` 读取端口。两个泛函在连续无界域经互易式关联，但在现有**低阶有限空间、有限边界、离散场和源的不同投影**下，不能未经验证就互为转置。把路径 Gauss 积分改成四面体相交后的精确积分，只解决读数求积；不能修复体源载荷与物理线流泛函不一致。

原 50 Hz 750 单元 TEAM-7 小盒出现约 **8.221 W 真实路径端口 vs 21.545 W 体积 Joule**，再加密也没有立即闭合。该结果应理解为**旧离散源–端口路径不一致和其他离散误差共同作用**，不能仅凭它断言无限边界或材料本构有错。

## 修正后的 Maxwell–MQS 总场变分形式

保持 `e^{+iωt}`、`mu=mu0`、导体 `σ>=0`、闭合端口与真实绝缘界面，直接求解全场 `A_h` 和导体标量势 `φ_h`：

```text
curl(mu0^-1 curl A) + sigma (i*omega*A + grad phi) = Σp Ip * Jsource,p
div[sigma (i*omega*A + grad phi)] = 0,  inside conducting material
n . J = 0,                                  insulator boundary
n x A = 0,                                  finite outer-domain boundary
E = -i*omega*A - grad phi
```

Nédélec 有向边元 `w_e` 上，按**真实闭合**源曲线 `Γ_p` 定义唯一的电磁源/端口泛函：

```text
L[p,e] = ∮_{Gamma_p} w_e . dl
right_hand_side_edge = L.T @ I
Z_medium = i*omega * L @ A_medium_edge
Z_vacuum = i*omega * L @ A_vacuum_edge
Delta_Z_material = Z_medium - Z_vacuum
```

同一 `L` 既是**强制源加载**，又是**实际端口感生电压读出**。闭合弦段与每个四面体相交，逐段精确积分最低阶 Whitney 切向场；输入的环向方向、真实 BFZI 馈回流与 2742 A·turn 倍数不会被两套不一致的实现改写。由于 `L` 对梯度场的闭合线积分为零，满足离散源连续性。

`A_vacuum` 使用**相同四面体节点与外边界、同一物理源向量**、仅 `σ=0`，扣除细丝在有限盒的真空响应，避免把真空数值自感/镜像量误认为海水或铝板的电磁增量。仅在同一离散上差分；真实铜截面的独立自感、集肤与邻近损耗仍不在该 `Delta_Z_material` 中。

由相同的 Galerkin 变分空间、配对线性源/端口与闭合边界，有

```text
Delta_Z_material = Delta_Z_material.T
P_port(I) = 0.5 * Re[ I.H @ Delta_Z_material @ I ]
P_volume(I) = 0.5 * ∫sigma*|Σp Ip E_p(x)|² dV
P_port(I) = P_volume(I)  (discrete variational energy identity)
```

**必须清楚：此等式是可信实现的必要条件，却不是连续 PDE 精度证明。** 若网格太粗，它仍可能两边完全一致地算错。整个项目仍应依次进行材料区 h 收敛、源弦段与截面加密、外域变大且同步维持近场/外域单元质量，以及 Elmer 等成熟 FEM 的**独立** B/J 测点比较。

## 新增正式实现

- `sdfmpneo_bfzi/direct3d_edge_source.py`：四面体源线裁剪与**真实边元泛函** `closed_edge_load(mesh, source_strands)`。
- `sdfmpneo_bfzi/direct3d_aphi.py`：允许把上述 `edge_source_load=L` 施加到 A–φ 弱式边元右端；不改变旧代码其他理论合同。
- `sdfmpneo_bfzi/direct3d_galerkin.py`：`solve_closed_ports(mesh, f, source_strands)`，同一有限盒分别求总场与真空场，返回 `material_increment_z`、两份全场参考及 `conductor_power`、`joule_density`。
- `examples/bfzi_d2_aphi.py`：真实 BFZI 圆形/圆角方形连续中心线经明确弦段逼近，加上完整真实馈回流，两独立封装（现有 Gmsh OBB 几何）；单位端口及 `I=(1,0),(0,1),(1,±1),(1,i)` 报告复 Z、两源功率。
- `examples/bfzi_d2_team7.py`：默认采用一致源+读出 Galerkin total-A 求解器，**空间磁场是 `B_total=curl A_total`**，不额外叠加 `B_src`；两个频率、4 条官方测点线与完整 JSON 导出不变。
- `tests/test_bfzi_d2_galerkin.py`：两独立端口互易/复相位/真空/TEAM7 基本检查（不以论文或实验数值匹配设强制门槛）。

## 本轮实际数值复算：固定铝板/孔、有限小外盒、50 Hz

相同 TEAM-7 材料几何：294×294×19 mm 带偏心 108 mm 方孔，铝板 `σ=3.526e7 S/m`；线圈仍采用 25×100 mm 截面 `2×2` Gauss、每角 5 弦段、2742 安匝。**不是**官方 COMPUMAG 大外域或 Elmer 正式网格。

| 分级 | 全部四面体 | 导电四面体 | `Re Delta_Z_material` (Ω) | `Im Delta_Z_material` (Ω) | 实际闭合端口功率 (W) | 导体体积 Joule (W) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 750 | 48 | 13.865036 | −32.457745 | 6.932518 | 6.932518 |
| 2 | 2304 | 384 | 13.667611 | −32.185850 | 6.833806 | 6.833806 |
| 3 | 5082 | 1296 | 14.182698 | −32.689346 | 7.091349 | 7.091349 |
| 4 | 9408 | 3072 | 13.474409 | −30.735179 | 6.737204 | 6.737204 |

四档中端口－体积功率的绝对误差分别约为 `3.8e-14`、`9.0e-14`、`7.3e-13`、`1.3e-12 W`。200 Hz 二级网格另得到 `Delta_Z=(22.82585487−159.47118075i) Ω`，端口和体积功率均约 `11.41292744 W`。双端口有非零互耦的 100 kHz 非轴对称导电测试在 `I=(1,0),(0,1),(1,i)` 下功率差 `<1e-16 W`，复阻抗自然互易，未手动对称化。

**必须保留的 FAIL/OPEN：** 四档 `Re Z` 和功率依然明显变化且不单调，不能声称已经达到 TEAM-7 高精度；此处只证明此前大功率差的根因是离散源/端口配对，已通过方法纠正。对照材料趋肤深度：TEAM-7 铝板 50Hz 约 **12.0 mm**，200Hz 约 **6.0 mm**；19mm 厚板必须继续细化。还没有完成源截面/弦段收敛、三档独立高质量外域网格、官方实验数据或成熟 Elmer 仿真曲线对照。本地验证的是隔离原型；GitHub 新增正式测试需工作站运行。

## 实行命令（无 GitHub Actions）

```powershell
git pull --ff-only

python -m pytest tests/test_bfzi_d2_galerkin.py tests/test_bfzi_d2_aphi.py tests/test_bfzi_d2_port.py tests/test_bfzi_d2_mesh_probe.py tests/test_bfzi_d2_team7.py -q

python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 1 --frequency 50 --corner-panels 5 --summary-only
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 2 --frequency 50 --corner-panels 5 --summary-only
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 3 --frequency 50 --corner-panels 5 --summary-only
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 4 --frequency 50 --corner-panels 5 --summary-only

# 工作站已安装 Gmsh 时，启动真实共形 TEAM-7 / BFZI 两独立 OBB
python -m examples.bfzi_d2_team7 --write-geo team7_elmer.geo --outer-box elmer
gmsh team7_elmer.geo -3 -format msh2 -o team7_elmer.msh
python -m examples.bfzi_d2_team7 --mesh team7_elmer.msh --frequency 50 --outer-box elmer --current-sign -1 --summary-only

python -m examples.bfzi_d2_aphi --case bfzi --write-geo bfzi_d2.geo --mesh-size 0.025
gmsh bfzi_d2.geo -3 -format msh2 -o bfzi_d2.msh
python -m examples.bfzi_d2_aphi --case bfzi --mesh bfzi_d2.msh --frequency 200000
```

**验收状态：D2 的“正确离散源/端口实现”已获得有限示例证据；TEAM-7 独立高保真精度、海水无限域和真实任意 SE(3) 双盒的全面高精度仍是 D2/D3 待完成。旧 G1 FAIL；教师冻结。**
