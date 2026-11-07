# D2｜独立三维 Maxwell–MQS A–φ 参考求解器：核心实现与待验收范围
> **最高优先级新结论（2026-11-06，覆盖本文旧“端口—Joule 不一致仍未定位”的叙述）：** 离散源–端口泛函不对偶是关键根因。当前主参考已切到 [一致闭合边元总场 A–φ](D2_CONSISTENT_GALERKIN_SOURCE_AND_PORT.md)，用同一 `L` 强制线源并读出端口，再以同域真空消除细丝数值自感；TEAM-7 四档固定几何的端口/体积功率均在浮点范围一致，旧体积加载源法改作历史方法。**真实空间场收敛与独立 Elmer 测点仍 OPEN**，不能把离散恒等式当连续物理精度证书。
>


> **2026-11-06 D2 本轮新进度（优先于以下旧叙述）：** 闭合端口引入**四面体交段精确 Nédélec 切向线积分**（非固定 Gauss 采样）和\`total_zero\`人工边界条件下**同一有限域真空基线扣除**，海水/铝板增量定义为
> \(\Delta Z_{\rm mat}=i\omega\oint_{\Gamma}(A_{\rm react,\sigma}-A_{\rm react,\sigma=0})\cdot dl\)。
> 已实际运行 50 Hz TEAM-7 保持官方材料几何的 4 档分层网格：750/2304/5082/9408 四面体，闭合源端口功率与导体 Joule 比值从 0.382 变化到 0.723，**仍未闭合、D2 OPEN**。该变化不只属于端口求积，而涉及导体 FEM 细化及有限外域能量通量。将外盒加大但不细化外域网格，已经观察到结果恶化，不把这种操作称“外域收敛”。具体数据和命令详见 [D2 TEAM-7 同域计算记录](D2_TEAM7_OFFICIAL_CASE.md)；精确路径积分及真空基准的数学式见 [第 17 节](../../../theory/sections/17_d2_aphi_reference_and_ports.tex)。本地独立 Python 模块核心及端口专项共 **9 项 pytest 通过**，远端完整仓库组合测试还需工作站复跑。**禁止神经教师、旧 G1、GitHub Actions。**
>


**2026-11-06；目标分支：research/bfzi-uwpt-electrothermal-surrogate。**

> 当前状态：已有一套**真正求解三维边元 A–φ 混合有限元方程**的独立初始实现，非旧 G1 磁 Green 五模态。针对基础三维封闭源、线性互易和功率的有限网格数值测试已在本轮隔离 Python 环境运行；**尚无 TEAM-7 官方几何/测量比对、无网格与外域收敛证书，不能标记 D2 PASS 或 BFZI 高精度教师通过。** 参见 [物理 V1.0 锁定](PHYSICAL_DIRECT_METHOD_LOCKED.md)、[D0/D1 方法迁移](METHOD_TRANSFER_D1_TO_D2.md)。

## 2026-11-07：真实 BFZI 两独立 SE(3) 封装的局部共形网格尺度

原 `gmsh_bfzi_two_boxes(scene,mesh_size_m)` 对两只薄绝缘封装和远场海水使用同一全域近似网格尺度。若为了两只厚仅约 30 mm 的薄盒把全域网格不断减小，外域自由度会急剧增长；反之采用过粗统一网格会使盒/海水界面缺少单元。这不是调端口电阻能解决的问题。

已修改 **真正的 BFZI Gmsh 网格生成主代码** `sdfmpneo_bfzi/direct3d_mesh.py`，新增 `near_mesh_size_m` 和 `near_padding_m`。它保留 OpenCASCADE 真实**两只独立三维有向 OBB**的 `BooleanFragments`，只对背景网格尺寸使用两只 OBB 的世界坐标 AABB 包络 `h_world = |R| @ half_extents`。分别生成 Gmsh 官方 `Box` 尺度场、设置 `VIn=near`、`VOut=far` 和平滑 `Thickness`，用 `Min` 合并。**背景轴对齐尺寸盒不替代真正旋转的绝缘体几何**。两个封装可分别在任意合法 SE(3) 姿态拥有不同位置的近场加密；远海水仍粗分辨。保持 MSH2.2 物理体积标签而禁止 `Mesh.SaveAll`（以免丢标签），导电海水与两个无电导绝缘盒通过材料共形四面体区分。依据 [Gmsh 官方背景网格尺度文档](https://www.gmsh.info/doc/texinfo/gmsh.html)。

```powershell
# Gmsh 已安装的 Windows 工作站，以下命令只是操作入口，当前环境尚未实跑 Gmsh
git pull --ff-only
python -m examples.bfzi_d2_aphi --case bfzi --write-geo bfzi_d2.geo --mesh-size 0.07 --near-mesh-size 0.008 --near-padding 0.025
gmsh bfzi_d2.geo -3 -format msh2 -o bfzi_d2.msh
python -m examples.bfzi_d2_aphi --case bfzi --mesh bfzi_d2.msh --frequency 200000 --output runs/bfzi_d2_localmesh.json
```

**严格限制：** 这是一项真正的网格生成算法改进，不是已完成 Gmsh/Elmer 精度验收；`Box` 背景场只控制局部目标尺寸，不保证导电海水近壁分辨率、铜集肤层或 Maxwell A–φ 物理误差自动达标。必须实际用 Gmsh 生成共形四面体、检查单元质量、加密近场 `h` 及远域边界 `R`、验证两组闭合端口空间 B/E/J/q 与同域 Elmer 独立参考。正式教师仍冻结。

## 1. D2 主求解器代码

- `sdfmpneo_bfzi/direct3d_aphi.py`：三维线性四面体最低阶**有向 Nédélec 边元**求海水/空气总域内的反作用磁矢势；导电域 P1 标量势 `phi`；全域 `H^1_0` Coulomb 规范乘子；导体分量的标量常数自由度逐个固定；复稀疏块系统与有限域边界。
- `sdfmpneo_bfzi/direct3d_mesh.py`：读取物理区域标记的 Gmsh MSH 2.2 四面体、按现有 BFZI 场景独立 SE(3) 旋转/平移生成两实体盒与外部立方体的 `BooleanFragments` 共形分域网格，绝缘实体 `sigma=0`，盒外均匀海水 `sigma>0`。Gmsh 是可选的本地网格工具，**不使用 GitHub Actions**。
- `sdfmpneo_bfzi/direct3d_probe.py`：三维磁场/涡流物理探针。最低阶边元每单元反作用 `curl A_r` 为常值；J 按电场四点均值采样（不是声称近壁点值高阶准确）。允许提供完整线源 `B_src` 得 `B_total`，绝不把 `B_secondary` 冒充总磁场。
- `examples/bfzi_d2_aphi.py`：可运行的 3D smoke 和**真实双连续 BFZI 闭合馈回流**案例入口；复用 `examples.bfzi_mqs_demo.build_demo_scene` 仅取用户当前 BFZI 几何，而**不运行旧 G1 求解器**。源使用 `source_reference.polygonal_port_vertices` 严格闭合线源，逐段解析 A / Biot–Savart；真实圆角通过物理折线弦逼近，须独立加密。
- `tests/test_bfzi_d2_aphi.py`、`tests/test_bfzi_d2_mesh_probe.py`：最少必要的方程、功率、源闭合及网格/测点检查。

以上核心模块**不调用** `sea_basis`、`green_coulomb` 或旧 G1 的 R/L/K 数值装配，也未引入超椭圆线圈。

## 2. 求解对象、变量和离散

在固定线性 `mu=mu0`、无运动、`e^{+iωt}`、闭合源位于绝缘区域的基线下，先由真实闭合导线 `Γ_p` 得到独立**已知入射势**

```text
A_src,p(x) = mu0/(4π) ∮_{Γp} dℓ / |x-r(ℓ)|,   div A_src,p=0 (off sources)
A_total = sum_p I_p A_src,p + A_react
E_sea = -iω A_total - grad φ,  J_sea = σ E_sea
curl μ0^-1 curl A_react = J_sea
div J_sea = 0           in Ωsea
n·J_sea = 0             on each insulating box
```

在整个有界参考计算域 `Q_R` 使用 `A_react∈H_0(curl;Q_R)`，导电海水域使用 `φ∈H¹(Ωsea)`；`A_react` 的 Coulomb 条件通过 `p∈H¹_0(Q_R)` 乘子：

```text
∫_Q μ0^-1 curl a · curl v* + ∫_Ωσ(iω a+grad φ)·v*
  + ∫_Q grad p·v*
      = -iω∫_Ωσ A_src·v*
∫_Ωσ( a + grad φ/(iω))·grad ψ*
      = -∫_Ωσ A_src·grad ψ*
∫_Q a · grad q* = 0
```

第二方程是电荷守恒除以 `iω`，使混合块的转置结构清楚；在导电区域与所有绝缘内壁自然满足 `n·J=0`。以 `secondary_zero` 或 `total_zero` 施加**有限立方体边界**，并不将其称为无穷海水精确条件。当前基线磁导率必须全部为 `mu0`；非均匀磁芯需要另加材料磁化驱动，不能在此对源表达随意修改 `mu`。

元素级：`w_ij = λ_i∇λ_j−λ_j∇λ_i`，`curl w_ij = 2∇λ_i×∇λ_j`，全局边固定升序、局部自动定向；单元 Nédélec `curl-curl` 与复导电质量矩阵、`A–∇φ` 耦合、`phi` 扩散刚度与 `A–∇p` 规范双线性项均按一致四面体形函数组装。源和场的体积分用四点二阶准确四面体规则。用稀疏矩阵完成复线性求解并按物理单位进行均衡。该阶数适合首版数值验证，**不是高频皮肤层无网格高精度的保证**。

## 3. 真实复端口和热源输出（2026-11-06 纠错）

分别求两个单位闭合端口，保存 `E_p(x),J_p(x),B_react,p(x)`；任意峰值相量电流 `I` 通过场线性叠加得到总 E、真实海水/铝板 Joule 热。

```text
ΔZ_port,pq = iω ∮_{Γ_p} A_react,q · dl                # 真实闭合源端口
Z_overlap,pq = iω ∫_Ω A_src,p · J_sea,q dV              # 有限域体积重叠，诊断量
Pport(I) = 0.5 Re[Iᴴ ΔZ_port I]                      # 只有从真实端口 Z 才能称端口功率
Pvolume(I) = 0.5 ∫_Ω σ |Σ_p I_p E_p(x)|² dV
q(x;I) = 0.5 σ |Σ_p I_p E_p(x)|²
```

**重要纠正：** 在有限人工外边界、不同的 `n×A_react` 或 `n×A_total` 条件下，`Z_overlap` 与 `ΔZ_port` 通常存在界面贡献；旧 `Herm(Z_overlap)=∫σ EᴴE` 的内部数值接近不能证明真实端口功率闭合，更不能证明连续无限海水精度。新 `direct3d_port.py` 用边元实际闭合路径线积分抽取 `ΔZ_port`，`direct3d_aphi.py` 保存所有有向边系数，支持 `outer_bc='secondary_zero'/'total_zero'`。`AphiResult.port_power` 现需显式传入真实路径 `port_impedance`；体积重叠诊断通过 `overlap_power` 单独访问，避免误标。

粗网格真实双端口下，`ΔZ_port` 未严格互易且其端口耗散与 `Pvolume` 存在差距，**判为仍缺源/网格/外域收敛**；不得对矩阵强行对称化或夹正使其“通过”。严格边界互易公式、线源的规范不变性和外域边界项见 [正式理论第 17 节](../../../theory/sections/17_d2_aphi_reference_and_ports.tex)。总 `Z_EM` 还需真空有限截面自互感和铜内部/邻近损耗等独立分账。

## 4. 本轮实际基础数值证据

隔离 Python 3 环境使用 `numpy/scipy/pytest` 运行已实现的核心求解程序，**5 项初期核心测试和 2 项闭合端口/边界定向专项测试在隔离代码副本中通过**，范围包括：导电方块在三维外空气域的非轴对称 `A–φ` 初始受迫电流、**体积重叠形式**的内部功率/代数互易检查、任意复相位下同场 Joule 非负、真空 `ΔZ=0`、严格闭合方形真实 Biot–Savart 源、3→6 外域网格基础运行。读入一个人工四面体 MSH2.2 并验证物理区域电导率赋值；磁场探针也已单独求值。

**这些是局部程序级验证，不是模型准确度门槛通过。** 本环境未安装 Gmsh，真实 BFZI 两只旋转盒的 `.geo` 几何虽已能生成代码，但还未用实际 Gmsh 完成网格并执行 FEM，更没有官方 TEAM-7 模型网格或实测数据可比对。

## 5. 本机执行（不自动用 Actions）

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d2_aphi.py tests/test_bfzi_d2_mesh_probe.py -q
python -m examples.bfzi_d2_aphi --case smoke

# 有 gmsh 时，开始真实 BFZI 两独立盒的共形网格
python -m examples.bfzi_d2_aphi --case bfzi --write-geo bfzi_d2.geo --mesh-size 0.055
gmsh bfzi_d2.geo -3 -format msh2 -o bfzi_d2.msh
python -m examples.bfzi_d2_aphi --case bfzi --mesh bfzi_d2.msh --frequency 200000
```

`0.055m` 网格只用于第一次通路，实际盒高度 `0.03m`，**必须显著加密盒壁和海水皮肤层**，否则近壁 `J/q` 误差不可接受。先达到可重现的两个单位端口和电磁/功率闭合，再扫网格、外立方体尺寸和源线段数。源码未创建任何工作流。

## 6. 紧接下来的 D2 真实性工作

- 从 Compumag/COMSOL/NGSolve 等**官方或可交叉核实**参考获得 TEAM Problem 7 的几何/网格和实测 `B_z(x,72mm,34mm)`、`J_y(x,72mm,19mm)`，`σ_Al=3.526e7 S/m`、`2742 AT`、`50/200 Hz`。本仓库的 `read_gmsh22` 支持真实物理域标签；需要独立匹配**真实螺旋/赛道形电流源和铜激励**，而非将现有 BFZI 两线圈冒称 TEAM-7。官方参考：[COMSOL TEAM7](https://doc.comsol.com/6.4/doc/com.comsol.help.models.acdc.multiturn_coil_asymmetric_conductor/multiturn_coil_asymmetric_conductor.html)、[Ansys TEAM7](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/lsdyna_vm/Hlp_V_VM_L_EMAG010.html)、[独立 H(curl) 数学分析](https://www.esaim-m2an.org/articles/m2an/pdf/2015/01/m2an140027.pdf)。
- TEAM-7 初次求解和单点源极限稳定后，按锁定规范执行**三档 h 及三档 R**，保留同一物理问题的表面/外场复 J/B、体积 Joule、互易和无源输出，和**独立**标准实验/工具结果对照。人工边界引入的误差必须独立记录。
- 再进入 D3：现有真实 BFZI 连续圆形/圆角方形双闭合源，按无盒→单盒→双盒同轴→偏移→两独立 SE(3) 的次序。需要把求解器扩展到适应局部皮肤层的网格、真实导体有限截面，最后才有资格给神经教师提供高精度标签。

**结论：D2 核心三维 FEM 已有初始可运行代码，D2 高保真参考认证仍为 OPEN，旧 G1=FAIL，正式神经教师冻结。**
