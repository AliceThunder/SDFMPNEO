# D2｜独立三维 Maxwell–MQS A–φ 参考求解器：核心实现与待验收范围

**2026-11-06；目标分支：research/bfzi-uwpt-electrothermal-surrogate。**

> 当前状态：已有一套**真正求解三维边元 A–φ 混合有限元方程**的独立初始实现，非旧 G1 磁 Green 五模态。针对基础三维封闭源、线性互易和功率的有限网格数值测试已在本轮隔离 Python 环境运行；**尚无 TEAM-7 官方几何/测量比对、无网格与外域收敛证书，不能标记 D2 PASS 或 BFZI 高精度教师通过。** 参见 [物理 V1.0 锁定](PHYSICAL_DIRECT_METHOD_LOCKED.md)、[D0/D1 方法迁移](METHOD_TRANSFER_D1_TO_D2.md)。

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

第二方程是电荷守恒除以 `iω`，使混合块的转置结构清楚；在导电区域与所有绝缘内壁自然满足 `n·J=0`。将 Dirichlet 外边界 `n×A_react=0` 施加于**有限立方体边界**，并不将其称为无穷海水精确条件。当前基线磁导率必须全部为 `mu0`；非均匀磁芯需要另加材料磁化驱动，不能在此对源表达随意修改 `mu`。

元素级：`w_ij = λ_i∇λ_j−λ_j∇λ_i`，`curl w_ij = 2∇λ_i×∇λ_j`，全局边固定升序、局部自动定向；单元 Nédélec `curl-curl` 与复导电质量矩阵、`A–∇φ` 耦合、`phi` 扩散刚度与 `A–∇p` 规范双线性项均按一致四面体形函数组装。源和场的体积分用四点二阶准确四面体规则。用稀疏矩阵完成复线性求解并按物理单位进行均衡。该阶数适合首版数值验证，**不是高频皮肤层无网格高精度的保证**。

## 3. 真实复端口和热源输出

分别求两个单位闭合端口入射源，存储 `E_p(x),J_p(x),B_react,p(x)`；电磁线性组合在同一离散上完成：

```text
ΔZ_sw,pq = iω ∫_Ωsea A_src,p · J_sea,q dV
Pport(I) = 0.5 Re[Iᴴ ΔZ_sw I]
Pvolume(I) = 0.5 ∫_Ωsea σ |Σ_p I_p E_p(x)|² dV
q(x;I) = 0.5 σ |Σ_p I_p E_p(x)|²
```

与源向量势的双线性积分不作共轭（互易复对称）；损耗矩阵 `Herm(ΔZ)` 是 Hermitian。代码同时从**各端口完整总电场**积出 `Pvolume`，没有用二次反作用场平方替代。两种功率关系在同一有限离散源模型下成立，不等于独立连续精度证书。绝对端口 `Z_EM` 还需要真空有限截面自/互感、铜近邻效应和损耗分账；这一版本只给海水**增量** `ΔZ_sw`。

## 4. 本轮实际基础数值证据

隔离 Python 3 环境使用 `numpy/scipy/pytest` 运行已实现的核心求解程序，**5 项核心测试全部通过**，范围包括：导电方块在三维外空气域的非轴对称 `A–φ` 初始受迫电流、`Pport=Pvolume`、双端口 `Z=Zᵀ`、任意复相位下 `Herm(Z)≥0`、真空 `ΔZ=0`、严格闭合方形真实 Biot–Savart 源、3→6 外域网格基础运行。读入一个人工四面体 MSH2.2 并验证物理区域电导率赋值；磁场探针也已单独求值。

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
