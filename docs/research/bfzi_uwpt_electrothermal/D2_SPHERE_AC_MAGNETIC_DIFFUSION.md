# D2 独立有限频率磁扩散：导电球解析解与三维 A–φ 实算

**2026-11-07。状态：新增独立磁扩散空间场验证，D2 仍 OPEN。** 本工作目的是保证 BFZI 物理直接法正确，而不是复现其他作者表格或对数据进行后验拟合。

## 为什么加入这个基准

现有 [三维一致闭合线源 Galerkin](D2_CONSISTENT_GALERKIN_SOURCE_AND_PORT.md) 已使有限离散的源加载和端口积分互为对偶，因此可到浮点精度满足端口—体积功率恒等式。**这项代数一致性不足以证明连续 Maxwell 场正确、高精度或趋肤层分辨充分。**此前真实 BFZI/TEAM-7 粗网格的解析源空间重构也显示严重的 E/B/Joule 误差。需要独立于源路径、端口积分类和论文数值数据的真正有限频率 Maxwell 磁扩散连续解。

选用成熟标准问题：在均匀峰值谐波磁场中放置非磁性均匀导电球（Wiley 已公开方法说明的 3D AC eddy-current benchmark：<https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6088>；原始球扩散与改进球 Bessel 解亦参阅 <https://ed.quantum-bg.org/Knoepfel.pdf>）。该问题有精确复频率响应、无需拟合，检验低频到集肤层的真实空间 E/B/Joule。

## 解析解和相量

取 `exp(+iωt)`，半径 `a`、真实导电率 `σ`、非磁 `μ0`，远场磁场复峰值矢量 `B∞`。令

```text
kappa = sqrt(i*omega*mu0*sigma)
A(x) = h(r)*(B∞ cross x), r=|x|
x_a = kappa*a
F = (3*a/2) / (2 + x_a*i1'(x_a)/i1(x_a))
D = a*a*(F - a/2)
h(r<=a) = F*i1(kappa*r)/(r*i1(kappa*a))
h(r>a)  = 1/2 + D/r**3
B = (2*h + r*h')*B∞ - (h'/r)*x*(x·B∞)
E = -i*omega*A
J = sigma*E (inside), 0 outside
q = 0.5*sigma*|E|**2 (inside)
P_sphere = (4*pi/3)*sigma*omega**2*|B∞|**2 * integral_0^a |h(r)|**2*r**4 dr
```

其中 `i1` 为第一类修正球 Bessel。通过匹配 `A`、`B_n` 与 `H_t` 得到系数，不需要分离误差阈值或拟合参数。解析场在非轴线球内任意三维方向的有限差分 `curl(B)=mu0*sigma*E` 残差约 `10^-9` 相对量级。该纯环流球体例子的电标势可以取零，与之前的**非圆形椭圆柱非零 phi**独立验证正交互补，不能以球例子省略 BFZI 的三维标量电势。

### 实际 FEM 对照

新 `sdfmpneo_bfzi/direct3d_aphi.py` 增加 `outer_bc='prescribed_total'`，逐条 Nédélec 边投影给定物理的**总 A 切向迹**。导电球试验只有一个形式为零的入射源；材料内部方程和数值 E/B/Joule 完全由原来实际的三维 `curl-curl + iωσ` A–φ–规范混合系统求解，解析场**仅作为边界激励及独立输出真值**，不在求解后替换所谓的“准确结果”。

`inscribed_sphere_tetra_mesh` 从规则立方体四面体作径向 cube-to-sphere 映射：外表面节点位于解析球面上，三角形仍是平的，所以数值域为一个内接多面体 `Ω_h ⊂ B_a`。解析场限制到 `Ω_h` 内仍精确满足相同的**内部 Maxwell PDE**。验证时分别比较 `Ω_h` 上的解析 E/B/Joule 和完整球体 `B_a` 的一维精确功率，避免把材料几何误差当成有限元误差。电磁场均采用四点四面体 Gauss 体积积分；最初只在重心取 B 会明显低估空间 B L² 误差，已修正。

## 本轮真实 50 Hz 数值

物理常数：`a=0.05 m`、`σ=3.526e7 S/m`、`f=50 Hz`、`B∞=(0,0,0.001) T`、`μ=μ0`。以下是本轮**隔离 Python/SciPy 重新组装的同一 Nédélec A–φ 弱式并实际求解所得数值**；已保存 [原始结构化证据](evidence/d2_sphere_ac_20261107.json)。仓库新增的组合测试仍需本机直接执行，不将本地独立实现测试偷换为远端完全跑过。

| 四面体数量 | E 体积相对 L² | B 体积相对 L² | FEM 体积 Joule (W) | 同多面体解析 Joule (W) | 同域 Joule 偏差 |
|---:|---:|---:|---:|---:|---:|
| 48 | 76.69% | 59.60% | 0.02098761 | 0.01238652 | +69.44% |
| 384 | 52.22% | 49.92% | 0.02897632 | 0.02152547 | +34.61% |
| 1,296 | 37.69% | 37.91% | 0.02811043 | 0.02431889 | +15.59% |
| 3,072 | 29.33% | 29.84% | 0.02750340 | 0.02539802 | +8.29% |
| 6,000 | 23.96% | 24.42% | 0.02722562 | 0.02591276 | +5.07% |

完整**理想球**的解析耗散恒为 `0.02684694409 W`。但内接多面体体积占真实球体体积的比例仅从 `0.7664` 增至 `0.9879`，而损耗在球表面附近更加集中。因此，不能把 FEM 功率和理想球功率的偶然接近当成“误差很小”；真正用于检验有限元算法的是右侧**同一多面体物理域**解析功率。五级空间场误差明确下降但仍不够小，恰恰显示一阶边元及材料边界三角剖分仍需高质量网格和近界面细化。

## 可复现代码

- [解析场、球体网格、三维 FEM 对照](../../../sdfmpneo_bfzi/direct3d_sphere_limit.py)
- [五级计算入口](../../../examples/bfzi_d2_sphere_limit.py)
- [独立 Maxwell 方程、界面、低感应极限、场收敛测试](../../../tests/test_bfzi_d2_sphere_limit.py)
- [正式物理推导第 18 节](../../../theory/sections/18_d2_sphere_ac_diffusion.tex)

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d2_sphere_limit.py -q
python -m examples.bfzi_d2_sphere_limit --divisions 2 4 6 8 10 --output runs/bfzi_d2_sphere_20261107.json
```

这不是历史 G1 角帽试验；没有引入超椭圆线圈、神经教师或 GitHub Actions。

## 方案中的后继工作

1. **D2 核心精度**：将本次精确球体边界参考用于检查高阶/局部加密边元的空间 E/B 和材料 Joule；保留此前非圆形椭圆柱的非零 `phi` 基准，用于电荷守恒。
2. **正式 TEAM-7**：Gmsh 共形材料网格、真实线圈体积电流截面、独立 Elmer 同外域同边界的 B/J 和频率证据。当前环境无实际 Gmsh/Elmer 可执行工具，尚无 TEAM-7 独立 PASS。
3. **真实 BFZI**：从圆形/圆角方形单根连续闭合绕线到双独立 SE(3) 绝缘盒，明确外域/源弦段/材料皮肤层/铜邻近损耗的各自误差；高保真参考未得到前保持教师冻结。

**D2 部分受限 PDE 基准已有实算；完整 BFZI 物理直接法 D2 OPEN，G1 FAIL。**
