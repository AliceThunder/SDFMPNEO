# BFZI P1-C：真实双盒全海水五模态原始 Coulomb L 的本机日志

> **2026-11-06 V1.0 方法路线更新（优先于以下历史正文）：** 物理直接法已按 [权威规范](../../PHYSICAL_DIRECT_METHOD_LOCKED.md) 和 [D0–D6 验收](../../PHYSICAL_DIRECT_METHOD_EVIDENCE_AND_GATES.md) 锁定。先独立复现共轴海水 Helmholtz 半解析与三维 A–φ 参考，再对完整 BFZI 复阻抗/感生电流/海水损耗逐项验证。**旧 G1/P1-C 角帽及原核五模态历史试验挂起，G1 FAIL，禁止正式神经教师与 GitHub Actions。** 下文属历史记录，不应继续按其“下一次运行”命令进行数值试错；本轮未改代码、未本机实跑。


- [最新完整原始日志](latest_full.log)
- [最新自动摘要](latest_summary.md)
- 本机独立入口：python run.py --mode bfzi-p1-full-sea --with-tests
- 运行正常、单元测试通过**不代表**真实 Coulomb L 物理收敛。

## 目前最新实测：101 passed，但 G1/P1-C 仍 FAIL

日志提交 aa8ec81，运行代码 dd6b055，101 passed in 43.36s。
历史缩影：
- 97-pass 对称差分路径：存在 98% 大项消去，负自能，真实 L 与 Taylor 强烈冲突，物理 FAIL。
- 99-pass 点心原核初版：内径向 30.248% 分轴差，物理 FAIL。
- 100-pass 盒壁端点径向分片：内径向改善至 2.831%，但内角向 3.607%、联合 7.349%，物理 FAIL。
- **最新 101-pass：** 新增的无 Coulomb 核 Gram G=∫Wi·Wj dx，x0/x1 整体差 2.248%，五个 Gii 差 2.209%–4.101%；同一个 x 改变时，L00/L22 弱自能差 53.978%/49.902%，但整体 L 范数只变化 1.841%。外 x 本身未收敛。

盒角顶点投影角向分片在完全相同细 x/内径向/极角规则下，与 uniform 的 L 差 1.593%、raw skew 从 2.333% 恶化至 2.946%；独立解析 SΩ 标量对照亦不能证明它一致优于 uniform。**不将其升为主算法默认**。当前五模态演示的两只盒 rotation 都是单位阵，非零旋转仅另有独立标量 Green 诊断。

## 本轮新增、下一次本机运行待证

优先定位外 x 的正权体积求积，而不盲目修改内 Green 核：

- x0=(2,3,6,2)、sphere_only=(3,4,8,2)、normal_only=(2,3,6,3)、x1=(3,4,8,3)，逐一分离球图采样和真实盒壳法向积分；每组记录五个 Gii、sphere/shell 各自的贡献比例和分轴 Gram 变化。
- face_tangent：在完全相同 x1 阶数下，仅将真实盒面切向最大子片宽改为实际五个 W 模式最窄 Gaussian 窗宽的一半，检验盒面切向欠解析，而非盲目提高全部阶数。
- 每组与**精确物理体积** V=4πR³/3−ΣVbox 和**精确中心二阶矩** M2=4πR⁵/5−ΣVbox(|cbox−couter|²+Σhi²/3) 比较。这两个公式包含实际双盒尺寸和位姿，不依赖 Green 内核。
- 另外两套固定 scrambled Sobol seeds 13、37，在相同真实盒壳/球分区独立抽样、比较真实电流 G；样本间分散度**不是严格误差上界**。

只改变外部积分，不更改 BFZI 几何、物理电流、原始未软化 1/r 核、原均匀内积分实验和日志名称。下一份日志优先看 p1_full_sea_outer_gram_axis / p1_full_sea_outer_gram_qmc 以及解析几何矩误差。

本机命令：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

日志由 Windows 本机自动 Git 提交，**不使用 GitHub Actions**。P1-C/G1 尚未认证，不启动神经教师训练。


## 第五次原始日志（cb02be3，代码 1661bd5；101 passed）与下一次物理 x 候选

最近一次完成的日志提供了与 Green 内核无关的实验证据：
外球球图从 x0 单独变化到更细规则时，实际海水解析体积误差从
1.7283% 降为 0.0877%；盒壳法向单独细化后仍为 1.7259%。
球图对无核 Gram 的**独立贡献**变化 15.705%，
盒壳法向对整个 Gram 的影响 0.873%，
盒面切向物理子片对弱自项 G00/G22 的影响各为 2.186%。
独立 Sobol 双种子最坏自项变化 4.044%，不构成误差上界。
真实原核 G1 仍 FAIL。

此次代码新增真正参与**五模态原始 1/r Green 磁能装配**
的外 x 候选规则 shell_phys：在正权球/盒壳分区上，
球域按真实两盒和扩展盒的角度/径向几何断点分片，
最大球径向片宽为实际高斯最窄窗宽，
盒面切向片宽为其一半。
此法不改变原始 BFZI 物理问题或内 Green 积分；
物理盒棱的球面投影曲线仍须独立验收。

新的日志会出现：
- sphere_radial、sphere_angles、sphere_phys、
  phys_tangent 四组无核 Gram 的独立真实体积和二阶矩对照；
- p1_full_sea_physical_outer_cross：
  在相同内 Green 规则下 phys_x_both 与旧 both 的实际五模态 L 对比；
- p1_full_sea_mode_delta：
  逐模态自磁能和按自能归一化互耦，避免最强 L44 掩盖问题。

本机：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

新物理规则尚待新本机日志，**不自动视为已收敛。
G1/P1-C=FAIL，不启动教师训练，不使用 GitHub Actions。**


## 第六次原始日志（8b743b0，代码 946c10c）：102 passed，新的物理 x 仍非参考解

最新真正实跑显示，phys_x_both 外 x 从 4,336 点增到 18,816 点，
原矩阵偏斜从 2.333% 降至 1.579%，
但五模式弱自项 L00、L11 的相对变化分别
高达 12.166%、22.944%；整体 Frobenius
仅改变 0.543%，是高能 L44 遮蔽了弱模态误差。
解析真实海水体积误差由旧细规则的 0.08525%
增至新规则的 0.10368%，没有单调改善。
**P1-C/G1 继续 FAIL。**

本轮新增真实原核点心湿射线的解析
SΩ 几何矩控制变量。对固定观察点、原始
不软化 1/r、原来完全相同的射线权重，
使用 U_cv=ΣqW(y)+[SΩ(x)−Σq]W(x)，
其中 SΩ 来自真正外球减两独立旋转绝缘盒
的解析 Green 标量势矩形面闭式积分。
对常量向量电流源恢复严格解析内势；
对一般变化的 curl W 尚需独立收敛。
这项校正使 L 的增量严格对称，
因此不能消除原矩阵的绝对反对称误差。

此前被解析标量真值否证的完整
corner_both 昂贵五模态试跑停止重复；
aligned_demo/two_rotated_boxes 的低成本
解析几何对照保留。新
scalar_control_both 在和旧 both 同样
外 x、内径向、极角、方位规则下
比较完整五自项、十互项及 raw skew，
与原 corner_both 大致同类计算量。

无核 Gram 继续新增两项真实物理分轴：
phys_tangent_half（盒面最大片宽由最窄
Gaussian 窗宽 1/2→1/4）；
phys_angle3（同径向/盒面切向，球图极角、
方位角各真实实体扇区 Gauss 2→3）。
这些案例只计算原五模态的无核 G、
精确海水体积和二阶矩及 sphere/shell 贡献，
不增加无根据的多轮重型磁核实验。

本机命令：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

新控制变量与新单轴外 x 精度尚待本机实跑。
**P1-C/G1=FAIL，正式教师训练关闭，禁止 GitHub Actions。**


## 第七次真实日志（f5f66cc；代码 7847765）与下一次仿射 Green 主算法

本机第七次原核 P1-C 五模式计算实际完成，
**103 passed in 46.35s，但磁能精度未认证**。
同 x/内径/角度下，精确标量 Coulomb
几何矩 SΩ 修正使 L00、L11 分别变化
7.176%、31.483%：弱模式内积分存在明显
常量分量求积缺陷，整体 L 只变 0.938%
的范数不能隐去它们。
原矩阵归一化偏斜 2.333%→2.352%
不代表数值精度改进；零阶控制变量
的原矩阵改变量严格对称。

**确定收敛的积分轴：** 实际盒面切向
最大 Gauss 子片宽从 4.5 cm→2.25 cm，
五模式无核 Gram 整体变 0.000726%、
最大自项变 0.00626%，不再无目标加密。

**尚未稳定的积分轴：** 真实球图几何角片
每片 Gauss 2→3，五模式无核
G00/G11 变化 2.324%/2.184%，
虽然解析湿域体积误差从 0.10368%
降至 0.002744%。故下一轮增加单独
phys_angle4_vs_phys_angle3（只算无核 Gram，
并保留独立解析体积与二阶矩检验）。

**本轮新增真实五模式核心实验：**
原点心湿射线 1/r Green 内势先有
SΩ(x)=∫Ω 1/|x−y| dy 的解析零阶校正，
再加入 MΩ(x)=∫Ω(y−x)/|x−y| dy
及实际非源引导 curl 模态解析 Jacobian J，
构造严格等价的仿射控制变量：

    U_aff(x)=ΣqW(y)+(SΩ−Σq)W(x)+J(MΩ−Σq(y−x))

对于任意全域仿射向量源和合法双旋转盒，
其 Green 内势严格由 SΩ、MΩ 恢复；
对实际空间变化 W 的残项需独立收敛。
新增的物理双盒各自非零旋转测试，
以及直接在旋转盒体积做三维 Gauss
求积与闭式 S/M 比较的独立几何矩验证，
避免将同一解析公式自我对照。

下一份日志运行时在相同 x1、
同一内径/极角/方位下输出三组真实
原核矩阵：both、scalar_control_both、
affine_control_both，比较全五个自项、
十个互项、原矩阵偏斜和逐模态相对差。
已连续重复并证伪的 18,816 x 节点
phys_x_both 完整 L 计算移出默认循环，
已测结果仍在旧日志与看板可追溯，
其外积分正权几何候选和无核 G 留存。

本机执行：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

**此次新增仿射校正尚待本机实跑。
P1-C/G1 仍 FAIL，磁教师不启动，
禁止 GitHub Actions。**


## 第八次本机日志 238f32d（运行代码 d7c59a5）：105 passed，仿射外推严重失败

105 项本机 pytest 通过，真实双盒五 curl 模态原核
L 程序执行完毕，**但 P1-C/G1 仍 FAIL**。
全域一阶仿射 Coulomb 几何矩控制变量
相对于零阶标量修正导致原矩阵范数
相对变化 **350.4647%**，
L00 自磁能的相对变化 **3541.339%**，
强模态 L44 也增加约 345.6%。
raw skew 虽降至 0.759%，绝不能视为算法精度提升。
仿射恒等式及独立非零旋转两盒的解析矩测试通过，
但把真实贴壁 Gaussian-curl 的局部大 Jacobian
用于整个海水域的一阶 Taylor 外推，
会加剧数值余项抵消，不能成为生产 G1 参考。

外 x 无核 Gram 的球域角 Gauss 3→4
整体仍差 0.5383%，单独球图差 5.3395%，
弱模式最坏自项仍差 1.1768%。
解析真实海水体积误差降低到 0.001616%
只能认证几何低阶矩，不能认证真实 W 加权积分。
盒面切向加密的 Gram 差已经远小，暂停该轴。

**这轮新原核主算法：**
不再重复已经严重失稳的全五模态仿射计算。
保留原真实域、未软化 1/r、
精确双盒湿射线和贴壁径向求积，
新增对每个外 x 的**解析几何矩引导的内角向选择**：

    S_ray = 0.5 ΣωΣ(b²-a²)
    M_ray = (1/3)ΣωuΣ(b³-a³)
    eS = |S_ray-S_exact|/S_exact
    eM = ||M_ray-M_exact||/(S_exact*max_ray)

所有射线 [a,b] 是真实球与双独立旋转盒的
物理湿区精确裁剪。几何矩角向误差过高时，
仅对相应 x 的极角及方位角点数各加倍；
仅在选出的角方向上评估昂贵真实 curl 模态。
首轮目标 0.25%，最大附加一层；
超出预算仍不满足时取实际尝试过的
最小误差层，并将未达到目标的点数
明确写入日志，不伪装为严格收敛。

下一份日志将新增：
- moment_angle_both：同旧 x1、
  同真实内径向 Gauss、同五个原始模式，
  与已实跑 both 和 scalar_control_both
  比较全部自/互磁能及 raw skew；
- p1_full_sea_moment_angle_quality：
  精确低阶几何矩筛选的观察点总数、
  加密点数、unresolved 点数、
  最大 S/M 相对几何误差。
几何低阶矩即使通过，也不意味着
真实非线性 curl 磁能已达到高精度。

本机命令：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

本轮新方案尚未本机实跑，
**G1/P1-C 仍 FAIL，神经教师关闭，
不使用 GitHub Actions。**


## 第九次本机 106-pass（a8d664e；代码 40c4890）：几何矩达标，真实磁能未达标

最新完整原始日志： [latest_full.log](latest_full.log)，
**106 passed in 43.91s，但 G1/P1-C 仍 FAIL**。

真实外 x=4336 个观察点，
几何矩选角规则使其中 1296 点（29.89%）
加密，unresolved=0；
最大解析标量几何矩误差 0.24912%，
一阶几何矩缩放误差 0.13166%。
但该规则与原 scalar_control_both 相比，
实际五模态 L00/L11 相差
**7.6446%/10.2046%**，
即使 raw skew 下降到 1.091%，
仍不代表磁能收敛。
几何 S/M 两种低阶矩不能作为
真实 Gaussian-curl 磁能误差上界。

### 本轮新增源模式中心径向求积算法

在原始 1/r、x 点心球角、
真实盒壁端点/贴壁尺度径向 Gauss 上，
新增独立可选的 mode_centered_radial_panels，
对实际 BFZI 模式 Gaussian 窗
中心 c、正定指标 G 沿射线计算

    r_peak = (uᵀG(c−x))/(uᵀGu)
    width_ray = 1/sqrt(uᵀGu)

只在真实模式 Gaussian 峰可能影响的
方向上，在物理湿射线区间内添加
r_peak±3、±2、±1、0 倍 width_ray
的 Gauss 子区间断点，
严格保留原球减双独立 SE(3) 实盒
湿区、物理 current、壁面端点、
未软化 Green 与正权 Jacobian。
这解决的是固定盒壁断点可能漏掉
位于射线区间内部的窄源峰，
并非用高斯替代原核或实际电流。
横向 3 倍窗宽筛选为效率启发式，
不声称为严格误差界。

**两项测试：**
- 不依赖实现的 erf 解析
  窄三维 Gaussian 球内射线积分，
  验证旧低阶 Gauss 漏采的
  模态源峰能否被新分片准确解析；
- 真实双盒小空间 curl 组装接口
  对应 Gaussian mode 的通路。

**下一次实际五模式单轴试验：**
保留原 moment_angle_both。
新增 mode_radial_both，
它与前者使用**完全相同**的 x1、
双盒闭式几何矩选出的角方向、
原内 Gauss 阶数、五 curl 模态，
只新增原模式 Gaussian 投影径向子片。
打印真实五自项、十互项、
归一化 raw skew、逐模态相对差，
并核对两次的角向筛选统计完全相同。
如果差异很大，应继续建立
径向源模式细分收敛层级；
如果很小，下一步重点应转为
真实 W 加权角向源分辨率检查，
而不能仅凭几何矩 0 unresolved。

本机执行：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

本轮新增算法尚待真实本机日志。
**P1-C/G1=FAIL，正式磁教师关闭，
不使用 GitHub Actions。**


## 第十次本机日志 1da0e43（代码 ca263e6；107 passed）：径向源窗约 1%，新增真实源中心角帽

本次完整五模态 G1 程序运行成功但物理精度 FAIL。
保持同一 4336 个 x、同一 S/M 几何选角
（1296 个点加密，unresolved=0）
与相同真实电流模式，仅增加 Gaussian
模式沿湿射线峰中心径向 Gauss 子片，
磁矩阵整体变化 **0.960502%**，
最大自能变化 **0.960534%**，
L00/L11 分别变 **0.470693%/0.596213%**；
raw skew 由 1.091112% 到 1.090614%，
无法解决之前 7%–10% 的角向弱模态变化。

本轮新候选 source_cap_both：
按五个真实高斯模式中心 c 和
旋转 SPD 空间宽度，在从每个 x
望向窄远端模式的球面局部角帽内
增加正权 Gauss/周期方位节点，
用背景 H0=1/(1+Σh) 和
各局部帽 Hm=h_m/(1+Σh)
做严格连续正权单位分解。
角帽重叠时没有物理域重复积分；
原五模式、外 x、实际湿射线裁剪、
未软化 1/r、贴壁和模式径向片均保留。
有限角权不保证总和正好为 4π，
新增 p1_full_sea_source_cap_quality
打印球帽数、额外方向数和最大总测度误差。

独立测试：窄三维 Gaussian 的
Newton 势具有全空间解析参考
π^(3/2)w³ erf(|c|/w)/|c|，
固定同一 radial source windows，
要求局部角帽恢复原粗全球角规则
遗漏的角向窄峰；另覆盖真实双盒
两 curl 模态的小空间角帽装配。

下一次实跑严格比较：
mode_radial_both 与 source_cap_both
使用相同 x1、S/M 选中背景角、
五真实模式、模式中心径向 Gaussian
窗口，只增加球帽 angular charts。
日志输出真实五自磁能、十互磁能、
raw skew、各模式变化、
正权角测度误差。新方案尚待本机实跑，
**P1-C/G1 FAIL，教师关闭，不使用 GitHub Actions。**

本机：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

## 24. 第十一次本机：107 passed、1 failed，源角帽单测比较了错误的物理量

最新日志提交 7906a58（运行代码 cd82e4f）：**107 passed，1 failed；由于单测失败，本次五模态主求解没有运行**。所以不能把上轮 source_cap_both 的候选当作已经实算，也不能用这次日志对其真实磁矩阵效果作判断。

失败测试为 test_source_centered_caps_resolve_narrow_angular_gaussian_against_exact_ball：它拿远离观察点的窄 Gaussian 电流的 **自磁能 Lgg** 同有限的 **点心 Green 内势 Ug(x)** 比较。因为在观测点的 Gaussian 值指数微小，Lgg=μ0/(4π)·Wg(x)·Ug(x) 自然趋零，即使 Ug(x) 已精确积分。原测试得到约 1.1343e-229，而错误真值约 1.5529e-11，属于测试定义混淆，**不足以证伪或验证真实角帽算法**。

本轮修复了主接口：assemble_coulomb_radial 增加可选 return_point_potentials=True，直接返回每个外观察点和源模态的**无 μ0/(4π) 前因子的原核 Green 内势**；既有对称磁矩阵 L 返回保持向后兼容。新解析测试比较 μ0/(4π)·Ug(0) 与独立 Newton Gaussian 解析真值，不再误拿 Lgg。当前真实双盒五模态及几何磁能仍未认证。

同时修复 examples/bfzi_run_record.py 中 pytest 终行解析：失败日志的完整 **1 failed, 107 passed** 不再被自动摘要截成只有 **107 passed**。原始日志仍完整保留以便追溯。

**下一次 Windows 本机执行：**

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

先确认 corrected Gaussian source-cap 解析单测通过，再获取新的 source_cap_both / mode_radial_both 五模式单轴对照、局部帽角测度及独立 S/M 几何矩误差。**P1-C/G1=FAIL，正式神经教师关闭，不使用 GitHub Actions。**


## 第十二次实跑（代码 78c8353，日志 53f635d）：108 passed，但未配平角帽显著破坏解析几何矩

同一五模式原核 Green x/径向规则，新增局部模式方向角帽后：L 整体相对变化 **12.859%**，弱自项 L00 **+105.634%**、L11 **+78.354%**，强自项 L44 **−12.661%**。同时，角向正权和相对完整 4π 偏差最高 **4.9684%**，真实外球减两盒解析 S/M 几何矩误差变为 **5.7221%/2.6418%**，比原几何选角的约 0.249%/0.132% 大幅恶化。raw skew 降至 0.905% 不能证明解更好。**108 项测试通过只代表程序完成，P1-C/G1 依然 FAIL**。

**本轮新实现：** 固定原背景和模式球帽真实角方向，在每个 x 的精确湿射线端点直接形成 A_k=Σ(b²−a²)/2、B_k=u_kΣ(b³−a³)/3，用正权相对熵指数倾斜修正权重，约束 Σw=4π、ΣwA=SΩ、ΣwB=MΩ（三分量），在规范化矩残差低于 2e-7 时接受。如果固定方向不存在数值可行的正权解或 Newton 停滞，则拒绝该观察点的角帽，回退到原先经过双盒 S/M 几何筛选的背景正角方向；记录 moment_fit_accepted、fallback_to_geometry_background、最坏拟合残差以及最终实际使用规则的 4π/S/M 误差。原始未软化 1/r、真实旋转盒湿域、wall_scale 和实际 Gaussian-curl 都不改变。

**新增必要测试：** 球心、单位外球的独立解析 S=2π、M=0 和 4π 总测度；真实双盒小空间同时接通正权配平/回退入口。匹配低阶矩仍无法证明非线性真实 curl 模态已经收敛。

下一次 Windows：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

本轮新代码未实跑；**P1-C/G1 FAIL，正式磁教师关闭，禁止 GitHub Actions。**


## 第十三次本机日志 634535f（运行代码 84a730a）：109 passed，源角帽几何矩已匹配，实际磁能尚差 6.96%

本次实跑 109 项 pytest 全部通过，并计算完真实双盒五模态原始 Coulomb L。
角帽正权矩匹配在提出的全部 **3500** 个观察点成功，**回退 0**；
最终角权相对 4π 最大误差 **3.33e-16**、
真实双盒解析 SΩ 最大误差 **0.0024911893**、
MΩ 缩放误差 **0.00091115346**。
这些数据证实先前未配平角帽的 4.97% 球面测度和 5.72% SΩ 误差已消除。

但 source_cap_both 相对同 x、同实际模式 Gaussian 径向分片及背景角的 mode_radial_both：
**全矩阵相对变化 6.9584%**，自项 L00 差 **1.9372%**、
L11 差 **6.5841%**、L44 差 **6.8950%**。
raw skew 1.0906%→0.7038% 不能认证物理磁能精度。
守恒解析 S/M 是必要不充分条件，必须直接做真实 curl 角向收敛。

**本轮新增真实五模式单轴实验：**
已有 source_cap_both 的局部源帽 Gauss 阶数为极角/方位 (4,10)，
新增 source_cap_refined_both 设置局部源帽 (6,14)。
保持原 **4336** 个外 x 点、原自适应低阶几何背景规则、
原径向模式窗口、原实际五个无源引导 curl 模式、
真实双独立 SE(3) 实盒外海水、未软化 1/r 完全不变。
两级局部帽各自在自己的节点上正权拟合相同 4π/S/M 解析真值，
避免因改变局部 Gauss 节点再次混入几何矩误差。

新日志包含 p1_full_sea_source_cap_refinement_quality、
p1_full_sea_source_cap_refinement_cross 和
source_cap_refined_vs_source_cap 五自十互差异。
独立 Gaussian Newton 势测试和单位球的 4π/S/M 正权测试
也同时调用细化球帽。新代码尚待本机实跑。

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

**P1-C/G1 继续 FAIL，正式神经教师不可开启；禁止 GitHub Actions。**


## 第十四次本机日志 4239872（运行代码 fb4ff27）：109 passed，但弱耦合角向反号

最新实跑对相同真实 x/双盒海水湿射线、五个固定无源引导 curl 模式、源窗口径向分片和相同解析 4π/SΩ/MΩ 正权拟合，仅把局部角帽阶数从 (4,10) 提升至 (6,14)，得 L 整体相对变化 **0.727321%**、最坏自能变化 **0.720073%**；弱自能 L00/L11 各变化约 **0.02356%/0.02491%**，但自归一化互耦最大变化 **2.77845%**，L02 从 **−1.5761e−11 → +8.5036e−12**，符号仍不稳定。两级局部帽均 3500 accepted、0 fallback，最终 4π 误差约 4.44e−16、解析 S/M 最大误差 0.2491%/0.0911%；总体 raw skew 从 0.7038% 降至 0.4993%，但强模态可以掩盖弱互耦数值问题。**P1-C/G1 继续 FAIL。**

**本轮实质代码：**
1. 不再为两个完全相同的实际 Gaussian 中心和 SPD 指标、但不同真实 curl 切向方向，重复生成并评价同一批球面角帽。合并的是计算节点，不是物理基；k 个重合帽的分区函数按 k·h 精确保留，因此和原重复角帽的离散权重之和逐点一致。新增直接解析节点权重测试。
2. 在原 (6,14) 局部帽基础上新增方位角四分之一步长相位，source_cap_phase_shift_both；保持真实几何、所有模式、所有 x/径向节点、同一背景角向规则和解析正权 S/M 约束，单独检验周期方位角别名误差。原角帽物理轴线与开角不变。
3. 对原始未对称 Green 装配矩阵新增逐模态对自能归一化的互易性差，输出最差模态对，避免强 L44 掩盖弱 L02 等的原矩阵不对称。

**下一轮日志：** source_cap_phase_shift_both、p1_full_sea_source_cap_phase_cross、source_cap_phase_shift_vs_refined、p1_full_sea_pair_reciprocity。优先核对 L02 的幅值与符号、正权配平接受/回退和去重后的角方向数。当前新代码尚待本机验证。

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

G1/P1-C FAIL，正式磁教师关闭；不使用 GitHub Actions。


## 28. 本轮远端尚无新相位日志：聚焦重算当前真实五模态问题，保留历史完整模式

2026-11-06 UTC 本轮检查分支 HEAD e17afa9c，远端 [latest_full.log](runs/p1_full/latest_full.log) 仍是运行代码 fb4ff27、日志 4239872 的 109-pass 历史实跑；没有 source_cap_phase_shift_both 或 p1_full_sea_pair_reciprocity，因此**不能宣称重复角帽合并、方位相位与弱互耦互易诊断已经本机验证**。最后一次可信的单轴结果仍是局部角帽 (4,10)→(6,14) 使 L 整体相对变化 0.727321%，强自能 L44 变化 0.720073%，L02 符号翻转。

**本轮改进计算工作流（不改任何物理源/电流或 Green 核）：** 原来的 P1-C main() 每次重复历史 x0/x1、内径/角、scalar-only、无源帽及外 x Gram 多轴试验；这些旧指标已存于历史日志，每轮重复耗时。新增 main_focused()，由默认 run.py --mode bfzi-p1-full-sea 调用，**只重算两个真实原核五 curl 磁矩阵**：同一个 x1=4336 点、背景 S/M 选角、同一个 (6,14) 真实模式帽、相同源窗径向 Gauss、同一个正权 4π/S/M 约束，唯一不同的局部帽方位相位 δ=0、δ=0.25。两种均从头求真实 L，不读取缓存；新增在简洁日志中直接输出 L02、L13、L14 及全部十对原未对称互易性误差。注意这不是少算原核精度，而是**停止反复计算已经有历史结果的其他数值试验**。

**全部历史试验没有删除：** 使用 --p1-full-history 仍执行原 main()（完整外 x Gram、背景角、历史内径/角和所有源帽变体）。两种模式均可 --with-tests，并由本机原有 Git 机制自动发布 latest_full.log 和 latest_summary.md，禁止 GitHub Actions。

下一次本机默认：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

需要完整历史分轴时：

    python run.py --mode bfzi-p1-full-sea --p1-full-history --with-tests

**本轮新聚焦代码尚未本机实跑，P1-C/G1=FAIL，正式磁教师关闭。**


## 29. 第十五次本机 110-pass：同阶源帽方位相位引起真实互耦反号，新增双真盒角投影分段 Gauss

最新完整本机日志提交 5c84a46、运行代码 c8e5aca，**110 项 pytest 全通过**。新聚焦执行确认 4336 个外 x、两组完整真实五模态未软化 1/r Green：同一个源帽 (6,14)，同一个径向 Gaussian 源窗和解析 4π/SΩ/MΩ 正权拟合，仅将方位节点平移 0.25 步。L 总体差 **0.725294%**、最大自项差 **0.721760%**、自归一化互耦最大差 **2.419479%**。L02 从 **+8.503606326e-12** 变为 **−1.263133158e-11**。最差未对称模态对 (1,4) 互易误差 **2.45056%→3.36955%**。即使 3672 个独立角帽、3500 accepted/0 fallback、真实 4π/SΩ/MΩ 低阶矩不变，**电流加权方位高阶离散仍然失败**。较小的 (0,2) 成对互易性不能抵消 L02 反号的不收敛事实。

**本轮主数值候选：** 在原 Gaussian-curl 模式中心局部角帽的切平面中，投影两个**真实独立 SE(3) 绝缘盒**的各 8 个顶点，以物理方位角作为全 [0,2π] 的扇区边界；每个扇区使用**正权二阶 Gauss**，局部极角仍 6 阶。该方案直接复用已有双盒顶点投影函数 _bfzi_projected_box_azimuth_rule，实际点心湿射线仍由原始未软化 1/r、真 OBB 外海水射线求交函数严格确定。新角点仍逐 x 重新拟合原来相同 4π、解析 SΩ/MΩ 正权矩；不修改 curl 模式、源项、位姿及外 x。

新独立测试检查：(a) 双盒角投影扇区 φ 正权和为 2π，所有源帽方向单位化；(b) 即使 φ 分区受两个真盒决定，独立单位球仍达到 S=2π、M=0、球角 4π 的正权配平；(c) 实际双盒小空间 curl 直接装配仍满足几何矩接受/回退限制。

**重要限制：** OBB 表面完整射线切向曲线同时取决于极角、方位角；顶点方位扇区**不是严格完整球面遮挡解析分割**，也不是 G1 连续参考真值。若本轮实测仍有高互耦误差，应考虑完整盒面切线曲线或独立外 x 收敛，不再反复降低 S/M 低阶几何阈值。

下一次默认聚焦实验只重新计算两个完整真实五模态 L：source_cap_refined_both（6×14 周期方位）与 source_cap_box_corners_both（极角6、真实双盒顶点方位扇区 Gauss2），输出 p1_full_sea_source_cap_corner_cross、各模式自互磁能、十对原核未对称互易性及真实 S/M 几何误差。既有全部历史实验仍可通过 --p1-full-history 运行。

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

**本轮新增候选尚待 Windows 本机实跑；P1-C/G1=FAIL，正式神经教师关闭，不使用 GitHub Actions。**


## 第十六次 112-pass：真实盒角扇区没解决弱互耦；新增逐极角盒边解析角断点

代码 1956ed3、日志 7bc9266：**112 项测试通过、五真实 curl 磁矩阵完成，但 P1-C/G1 FAIL**。双盒角点方位扇区 Gauss2 相对原局部均匀 6×14，虽然全 L 相差仅 **0.065772%**，弱 L00 仍相差 **0.730025%**、自能归一化互耦最差变化 **2.783924%**，L02 从 +8.503606326e−12 反号为 −1.581488039e−11。最差原核模态对互易性 2.45056%→2.61120%。两规则 3500 accepted、0 fallback，独立 4π/S/M 矩误差一致，物理误差不能归因于旧低阶几何矩失配。

**本轮代码：** 在源角帽每个极角 mu 上，对两真实独立 SE(3) 盒各 12 条实际棱边，精确求 (edge(t)−x) 与极轴夹角的带符号二次方程交点；该交点方位新增至上一轮保留的全部盒角方位断点。每个新增扇区继续正权二阶 Gauss，原局部极角六阶、五 curl 模式、原始 1/r、真实湿射线、Gaussian 模式径向面板、同一 x1=4336 和解析 4π/S/M 正权拟合均不变；不引入 FEM 网格。单元测试包含真实上盒 z=.115、y=.1、mu=.67 下处于盒角扇区**内部**的盒边交点，以及 2π/4π/S/M 和双盒小空间 curl 通路。

**下一份默认真实五模态主试验：** source_cap_box_corners_both（原角点分段）对 source_cap_box_edges_both（额外逐 mu 边交分段），报告五个自能、十个互耦、L02/L13/L14、全部原矩阵对称性误差、角方向数量及 accepted/fallback/最终 S/M。新代码尚待本机实跑，不作为磁教师。

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

需全部历史分轴时附加 --p1-full-history。**P1-C/G1 FAIL，教师关闭，不使用 GitHub Actions。**
