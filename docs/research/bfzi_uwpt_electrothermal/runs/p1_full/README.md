# BFZI P1-C：真实双盒全海水五模态原始 Coulomb L 的本机日志

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
