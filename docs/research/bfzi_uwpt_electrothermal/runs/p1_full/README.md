# P1-C：真实双盒全海水原始 Coulomb L 的本机记录

本目录保存独立入口 python run.py --mode bfzi-p1-full-sea --with-tests 的实际结果，
不与 P1-A/P1-B 解析球或双紧支涡流日志混淆。

- [latest_full.log](latest_full.log)：pytest 和真实五模式五自项、十互项、x／内径向／内角向变化及专项诊断。
- [latest_summary.md](latest_summary.md)：检出代码、时间及运行状态。**程序 PASS 不代表物理 PASS。**

## 真实 P1-C 三轮失败证据

1. 对称差分原核（代码 a9f48db）：97 passed、程序完成，但内 y 变化 23.560%，x/y 联合 28.437%，与 Taylor 差 80.941%，98% 大项抵消，粗规则负自项。物理 FAIL。
2. 点心直接原核初版（代码 8c26357）：99 passed，但盒壁未相对射线端点分片，固定细 x 内径向变化 **30.248%**、最细 raw skew 1.746%。物理 FAIL。
3. **目前最新原始日志**（代码 35895a4、提交 b383da2a）：100 passed，真实盒壁端点分片改进使内径向变化降至 **2.831%**，但内角向 **3.607%**、外 x **1.841%**、联动 **7.349%**、最细 raw skew **2.333%**。真实双盒解析 SΩ 对照在近面／棱／角细角规则残差为 0.05998%／0.02679%／**0.11072%**。物理仍 FAIL。

由于最强模态 L44 占主导，仅改变 x 规则时 L00、L22 自项分别变化约 **54% 和 50%**，但总体 Frobenius 只有 1.841%；不得以总体矩阵变化代替每模态精度。

## 新代码已提交，下次本机运行待证

从每个观察点投影两个实际 OBB 的各八个顶点，在点心极轴横平面获得真实顶点方位，将 [0,2π] 划分物理扇区，每段执行正权 Gauss。这是角积分坐标分片，**不完整解析盒棱在球面极角的轮廓**。

- 原五种 uniform 规则不变；另在完全相同细 x1 和内径向／极角规则下，只改变方位规则得到 corner_both，与 uniform_both 比较。
- 独立解析 SΩ 几何对照新增 aligned_demo（原两盒均单位旋转）及 two_rotated_boxes（TX 绕 x 轴 +0.24 rad、RX 绕 y 轴 -0.31 rad）；分别记录两种角规则在面/棱/角粗细比较。
- p1_full_sea_mode_delta 报告五个 self 相对变化及自能归一化互项变化；p1_full_sea_corner_cross 记录角规则同条件差异。

**纠正以前错误描述：** 原五模式演示函数 build_demo_scene 的两盒旋转矩阵都为单位阵，并非非零旋转盒；新代码标记 five_mode_poses=both_identity。独立旋转双盒的常量标量 Green 对照尚不构成任意位姿磁矩阵认证。

执行命令：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

用户本机自动更新本目录日志，禁止 GitHub Actions。正式神经教师仍不训练。**P1-C/G1=FAIL。**
