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
