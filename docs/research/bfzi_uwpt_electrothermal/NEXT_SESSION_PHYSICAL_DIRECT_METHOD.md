# 新会话衔接：BFZI 物理直接法 V1.0 已锁定

**2026-11-06 状态：物理定义、方法优先级及 D0–D6 参考验收规范已经提交至分支 research/bfzi-uwpt-electrothermal-surrogate；尚未实施独立三维 A–φ 参考，G1/P1-C 仍 FAIL。**

## 推荐复制的提示词

继续推进 SDFMPNEO 仓库 research/bfzi-uwpt-electrothermal-surrogate 分支的 BFZI 电磁热代理项目。**现在首先完成可信的 B 类物理直接法，不再继续修补旧 G1 五模态 Coulomb 角帽/积分规则，也不启动神经教师。**

请先阅读以下新锁定的权威文件，并遵守原有 BFZI 圆形/圆角方形单根连续线圈、两个独立 SE(3) 实心绝缘盒、真实闭合馈回流、导电海水和 e^{+iωt} 范围：
1. docs/research/bfzi_uwpt_electrothermal/PHYSICAL_DIRECT_METHOD_LOCKED.md；
2. docs/research/bfzi_uwpt_electrothermal/PHYSICAL_DIRECT_METHOD_EVIDENCE_AND_GATES.md；
3. theory/sections/15_physical_direct_reference_lock.tex（已由 theory/main.tex 包含）；
4. docs/research/bfzi_uwpt_electrothermal/PROJECT_PROGRESS.md。

下一步**按 D0→D1→D2**推进。先结合 Shafiei 2025 IEEE Access（DOI:10.1109/ACCESS.2025.3539925）、Wang 2025 Measurement（DOI:10.1016/j.measurement.2025.117224）和标准三维涡流文献，明确并复现共轴/分层海水 Helmholtz–Bessel 半解析基准所需的精确几何、频率、源、边界、公式和输出；随后建立独立、可复现的三维 Maxwell–MQS A–φ 高保真参考与 TEAM-7 等标准算例，并将真实 BFZI 双端口复阻抗、空间 J/B/E、海水 Joule 损耗、互易被动和无限域误差作为验收对象。允许为独立参考使用 FEM；BEM/Green–Galerkin 是待独立验证的高效候选，不能冒充已收敛的参考解。

优先核对原始论文的真实假设和可用数据，**不要未经核对便把共轴解析式推广到任意双封装 SE(3)**。对任何不能从论文或仓库证实的内容明确标注“未证实/待实现”。以物理主算法和必要验证为重心，避免大量与主算法无关的审计脚手架。禁止 GitHub Actions。阶段性进度、公式和参考文献持续提交仓库；需本机运行时给出精确命令并由原有 Git 日志机制保存。

## 固定不变的物理判断

- 现有真实五模态 P1-C/G1 磁矩阵并不等价于完成完整海水受迫电磁求解；
- 真实闭合双端口和有限铜截面损耗不能由任意细丝零半径自积分代替；
- 海水是无限或具有正确远场条件的导电域，绝缘双盒表面 n·J_e=0；
- 端口必须求出复 Z_EM，并用不同相位组合测试海水体积 Joule 功率与端口海水附加阻抗的同源一致性；
- 对称性、正权 4π/S/M、Schur 补被动或两种求积相近，均不足以作为独立物理真值；
- 历史 P0/P1-A/P1-B 的限定通过继续保留；G1/P1-C 和正式神经教师保持未认证/关闭。
