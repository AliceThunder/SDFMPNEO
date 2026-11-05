# P1-C 原核 1/r：五个真实 BFZI 全海水 curl 模式的本机日志

本目录属于独立运行入口 **python run.py --mode bfzi-p1-full-sea --with-tests**；
不与历史球解析基准 runs/p1/ 或 G1 总日志相混淆。

**初建此目录时尚无真实 P1-C 本机运行结果。** 旧 runs/p1/
文件包含 94 passed 但 P1-C 主导入 SyntaxError 的失败记录，不是本关结果。
只在本机真正运行并成功推送后才会出现：

- latest_summary.md：完整运行状态、测试统计和全部 P1-C 收敛摘要。
- latest_full.log：五个自项、十个互项、x/y 独立分轴变化与差分/Taylor 交叉指标。

本轮主程序在固定的真实两旋转绝缘盒外海水域，使用现有五个
非源引导 curl 模式与精确双盒标量几何矩 SΩ，
优先采用未软化 Coulomb **对称差分原核算子**；
最后用同一细化规则的原 Taylor 算子交叉核算。
两个算法自身的 x/y 误差或相互接近均不能替代独立真值。
对称差分本来就对称，不得以 skew=0 宣布通过。

应先确认 P1-C 完整执行、分轴自项与互项是否稳定、
有无非正特征值或严重减除抵消，再开展近面/棱角的独立精度基准。
**P1-C/G1 暂不认证，禁止开始正式教师训练。**

运行（用户 Windows 本机）：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

输出由已有本机 Git 机制自动提交推送，**不使用 GitHub Actions**。
