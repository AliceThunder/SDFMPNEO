# P1-C：真实双盒全海水原始 Coulomb L 的本机证据

该目录只保存独立入口 `python run.py --mode bfzi-p1-full-sea --with-tests` 的实际结果，不覆盖原 P1-A/P1-B 解析球与紧支涡流日志。

- [latest_full.log](latest_full.log)：原始 pytest、真实五个 curl 模式的五自项与十互项、外 x 和点心径/角分轴变化，以及诊断。
- [latest_summary.md](latest_summary.md)：代码基线、程序执行状态及摘要。**程序 PASS 不代表连续磁能求积 PASS。**

## 已有运行证据

- 旧对称差分原核（代码 a9f48db）：97 passed，但内 y 独立变化 23.560%、x/y 联合变化 28.437%，与 Taylor 结果相差 80.941%，并存在 98% 大项抵消及负自项。物理 FAIL。
- 新点心直接原核（代码 8c26357，日志提交 02805877）：99 passed，外 x 变化 5.937%、固定细 x 内径向变化 **30.248%**、随后内角向变化 1.936%，联合变化 35.311%，最细原矩阵 skew 1.746%。物理仍 FAIL，最小特征值为正不足以认证。

## 已提交、等待下一份本机日志的算法改进

点心射线求积保持原始未软化 1/r，并沿两只真实独立旋转绝缘盒和外球解析裁剪；对每条真实湿区间的**两端**以 wall_scale × (1,3,9,27,81) 锚定径向分片，解决此前只围绕观察点分片、漏掉实际盒壁尖层的问题。

同时增加 `p1_full_sea_scalar`：每只盒的近面／棱／角共六个真实湿点，直接积分常量测试密度并与独立解析的 `SΩ` 比较。因为常量密度的内径向被积式严格为 r，该诊断只检验**射线裁剪和角向求积**。它不是实际合规电流，也不能当成 G1 PASS。

主实验仍打印 `p1_full_sea_self`、`p1_full_sea_mutual`、`p1_full_sea_inner_axes`、`p1_full_sea_change`。先根据独立标量近壁角向误差决定是否改几何角向分片，再看真实 curl 内径向误差，避免继续盲目增加求积阶数。

运行：

```powershell
git pull --ff-only
python run.py --mode bfzi-p1-full-sea --with-tests
```

本机日志自动 Git 提交推送，**不使用 GitHub Actions**。P1-C 未认证、整体 G1=FAIL，正式教师训练不得开始。
