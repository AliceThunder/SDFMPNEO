# BFZI 最近一次本地运行日志

此目录由用户本地的 `python run.py --mode bfzi-convergence ...` 或 `python run.py --mode bfzi-g1-foundations ...` 自动维护，不使用 GitHub Actions 或云端后台任务。

**以下两个文件已在用户本机实际生成并成功推送 GitHub，后续运行自动覆盖更新：**

- [latest_summary.md](latest_summary.md)：精简的嵌套空间对照表、误差上界、磁积分 A/B 和最终门槛；供 GitHub 浏览和后续会话快速读取。
- [latest_full.log](latest_full.log)：原始 stdout/stderr，包括完整场探针数组和自适应迭代信息；供复查问题和二次分析。

自动流程：`run.py` 接管 BFZI 收敛实验输出 → 屏幕只显示关键物理指标 → 写入上述文件 → 本机 `git add` 指定日志文件、`git commit --only`、`git push origin HEAD:refs/heads/research/bfzi-uwpt-electrothermal-surrogate`。使用的是本机已配置的 Git 提交身份与推送权限。只提交上述两个日志文件，不把代码、数据集或其他暂存内容混入提交。运行失败也保存并尝试推送日志；推送失败会明确提示，原始文件仍保留本地。

```powershell
git pull --ff-only
python run.py --mode bfzi-g1-foundations --with-tests
# 可选旧版外域算子归因（不作为主要 G1 路线）：
# python run.py --mode bfzi-convergence --basis-domain-ball --basis-ball-targeted --basis-domain-far --with-tests
# 只保留本地，不推送：
python run.py --mode bfzi-convergence --basis-ball-targeted --no-publish-log
# 如需看到全部原始标准输出：
python run.py --mode bfzi-convergence --basis-ball-targeted --verbose-log
```

**最近本机实测（2026-11-04 UTC，代码基线 `cd73a65`）：** `62 passed in 43.75s`，新的 `bfzi-g1-foundations` 三路物理基准首次正常运行：5 模态原始 Coulomb Green L 的局部角度加密 `dL=4.51102e-7`，但 x/y 两轴同时加密仍 `dL=0.0298039`、`dZ=0.00182062`、探针 `dq=0.00058998`，与旧球云法 `dL=0.0778894`。系统性紧支 curl 27→71→211 构造成功；球腔 `l=1` 的有限 Dirichlet 截断 J 误差随外球 R=0.52/0.85/1.30m 分别为 9.345%/1.488%/0.199%。**下一次运行同一命令会增加全新的 2×2 独立 x/y 求积拆分，以及未经强制对称化的磁矩阵 skew。新版尚未本机实测；229 模态及完整 BFZI 连续 G1 仍 FAIL。**

**物理有效性：** 程序成功或测试通过不代表连续物理场达到精度目标。父空间内的 `FULL_PARENT_ONLY`、功率代数闭合和响应门槛也不能代替独立收敛验证。


## P1-C 专项实海水入口（新增）

[P1-C 五模式真实海水 Coulomb L 证据目录](p1_full/README.md) 与旧全局
latest_full.log / p1/ 完全独立。首次建立目录只含说明文件，
**尚无实际运行日志**。本机执行：

    git pull --ff-only
    python run.py --mode bfzi-p1-full-sea --with-tests

随后才会自动生成并尝试推送 p1_full/latest_summary.md 与
p1_full/latest_full.log；保存五个自项、十个互项、
base/x_only/y_only/both 的独立分轴变化及两种原核表示交叉比较。
程序正常结束也不意味着 G1 物理精度验收通过。
