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

**最近真正本机实测（2026-11-04 UTC，旧代码基线 `19c2c52`）：** `60 passed in 49.90s`；229 模态外域 0.85→1.05→1.30m 两步均 `domain_response_gate=FAIL`，最后一步 `dZ=0.9506%`、共同域 `dJ=2.2970%`、设备近区 `dJ=1.3923%`。**新 G1 代码尚未运行**。下一次使用 `--mode bfzi-g1-foundations`，摘要将专门保留原始 Coulomb Green L 奇异分拆 5 模态实验、27→71→211 紧支 curl 层级和解析球腔无限域边界的全部指标。它们是先导物理基准，尚不能用作 229 模态高精度教师。

**物理有效性：** 程序成功或测试通过不代表连续物理场达到精度目标。父空间内的 `FULL_PARENT_ONLY`、功率代数闭合和响应门槛也不能代替独立收敛验证。
