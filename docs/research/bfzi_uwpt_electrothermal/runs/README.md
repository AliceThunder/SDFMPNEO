# BFZI 最近一次本地运行日志

此目录由用户本地的 `python run.py --mode bfzi-convergence ...` 自动维护，不使用 GitHub Actions 或云端后台任务。

**以下两个文件已在用户本机实际生成并成功推送 GitHub，后续运行自动覆盖更新：**

- [latest_summary.md](latest_summary.md)：精简的嵌套空间对照表、误差上界、磁积分 A/B 和最终门槛；供 GitHub 浏览和后续会话快速读取。
- [latest_full.log](latest_full.log)：原始 stdout/stderr，包括完整场探针数组和自适应迭代信息；供复查问题和二次分析。

自动流程：`run.py` 接管 BFZI 收敛实验输出 → 屏幕只显示关键物理指标 → 写入上述文件 → 本机 `git add` 指定日志文件、`git commit --only`、`git push origin HEAD:refs/heads/research/bfzi-uwpt-electrothermal-surrogate`。使用的是本机已配置的 Git 提交身份与推送权限。只提交上述两个日志文件，不把代码、数据集或其他暂存内容混入提交。运行失败也保存并尝试推送日志；推送失败会明确提示，原始文件仍保留本地。

```powershell
git pull --ff-only
python run.py --mode bfzi-convergence --basis-ball-targeted --with-tests
# 只保留本地，不推送：
python run.py --mode bfzi-convergence --basis-ball-targeted --no-publish-log
# 如需看到全部原始标准输出：
python run.py --mode bfzi-convergence --basis-ball-targeted --verbose-log
```

**已经验证的自动上传：** 2026-11-03 用户 Windows 本地运行（代码基线 `e95a377`），`60 passed in 37.98s`，229 模态数值实验正常完成，最新摘要与完整日志已上传。当前代码追加了 **207→210→213→229** 定向尺度分解及 R Gram 独立性指标；需再次运行才能产生新数值证据。

**物理有效性：** 程序成功或测试通过不代表连续物理场达到精度目标。父空间内的 `FULL_PARENT_ONLY`、功率代数闭合和响应门槛也不能代替独立收敛验证。
