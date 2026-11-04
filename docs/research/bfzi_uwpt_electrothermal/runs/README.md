# BFZI 最近一次本地运行日志

此目录由用户本地的 `python run.py --mode bfzi-convergence ...` 自动维护，不使用 GitHub Actions 或云端后台任务。

**以下两个文件已在用户本机实际生成并成功推送 GitHub，后续运行自动覆盖更新：**

- [latest_summary.md](latest_summary.md)：精简的嵌套空间对照表、误差上界、磁积分 A/B 和最终门槛；供 GitHub 浏览和后续会话快速读取。
- [latest_full.log](latest_full.log)：原始 stdout/stderr，包括完整场探针数组和自适应迭代信息；供复查问题和二次分析。

自动流程：`run.py` 接管 BFZI 收敛实验输出 → 屏幕只显示关键物理指标 → 写入上述文件 → 本机 `git add` 指定日志文件、`git commit --only`、`git push origin HEAD:refs/heads/research/bfzi-uwpt-electrothermal-surrogate`。使用的是本机已配置的 Git 提交身份与推送权限。只提交上述两个日志文件，不把代码、数据集或其他暂存内容混入提交。运行失败也保存并尝试推送日志；推送失败会明确提示，原始文件仍保留本地。

```powershell
git pull --ff-only
python run.py --mode bfzi-convergence --basis-domain-ball --basis-ball-targeted --basis-domain-extend --with-tests
# 只保留本地，不推送：
python run.py --mode bfzi-convergence --basis-ball-targeted --no-publish-log
# 如需看到全部原始标准输出：
python run.py --mode bfzi-convergence --basis-ball-targeted --verbose-log
```

**最新本机上传（2026-11-04 UTC，代码基线 `60a3ba7`）：** `60 passed in 39.33s`；同一套 229 物理电流方向的外域 0.52→0.70→0.85m 两个步长均 `domain_response_gate=FAIL`。0.70→0.85m 的共同海水 J 变化 `1.9287%`、原有四个体积探针 q 变化 `4.0557%`（新版本才增加 12 个封装近壁探针）、海水功率变化 `0.4269%`。新命令（尚未本机运行）从 `0.85m` 直接计算 `1.05/1.30m`，无需重算两个旧外域；Markdown 摘要将并列记录共同 0.52m 湿区和设备附近 0.35m 区域的 J、两端口 J 与空间 q，并保留逐步 PASS/FAIL。磁 Green L 算子独立验收仍 FAIL，外域平台通过也不等于最终 G1 通过。

**物理有效性：** 程序成功或测试通过不代表连续物理场达到精度目标。父空间内的 `FULL_PARENT_ONLY`、功率代数闭合和响应门槛也不能代替独立收敛验证。
