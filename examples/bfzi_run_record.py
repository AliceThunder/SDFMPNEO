"""Concise BFZI console output and versioned latest-run evidence (no CI).

Use the existing local Git credentials to update ONLY the two designated
log files. Full numerical output is retained for scientific interpretation.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import traceback


LOG_DIR = Path("docs/research/bfzi_uwpt_electrothermal/runs")
BRANCH = "research/bfzi-uwpt-electrothermal-surrogate"
FIELDS = (
    "modes", "relative_Rinv_residual", "parent_dJ_energy_upper",
    "sea_power", "dR", "dK", "dK_dual", "dL", "dZ", "dImZ",
    "dJ_volume", "dJ_ports_volume", "dq", "max_probe_dq",
    "peak_scaled_max_dq", "normal_peak_scaled_dq",
    "dJ_common", "dJ_ports_common", "dJ_near035",
    "dJ_ports_near035", "dq_near035", "domain_response_gate",
    "dR_common", "dK_common", "dL_common",
    "dZ_RK_only", "dZ_L_only",
    "dJ_near035_RK_only", "dJ_near035_L_only",
    "operator_gate", "response_gate", "combined_gate",
)
MATCH_FIELD = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([^\s,]+)")


def _fields(line: str):
    return dict(MATCH_FIELD.findall(line))


def _short(line: str) -> str | None:
    line = line.strip()
    if line.startswith("g1_"):
        return line
    if line.startswith(("Building enriched", "Refined R/K +", "Independent ")):
        return line
    if line.startswith("refined R/K nodes="):
        return line
    if line.startswith("assembling B nodes="):
        return line
    if line.startswith(("adaptive_screen:", "magnetic_gate_failed:")):
        return line
    if line.startswith(("snapshot_rank_4:", "full_weak_form:")):
        values = _fields(line)
        return line.split(":", 1)[0] + ": " + ", ".join(
            f"{k}={values[k]}" for k in ("dZ", "dq", "dP")
            if k in values
        )
    if line.startswith(("most_influential_for_volume_J:",
                        "most_influential_for_normal_Joule:")):
        return line
    if line.startswith(("ball_", "rk_normal_", "rk_far_")) and (
        "_vs_A:" in line or (
            line.startswith("rk_far_azimuth_")
            and re.search(r"_vs_\d+:$",line.split(":",1)[0]+":")
        )
    ):
        label, rest = line.split(":", 1)
        values = _fields(rest)
        return label + ": " + ", ".join(
            f"{k}={values[k]}" for k in FIELDS if k in values
        )
    if "_to_" in line and ": dZ=" in line:
        label, rest = line.split(":", 1)
        # Show final physical stages only. All early stages remain in the file.
        if not any(part in label for part in (
            "to_anisotropic_face_curls", "to_global_short_scale",
            "to_global_multiscale", "to_corner_source_curls",
        )):
            return None
        values = _fields(rest)
        return label + ": " + ", ".join(
            f"{k}={values[k]}" for k in (
                "dZ", "dJ_volume", "dJ_ports_volume", "dq",
                "max_probe_dq", "dP",
            ) if k in values
        )
    if line.startswith(("anisotropic_face_curls_normal_sweep:",
                        "global_short_scale_normal_sweep:",
                        "global_multiscale_normal_sweep:",
                        "corner_source_curls_normal_sweep:")):
        values = _fields(line)
        return line.split(":", 1)[0] + ": " + ", ".join(
            f"{k}={values[k]}" for k in (
                "dq", "peak_scaled_max_dq",
            ) if k in values
        )
    if line.startswith("global_scale_condition_"):
        return line
    if line.startswith("fixed_domain_family:"):
        return line
    if line.startswith("radius="):
        return line
    if line.startswith("domain_operator_step:"):
        pair = line.split(":",1)[1].strip().split(" ",1)[0]
        values = _fields(line)
        return "domain_operator_step: "+pair+" "+", ".join(
            f"{k}={values[k]}" for k in (
                "dR_common", "dK_common", "dL_common",
                "dZ_RK_only", "dZ_L_only",
                "dJ_near035_RK_only", "dJ_near035_L_only",
            ) if k in values
        )
    if line.startswith("domain_step:"):
        label, rest = line.split(":",1)
        values = _fields(rest)
        pair = rest.strip().split(" ",1)[0]
        return label+": "+pair+" "+", ".join(
            f"{k}={values[k]}" for k in
            ("dZ","dJ_common","dJ_near035","dq_near035",
             "dq","dP","domain_response_gate")
            if k in values
        )
    if ": modes=" in line and any(line.startswith(x) for x in (
        "anisotropic_face_curls:", "global_short_scale:",
        "global_multiscale:", "corner_source_curls:",
    )):
        values = _fields(line)
        return line.split(":", 1)[0] + ": " + ", ".join(
            f"{k}={values[k]}" for k in (
                "modes", "relative_Rinv_residual",
                "parent_dJ_energy_upper", "sea_power",
            ) if k in values
        )
    if ("Traceback (" in line or line.startswith((
        "RuntimeError:", "ValueError:", "AssertionError:", "ERROR",
        "FAILED", "Warning:", "WARNING:",
    ))):
        return line
    return None


class _LogWriter:
    def __init__(self, output, verbose: bool):
        self.output = output
        self.verbose = verbose
        self.remaining = ""
        self.console = sys.stdout

    def write(self, chunk):
        self.output.write(chunk)
        if self.verbose:
            self.console.write(chunk)
            return len(chunk)
        self.remaining += chunk
        while "\n" in self.remaining:
            line, self.remaining = self.remaining.split("\n", 1)
            message = _short(line)
            if message:
                self.console.write(message + "\n")
        return len(chunk)

    def flush(self):
        self.output.flush()
        if self.verbose:
            self.console.flush()


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )


def _summary(root, flags, started, ended, status, raw):
    lines = raw.splitlines()
    stages, comparisons, gates, domains, domain_steps = [], [], [], [], []
    domain_operators = []
    g1_pilot = []
    for rawline in lines:
        line = rawline.strip()
        if line.startswith("g1_"):
            g1_pilot.append(line)
        if ": modes=" in line:
            label, rest = line.split(":", 1)
            values = _fields(rest)
            if "modes" in values and "parent_dJ_energy_upper" in values:
                stages.append((label, values))
        if "_to_" in line and ": dZ=" in line:
            label, rest = line.split(":", 1)
            comparisons.append((label, _fields(rest)))
        if line.startswith("radius=") and "sea_power=" in line:
            domains.append(_fields(line))
        if line.startswith("domain_step:"):
            domain_steps.append((line.split(":",1)[1].strip().split(" ",1)[0],
                                 _fields(line)))
        if line.startswith("domain_operator_step:"):
            domain_operators.append((
                line.split(":",1)[1].strip().split(" ",1)[0],
                _fields(line),
            ))
        if ("_vs_A:" in line
                or (line.startswith("rk_far_azimuth_")
                    and re.search(r"_vs_\d+:",line))
                or line.startswith((
                    "adaptive_screen:", "global_scale_condition_",
                ))):
            gates.append(line)

    commit = _git(root, "rev-parse", "--short", "HEAD")
    code_ref = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    run_mode = (
        flags[flags.index("--mode")+1]
        if "--mode" in flags and flags.index("--mode")+1 < len(flags)
        else "bfzi-convergence"
    )
    command = "python run.py --mode " + run_mode + " " + " ".join(
        x for x in flags if x.startswith("--") and x not in (
            "--mode", "--verbose-log", "--no-publish-log", "--with-tests",
        )
    )
    test_result = re.search(r"\d+ passed in [\d.]+s", raw)
    parts = [
        "# BFZI 最近一次本地运行摘要",
        "",
        f"- 运行状态：**{status}**",
        f"- 开始（UTC）：{started}",
        f"- 结束（UTC）：{ended}",
        f"- 当时检出的代码提交：`{code_ref}`",
        f"- 执行：`{command.strip()}`",
        f"- 同批 pytest：{test_result.group(0) if test_result else '未包含或未通过'}",
        "- 物理验收：**未自动认证**。下面仅为本次数值相互比较；请根据收敛实验人工判断。",
        "- [查看完整原始日志](latest_full.log)；此摘要与原始日志均由本机 Git 更新，不使用 GitHub Actions。",
    ]
    if g1_pilot:
        parts.extend([
            "", "## G1 三条基础路线的实验基准（尚非认证）", "",
        ])
        parts.extend(f"- `{line}`" for line in g1_pilot)
        parts.append(
            "\nGreen L 独立奇异拆分当前仅在小试探空间运行；"
            "紧支 curl 层级仅证明构造/嵌套，尚未证明逼近收敛；"
            "解析透明边界只适用于球腔偶极的 l=1 扇区。"
        )
    if stages:
        parts.extend([
            "", "## 嵌套物理空间", "",
            "| 层级 | 模态数 | 父空间电流上界 | 海水功率 (W) |",
            "|---|---:|---:|---:|",
        ])
        for label, values in stages:
            parts.append(
                f"| {label} | {values.get('modes','—')} | "
                f"{values.get('parent_dJ_energy_upper','—')} | "
                f"{values.get('sea_power','—')} |"
            )
    if comparisons:
        parts.extend([
            "", "## 相邻空间变化（相对于前一级，不是连续真解误差）", "",
            "| 阶段 | dZ | dJ_volume | dJ_ports_volume | dq |",
            "|---|---:|---:|---:|",
        ])
        for label, values in comparisons:
            parts.append(
                f"| {label} | {values.get('dZ','—')} | "
                f"{values.get('dJ_volume','—')} | "
                f"{values.get('dJ_ports_volume','—')} | "
                f"{values.get('dq','—')} |"
            )
    if domains:
        parts.extend([
            "", "## 有限外域：相同物理方向、不同人工边界", "",
            "| 外域半径 (m) | 模态数 | R/K 节点 | 海水功率 (W) |",
            "|---:|---:|---:|---:|",
        ])
        for d in domains:
            parts.append(
                f"| {d.get('radius','—')} | {d.get('modes','—')} | "
                f"{d.get('fine_RK_nodes','—')} | {d.get('sea_power','—')} |"
            )
        parts.extend([
            "",
            "| 半径比较 | dZ | 0.52m dJ | 0.52m 双端口 dJ | 0.35m dJ | 0.35m 双端口 dJ | 0.35m dq | 探针 dq | dP | 阶段响应门槛 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for label,d in domain_steps:
            parts.append(
                f"| {label} | {d.get('dZ','—')} | "
                f"{d.get('dJ_common','—')} | "
                f"{d.get('dJ_ports_common','—')} | "
                f"{d.get('dJ_near035','—')} | "
                f"{d.get('dJ_ports_near035','—')} | "
                f"{d.get('dq_near035','—')} | "
                f"{d.get('dq','—')} | {d.get('dP','—')} | "
                f"{d.get('domain_response_gate','—')} |"
            )
        if domain_operators:
            parts.extend([
                "",
                "## 外域响应拆分：R/K 与磁 Green L",
                "",
                "| 半径比较 | dR 共坐标 | dK 共坐标 | dL 共坐标 | 仅变 R/K 时 dZ | 仅变 L 时 dZ | 近区 dJ（仅变 R/K） | 近区 dJ（仅变 L） |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for label,d in domain_operators:
                parts.append(
                    f"| {label} | {d.get('dR_common','—')} | "
                    f"{d.get('dK_common','—')} | "
                    f"{d.get('dL_common','—')} | "
                    f"{d.get('dZ_RK_only','—')} | "
                    f"{d.get('dZ_L_only','—')} | "
                    f"{d.get('dJ_near035_RK_only','—')} | "
                    f"{d.get('dJ_near035_L_only','—')} |"
                )
            parts.append(
                "\n拆分使用相同物理高斯方向与前一半径的 R 白化坐标，"
                "只更换一组算子做敏感性分析；混合系统不是实际物理外域解，"
                "两列误差不可相加，也不能代替磁积分收敛证书。"
            )
        parts.append(
            "\n共同海水区域指标是固定 0.52 m 球域中的加权采样差异；"
            "设备近区是半径 0.35 m 的固定湿区权重子集。"
            "两者分别显示人工边界附近和封装周围的空间误差。"
            "各半径仍独立重组磁 L，不能代替无穷海水域的收敛证明。"
        )
    parts.extend(["", "## 收敛判定与诊断", ""])
    parts.extend(f"- `{line}`" for line in gates)
    if not stages and not comparisons and not gates and not domains and not g1_pilot:
        parts.append("- 该运行没有可自动提取的嵌套指标；请看完整日志。")
    parts.extend([
        "",
        "**注意：** 留一模态、全域电流变化和 A/B 磁积分响应通过，"
        "都不能单独构成连续物理参考解的精度证明。",
        "",
    ])
    return "\n".join(parts)


def _publish(root, paths):
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    if branch.returncode or branch.stdout.strip() != BRANCH:
        print("  [日志已保存；未推送] 当前不是 BFZI 研究分支。")
        return False
    relative = [str(p.relative_to(root)).replace("\\", "/") for p in paths]
    add = _git(root, "add", "--", *relative)
    if add.returncode:
        print(f"  [日志已保存；Git add 失败] {add.stderr.strip()}")
        return False
    changed = _git(root, "diff", "--cached", "--quiet", "--", *relative)
    if changed.returncode == 0:
        print("  GitHub 日志内容未变化，无须重复提交。")
        return True
    if changed.returncode != 1:
        print(f"  [日志已保存；Git diff 失败] {changed.stderr.strip()}")
        return False
    commit = _git(
        root, "commit", "--only",
        "-m", "logs(BFZI): update latest local convergence result",
        "--", *relative,
    )
    if commit.returncode:
        print(f"  [日志已保存；Git commit 失败] {commit.stderr.strip()}")
        return False
    push = _git(root, "push", "origin", f"HEAD:refs/heads/{BRANCH}")
    if push.returncode:
        print(f"  [日志已在本地提交；推送失败] {push.stderr.strip()}")
        print("  请检查认证或远端是否前进；解决后运行 git push origin HEAD")
        return False
    print(f"  [已推送 GitHub] {BRANCH} · "
          "docs/research/bfzi_uwpt_electrothermal/runs/latest_summary.md")
    return True


def run_and_publish(run, *, flags, verbose=False, publish=True,
                    with_tests=False):
    """Run BFZI once, stream only useful metrics, publish local evidence."""
    root = Path(__file__).resolve().parents[1]
    output_dir = root / LOG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "latest_full.log"
    summary_path = output_dir / "latest_summary.md"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("[BFZI] 运行开始；详细结果会保存在仓库日志目录。", flush=True)
    failed = 0
    with full_path.open("w", encoding="utf-8") as file:
        writer = _LogWriter(file, verbose)
        if with_tests:
            print("[BFZI] 运行 3 组 BFZI pytest...", flush=True)
            test = subprocess.run(
                [sys.executable, "-m", "pytest",
                 "tests/test_bfzi_geometry_v1.py",
                 "tests/test_bfzi_mqs_v1.py",
                 "tests/test_bfzi_neural_v1.py", "-q"],
                cwd=root, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            writer.write("pytest:\n" + test.stdout + test.stderr + "\n")
            print("[BFZI] pytest " +
                  ("PASS" if test.returncode == 0 else "FAIL") +
                  " · " + (test.stdout.strip().splitlines()[-1]
                             if test.stdout.strip() else "无输出"), flush=True)
            if test.returncode:
                failed = test.returncode
        if not failed:
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    run()
            except (Exception, KeyboardInterrupt):
                writer.write(traceback.format_exc())
                writer.flush()
                failed = 1
        if writer.remaining and not verbose:
            message = _short(writer.remaining)
            if message:
                print(message)
        writer.flush()
    ended = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw = full_path.read_text(encoding="utf-8")
    status = "PASS（程序完成；不代表物理收敛）" if not failed else "FAIL（测试或执行未通过）"
    summary_path.write_text(
        _summary(root, flags, started, ended, status, raw),
        encoding="utf-8",
    )
    print(f"[BFZI] 执行状态：{status}；详细日志 {full_path.relative_to(root)}")
    if publish:
        uploaded = _publish(root, [summary_path, full_path])
        if not uploaded and failed == 0:
            return 2  # The solve completed, but publishing its log failed.
    else:
        print("[BFZI] 已按 --no-publish-log 仅保存本地结果。")
    return failed
