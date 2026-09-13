"""Command-line entry for the single unified SDF-MPNEO model."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_project_runner():
    path = Path.cwd() / "run.py"
    if not path.is_file():
        raise SystemExit(
            "SDF-MPNEO 的正式配置入口是项目根目录 run.py；"
            "请在包含 run.py 的项目目录运行 sdfmpneo 或 python run.py。"
        )
    spec = importlib.util.spec_from_file_location("sdfmpneo_project_runner", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载统一入口：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise SystemExit(f"统一入口缺少 main()：{path}")
    return module


def main(argv=None):
    return _load_project_runner().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
