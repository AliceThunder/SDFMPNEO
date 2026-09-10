from pathlib import Path


def test_qt_monitor_describes_only_current_fixed_network_phases():
    source = (Path(__file__).parents[1] / "sdfmpneo" / "training" / "qt_monitor.py").read_text(encoding="utf-8")
    for text in (
        "自动选择热空间阶数", "连续优化解析网络参数", "独立物理残差验证",
        "残差验证剪枝", "有效响应通道", "训练最大残差", "验证最大残差",
    ):
        assert text in source
    for obsolete in (
        "candidate_search", "搜索响应神经元", "quadratic_seed", "geometry_seed",
        "Grow", "Enrich", "Split",
    ):
        assert obsolete not in source
