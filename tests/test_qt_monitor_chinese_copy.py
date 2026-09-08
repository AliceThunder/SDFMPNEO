from pathlib import Path


def test_training_monitor_uses_chinese_plot_copy_and_detailed_counters():
    source = (Path(__file__).parents[1] / "sdfmpneo" / "training" / "qt_monitor.py").read_text(
        encoding="utf-8")
    for text in (
        "均方物理残差", "物理残差范数", "响应神经元数量", "训练配点数量",
        "指标记录序号", "训练均方根残差", "训练最大残差", "独立验证最大残差",
        "响应神经元 {budget}", "配点扩充", "收敛目标 ≤",
    ):
        assert text in source
    for old_english in (
        "Mean squared physical residual", "Physical residual norms",
        "Accepted response neurons", "Collocation refinement", "Accepted update",
        "Train RMS", "Train maximum", "Validation maximum",
    ):
        assert old_english not in source
