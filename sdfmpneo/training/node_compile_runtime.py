from __future__ import annotations

from sdfmpneo.analytic.parametric_realization import compile_parametric_nodes


def install_node_only_training_compile() -> None:
    """Route training-only compile calls to the exact node/source-only compiler."""
    from . import research as training_research
    from . import late_stage_runtime

    training_research.compile_parametric_realization = compile_parametric_nodes
    late_stage_runtime.compile_parametric_realization = compile_parametric_nodes
