from __future__ import annotations

from .state_graph import (
    AnalyticStateSource,
    add_response_state,
    clone_state_graph,
    compile_state_series,
    enrich_response_state,
    has_multi_source_states,
    replace_source_weights,
    response_source_count,
    response_sources,
    split_response_source,
    weight_parameter_count,
    weight_parameter_map,
)
from .state_realization import (
    compile_state_nodes,
    compile_state_realization,
    evaluate_state_stable,
    evaluate_state_stable_with_jacobians,
    state_weight_value_jacobian,
)

_INSTALLED = False


def install_analytic_state_graph() -> None:
    """Upgrade the public parametric graph to residual-driven analytic states.

    The public graph class and old single-source API remain valid. New states can
    carry an unbounded active source set; no fixed K/source-count hyperparameter
    is introduced.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .parametric import ParametricAnalyticEvolutionGraph
    from . import parametric_realization as pr
    from . import operator as operator_module
    from . import geometry_operator as geometry_operator_module
    import sdfmpneo.analytic as analytic_api
    from sdfmpneo.training import research as training_research
    import sdfmpneo.research as research_module
    from sdfmpneo.training.state_persistence_runtime import install_state_persistence

    if not hasattr(
        ParametricAnalyticEvolutionGraph,
        "_sdfmpneo_original_add_product_response",
    ):
        ParametricAnalyticEvolutionGraph._sdfmpneo_original_add_product_response = (
            ParametricAnalyticEvolutionGraph.add_product_response
        )
    original_add = (
        ParametricAnalyticEvolutionGraph._sdfmpneo_original_add_product_response
    )

    def add_product_response(self, name, target_mode, parents, weight):
        original_add(self, name, target_mode, parents, weight)
        self._sdfmpneo_state_sources = getattr(
            self, "_sdfmpneo_state_sources", {}
        )
        self._sdfmpneo_state_sources[name] = (
            AnalyticStateSource.make(parents, weight),
        )

    ParametricAnalyticEvolutionGraph.add_product_response = add_product_response
    ParametricAnalyticEvolutionGraph.add_response_state = add_response_state
    ParametricAnalyticEvolutionGraph.enrich_response_state = enrich_response_state
    ParametricAnalyticEvolutionGraph.response_sources = response_sources
    ParametricAnalyticEvolutionGraph.response_source_count = response_source_count
    ParametricAnalyticEvolutionGraph.weight_parameter_count = property(
        weight_parameter_count
    )
    ParametricAnalyticEvolutionGraph.weight_parameter_map = weight_parameter_map
    ParametricAnalyticEvolutionGraph.clone = clone_state_graph
    ParametricAnalyticEvolutionGraph.split_response_source = split_response_source
    ParametricAnalyticEvolutionGraph.compile = compile_state_series

    # Replace stable realizations at their defining module and already-imported
    # bindings. The single-source special case remains mathematically identical.
    pr.compile_parametric_nodes = compile_state_nodes
    pr.compile_parametric_realization = compile_state_realization
    pr.evaluate_parametric_stable = evaluate_state_stable
    pr.evaluate_parametric_stable_with_jacobians = (
        evaluate_state_stable_with_jacobians
    )
    analytic_api.compile_parametric_realization = compile_state_realization
    analytic_api.evaluate_parametric_stable = evaluate_state_stable
    analytic_api.AnalyticStateSource = AnalyticStateSource
    operator_module.evaluate_parametric_stable = evaluate_state_stable
    geometry_operator_module.evaluate_parametric_stable = evaluate_state_stable
    training_research.compile_parametric_realization = compile_state_realization
    training_research.evaluate_parametric_stable = evaluate_state_stable
    training_research._clone = clone_state_graph
    research_module.evaluate_parametric_stable = evaluate_state_stable

    # Modules below imported the evaluator by value before installation.
    try:
        from sdfmpneo.training import parametric_residual
        parametric_residual.evaluate_parametric_stable_with_jacobians = (
            evaluate_state_stable_with_jacobians
        )
    except ImportError:
        pass

    for module_name in (
        "sdfmpneo.certification.geometry_mass_providers",
        "sdfmpneo.certification.geometry_mass_residual_domain",
        "sdfmpneo.certification.geometry_dynamics",
    ):
        try:
            module = __import__(module_name, fromlist=["evaluate_parametric_stable"])
            if hasattr(module, "evaluate_parametric_stable"):
                module.evaluate_parametric_stable = evaluate_state_stable
        except ImportError:
            pass

    install_state_persistence(research_module.ResearchElectroThermalModel)
    _INSTALLED = True


__all__ = [
    "AnalyticStateSource",
    "install_analytic_state_graph",
    "response_sources",
    "response_source_count",
    "weight_parameter_count",
    "weight_parameter_map",
    "has_multi_source_states",
    "clone_state_graph",
    "replace_source_weights",
    "split_response_source",
    "compile_state_realization",
    "evaluate_state_stable",
    "evaluate_state_stable_with_jacobians",
    "state_weight_value_jacobian",
]
