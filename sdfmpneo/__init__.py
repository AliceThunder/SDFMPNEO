"""SDF-MPNEO executable core."""

from .analytic import (
    AnalyticEvolutionGraph,
    AnalyticSeries,
    CertifiedAnalyticEvolutionOperator,
    GeometryConditionedAnalyticEvolutionOperator,
    MultiChartGeometryAnalyticEvolutionOperator,
    MultiChartGeometryAnalyticPrediction,
    ParametricAnalyticEvolutionGraph,
    ParametricAnalyticSeries,
)
from .analytic.fixed_response_network import FixedAnalyticResponseNetwork
from .electrothermal import CertifiedElectroThermalVectorField
from .em import (
    ParametricEMProblem,
    ReducedEMModel,
    ResidualGreedyEMReducer,
    SolidTerminalPortSet,
    SparseEnergyReducedEMModel,
    SparseEnergyResidualGreedyEMReducer,
)
from .geometry_family import CertifiedGeometryElectroThermalFamily
from .geometry_multichart import (
    CertifiedMultiChartGeometryFamily,
    MultiChartGeometrySelection,
)
from .model import (
    ExecutableSDFMPNEOModel,
    OnlinePrediction,
    ParametricExecutableSDFMPNEOModel,
    ParametricOnlinePrediction,
)
from .tetra_core import TetrahedralElectroThermalCore
from .thermal import ThermalSpectralModel
from .training import GeometryElectroThermalResidual, SolutionDataFreeClosedLoopTrainer

__version__ = "0.9.0"

__all__ = [
    "AnalyticEvolutionGraph",
    "AnalyticSeries",
    "ParametricAnalyticEvolutionGraph",
    "ParametricAnalyticSeries",
    "FixedAnalyticResponseNetwork",
    "CertifiedAnalyticEvolutionOperator",
    "GeometryConditionedAnalyticEvolutionOperator",
    "MultiChartGeometryAnalyticPrediction",
    "MultiChartGeometryAnalyticEvolutionOperator",
    "CertifiedElectroThermalVectorField",
    "CertifiedGeometryElectroThermalFamily",
    "CertifiedMultiChartGeometryFamily",
    "MultiChartGeometrySelection",
    "GeometryElectroThermalResidual",
    "SolutionDataFreeClosedLoopTrainer",
    "ThermalSpectralModel",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "SparseEnergyReducedEMModel",
    "SparseEnergyResidualGreedyEMReducer",
    "SolidTerminalPortSet",
    "ExecutableSDFMPNEOModel",
    "OnlinePrediction",
    "ParametricExecutableSDFMPNEOModel",
    "ParametricOnlinePrediction",
    "TetrahedralElectroThermalCore",
]

from .research import ResearchElectroThermalModel, demo_research_model, model_from_config
from .training.research import ResearchTrainingConfig, ResearchTrainingReport, train_research_graph
from .training.cpp_guard import install_cpp_auto_build_guard
from .training.cpp_runtime import install_cpp_training_backend
from .training.parallel_runtime import install_training_acceleration
from .training.adaptive_runtime import (
    ResearchTrainingContinuation,
    install_adaptive_training,
    install_geometry_continuation_persistence,
)
from .training.late_stage_runtime import install_late_stage_training
from .training.late_stage_batch import install_late_stage_batching
from .training.cpp_visibility import install_cpp_training_visibility
from .training.node_compile_runtime import install_node_only_training_compile
from .training.observation_runtime import install_observation_training_acceleration
from .training.cpp_dag_runtime import install_native_dag_training
from .training.coverage_runtime import install_high_dimensional_collocation
from .training.max_residual_runtime import install_max_residual_training
from .analytic.state_runtime import install_analytic_state_graph
from .training.residual_state_runtime import install_residual_driven_state_training
from .training.state_search_runtime import (
    install_geometry_seed_coalescing,
    install_state_search_policy,
)
from .training.state_split_runtime import install_screened_split_policy
from .training.block_sparse_search_runtime import install_block_sparse_state_search
from .training.convergence_rescue_runtime import install_convergence_rescue
from .training.state_linearization_runtime import install_independent_state_native_linearization
from .training.fixed_network_runtime import install_fixed_analytic_response_network
from .training.geometry_context_runtime import install_concurrent_geometry_context_cache

# Install the one-shot auto-build guard before any runtime can invoke a compiled
# kernel. The C++ layer stays lazy: importing sdfmpneo never launches a compiler.
install_cpp_auto_build_guard()
install_cpp_training_backend()
# Upgrade the public parametric graph before legacy training wrappers capture
# evaluator bindings. Old checkpoints and the adaptive DAG remain loadable.
install_analytic_state_graph()
install_training_acceleration()
install_adaptive_training()
install_late_stage_training()
install_late_stage_batching()
install_node_only_training_compile()
install_observation_training_acceleration()
install_native_dag_training()
install_high_dimensional_collocation()
install_max_residual_training()
install_residual_driven_state_training()
install_state_search_policy()
install_screened_split_policy()
install_block_sparse_state_search()
install_convergence_rescue()
install_independent_state_native_linearization()
# Fresh/empty models use a fixed-depth low-rank analytic response network. This
# final training binding removes normal Grow/Enrich/Split/candidate search: all
# network parameters exist from the start and are optimized continuously. A
# loaded non-empty legacy DAG deliberately falls back to its historical trainer.
install_fixed_analytic_response_network()
# Timing wrappers are last so they observe the actual final training path.
install_cpp_training_visibility()

# Export the trainer after runtime installation so callers receive the fixed
# analytic-network trainer for fresh models.
from .training.research import train_research_graph as train_research_graph

__all__ += ["ResearchElectroThermalModel", "ResearchTrainingConfig", "ResearchTrainingReport",
            "ResearchTrainingContinuation", "demo_research_model", "model_from_config",
            "train_research_graph"]

from .geometry_research import GeometryResearchModel, geometry_model_from_config
# Geometry models create an empty compatibility graph carrying the declared
# geometry/current names; fixed-network training replaces that empty graph while
# retaining the same input ordering and physical geometry context.
install_geometry_seed_coalescing(GeometryResearchModel)
install_geometry_continuation_persistence(GeometryResearchModel)
install_concurrent_geometry_context_cache(GeometryResearchModel)
__all__ += ["GeometryResearchModel", "geometry_model_from_config"]
