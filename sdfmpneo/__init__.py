"""SDF-MPNEO executable core."""

import os as _os

# Independent collocation parallelism is the outer level; keep dense BLAS kernels
# single-threaded by default to avoid nested oversubscription. Explicit caller
# environment settings still win.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_thread_env, "1")

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
from .training.parallel_runtime import install_training_acceleration
from .training.adaptive_runtime import (
    ResearchTrainingContinuation,
    install_adaptive_training,
    install_geometry_continuation_persistence,
)
from .training.late_stage_runtime import (
    install_geometry_working_set_cache,
    install_late_stage_training,
)
from .training.late_stage_batch import install_late_stage_batching
from .training.node_compile_runtime import install_node_only_training_compile

install_training_acceleration()
install_adaptive_training()
install_late_stage_training()
install_late_stage_batching()
install_node_only_training_compile()

# Export the trainer after runtime installation so callers receive the selective,
# structurally accelerated implementation rather than a historical function object.
from .training.research import train_research_graph as train_research_graph

__all__ += ["ResearchElectroThermalModel", "ResearchTrainingConfig", "ResearchTrainingReport",
            "ResearchTrainingContinuation", "demo_research_model", "model_from_config",
            "train_research_graph"]

from .geometry_research import GeometryResearchModel, geometry_model_from_config
install_geometry_continuation_persistence(GeometryResearchModel)
install_geometry_working_set_cache(GeometryResearchModel)
__all__ += ["GeometryResearchModel", "geometry_model_from_config"]
