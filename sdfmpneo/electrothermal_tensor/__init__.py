"""Numerical primitives reused by the unified SDF-MPNEO solver.

The former quadratic-Joule neural ROM is no longer a public model path.  This
subpackage now exposes only generic integration/network/thermal-operator tools
used by the single unified residual-corrected architecture.
"""

from .integrators import (
    GeneralizedThermalSpectrum,
    IntegrationResult,
    integrate_etd2,
    integrate_etd2_adaptive,
    integrate_imex_euler,
    integrate_reference,
)
from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .vector_field import ReducedThermalOperator

__all__ = [
    "FeatureNormalizer",
    "GeneralizedThermalSpectrum",
    "IntegrationResult",
    "ReducedThermalOperator",
    "ResidualMLPConfig",
    "build_residual_mlp",
    "integrate_etd2",
    "integrate_etd2_adaptive",
    "integrate_imex_euler",
    "integrate_reference",
]
