from .fixed_response_physical_init import _PhysicalFeatureInitMixin
from .fixed_response_init import _NetworkInitMixin
from .fixed_response_funnel_structure import _FunnelStructureMixin
from .fixed_response_structure import _NetworkStructureMixin
from .fixed_response_fast import _DepthOneFastMixin
from .fixed_response_activation import _LayerActivationMixin
from .fixed_response_funnel_fast import _FunnelFastMixin
from .fixed_response_generic import _GenericAnalyticMixin
from .fixed_response_metadata import _NetworkMetadataMixin


class FixedAnalyticResponseNetwork(
    _PhysicalFeatureInitMixin,
    _NetworkInitMixin, _FunnelStructureMixin, _NetworkStructureMixin,
    _DepthOneFastMixin, _LayerActivationMixin, _FunnelFastMixin,
    _GenericAnalyticMixin, _NetworkMetadataMixin,
):
    """Finite-horizon multilayer analytic response network with low-rank coupling."""

    kind = "fixed_analytic_response_network"
    format_version = 7


__all__ = ["FixedAnalyticResponseNetwork"]
