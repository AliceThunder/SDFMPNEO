from .fixed_response_init import _NetworkInitMixin
from .fixed_response_structure import _NetworkStructureMixin
from .fixed_response_fast import _DepthOneFastMixin
from .fixed_response_generic import _GenericAnalyticMixin
from .fixed_response_metadata import _NetworkMetadataMixin


class FixedAnalyticResponseNetwork(
    _NetworkInitMixin, _NetworkStructureMixin, _DepthOneFastMixin,
    _GenericAnalyticMixin, _NetworkMetadataMixin,
):
    """Finite-horizon analytic flow with low-rank capacity that scales to high thermal rank."""

    kind = "fixed_analytic_response_network"
    format_version = 6


__all__ = ["FixedAnalyticResponseNetwork"]
