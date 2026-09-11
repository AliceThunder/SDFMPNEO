"""Bounded-memory collocation selection for fixed-network Gauss-Newton."""
from __future__ import annotations

from .research_helpers import (
    _combined_metrics,
    _evaluate_network,
    _evaluate_semigroup,
    _hard_subset,
)

# Keep the selected residual-row count large enough to represent more than the
# single worst few geometry points, while still bounding the dense parameter
# Jacobian at high thermal rank.  At r=198 the defaults select six physics
# points (1188 residual rows) instead of the former three.
_MAX_PHYSICS_JACOBIAN_ROWS = 1280
_MAX_SEMIGROUP_JACOBIAN_ROWS = 640


def linearization(network, field, points, semigroup_points, evaluated, config, monitor):
    n_modes = max(1, int(network.n_modes))
    physics_budget = min(
        int(config.jacobian_point_budget),
        max(1, _MAX_PHYSICS_JACOBIAN_ROWS // n_modes),
    )
    physics_subset = _hard_subset(points, evaluated.physics_norms, physics_budget)
    physics = _evaluate_network(
        network,
        field,
        physics_subset,
        jacobian=True,
        monitor=monitor,
        work_label="physics_jacobian",
    )

    semigroup = []
    if len(semigroup_points):
        semigroup_budget = min(
            int(config.semigroup_jacobian_point_budget),
            max(1, _MAX_SEMIGROUP_JACOBIAN_ROWS // n_modes),
        )
        semigroup_subset = _hard_subset(
            semigroup_points,
            evaluated.semigroup_norms,
            semigroup_budget,
        )
        semigroup = _evaluate_semigroup(
            network,
            semigroup_subset,
            jacobian=True,
            monitor=monitor,
            work_label="restart_jacobian",
        )
    return _combined_metrics(physics, semigroup)


__all__ = ["linearization"]
