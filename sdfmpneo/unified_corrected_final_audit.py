"""Final held-out audit adapter using the corrected production truth tensors."""
from __future__ import annotations

from . import unified_final_audit as _base
from .unified_corrected_truth import solve_port_truth_tensors, solve_truth_tensors


def run_final_held_out_audit(settings, model, geometries, monitor=None):
    # The base audit is deliberately reused so its current/circuit/integrator and
    # thermal checks remain a single implementation.  Its module-level truth
    # hooks are rebound for this call to the production corrected truth.
    old_truth = _base.solve_truth_tensors
    old_port = _base.solve_port_truth_tensors
    _base.solve_truth_tensors = solve_truth_tensors
    _base.solve_port_truth_tensors = solve_port_truth_tensors
    try:
        return _base.run_final_held_out_audit(settings, model, geometries, monitor=monitor)
    finally:
        _base.solve_truth_tensors = old_truth
        _base.solve_port_truth_tensors = old_port


__all__ = ["run_final_held_out_audit"]
