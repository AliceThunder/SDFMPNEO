"""Paper-grade held-out checks for a saved geometry-family SDF-MPNEO model.

This script never trains the model.  It draws deterministic interior held-out
geometry/initial/current cases, compares the analytic surrogate against an
independent Radau trajectory with full sparse electromagnetic equilibrium, and
reports the exact geometry-box mass-residual norm equivalence plus sampled
M-energy contractivity diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from sdfmpneo import ResearchElectroThermalModel
from sdfmpneo.certification import (
    certify_geometry_mass_residual_equivalence,
    sampled_geometry_contractivity,
    validate_geometry_trajectory,
)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "__dict__"):
        return {key: jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--times", nargs="+", type=float, default=[0.0, 1.0, 100.0, 1e4, 1e5])
    parser.add_argument("--output", default="results/uwpt/theory_validation.json")
    args = parser.parse_args()

    model = ResearchElectroThermalModel.load(args.model)
    if not hasattr(model, "geometry_names"):
        parser.error("the saved model is not a geometry-family model")
    if model.training_config is None:
        parser.error("saved training configuration is required for held-out sampling")
    if args.cases < 1:
        parser.error("--cases must be positive")

    cfg = model.training_config
    n_modes = model.graph.n_modes
    n_geometry = len(model.geometry_names)
    n_current = model.current_matrix.shape[1]

    # Deliberately use a seed distinct from training/validation/EM-anchor seeds,
    # and keep samples away from box faces/axis probes used by geometry seeding.
    engine = qmc.Halton(n_geometry + n_modes + n_current, scramble=True, seed=911)
    unit = 0.15 + 0.70 * engine.random(args.cases)

    g_lo, g_hi = np.asarray(model.lower), np.asarray(model.upper)
    a_lo = np.asarray(cfg.initial_lower[:n_modes])
    a_hi = np.asarray(cfg.initial_upper[:n_modes])
    # Geometry training prepends normalized geometry to operating bounds.
    u_lo = np.asarray(cfg.operating_lower[n_geometry:n_geometry + n_current])
    u_hi = np.asarray(cfg.operating_upper[n_geometry:n_geometry + n_current])

    cases = []
    for row in unit:
        cursor = 0
        geometry = g_lo + row[cursor:cursor+n_geometry] * (g_hi-g_lo)
        cursor += n_geometry
        initial = a_lo + row[cursor:cursor+n_modes] * (a_hi-a_lo)
        cursor += n_modes
        current = u_lo + row[cursor:cursor+n_current] * (u_hi-u_lo)
        validation = validate_geometry_trajectory(
            model,
            args.times,
            geometry=geometry,
            a0=initial,
            operating=current,
            full_electromagnetics=True,
        )
        cases.append(validation)

    result = {
        "used_for_training": False,
        "model_scope": "magnetoquasistatic electromagnetics + conduction heat transfer; no fluid flow",
        "mass_residual_equivalence": certify_geometry_mass_residual_equivalence(model),
        "sampled_contractivity": sampled_geometry_contractivity(model, validation=True, seed=701),
        "held_out_cases": cases,
        "limitations": [
            "full sparse EM + Radau uses the saved thermal ROM and therefore does not validate thermal-rank truncation",
            "mesh and outer-domain convergence require separate studies",
            "sampled contractivity is not a continuous-domain kappa certificate",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
