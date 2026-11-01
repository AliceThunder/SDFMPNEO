"""Independent BFZI seawater convergence probes (not certification).

Quick (default) uses a modest mode count; --full also tests more localized
box-face modes and higher cubature resolution. Results compare both two-port
reaction matrices and spatial seawater Joule density at fixed wet probes.
Commands from repository root:
    python -m examples.bfzi_convergence
    python -m examples.bfzi_convergence --full
    python run.py --mode bfzi-convergence [--full]
"""
from __future__ import annotations

from dataclasses import replace
import sys
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import (
    SeaMQSConfig, build_bfzi_source, build_sea_basis,
    solve_seawater_reference,
)


def main(full: bool = False):
    scene = build_demo_scene()
    base = dict(radial_order=4, polar_order=6, azimuthal_order=8,
                field_softening=0.025)
    configs = [
        ("ungraded", scene, 1, 1, SeaMQSConfig(**base, wall_layers=0)),
        ("one_wall_layer", scene, 1, 1, SeaMQSConfig(**base, wall_layers=1)),
        ("two_wall_layers", scene, 1, 1, SeaMQSConfig(**base, wall_layers=2)),
        ("two_face_modes", scene, 2, 1, SeaMQSConfig(**base)),
        ("refined_cubature", scene, 2, 1, SeaMQSConfig(
            radial_order=5, polar_order=8, azimuthal_order=12,
            field_softening=0.025,
        )),
        ("larger_domain", replace(scene, outer_radius=0.70), 2, 1,
         SeaMQSConfig(**base)),
        ("smaller_softening", scene, 2, 1, SeaMQSConfig(
            **{**base, "field_softening": 0.018},
        )),
    ]
    if full:
        configs.extend([
            ("refined_face_basis", scene, 2, 2, SeaMQSConfig(**base)),
            ("higher_cubature", scene, 2, 1, SeaMQSConfig(
                radial_order=6, polar_order=10, azimuthal_order=16,
                field_softening=0.025,
            )),
            ("smaller_softening_2", scene, 2, 1, SeaMQSConfig(
                **{**base, "field_softening": 0.012},
            )),
            ("larger_domain_2", replace(scene, outer_radius=0.85), 2, 1,
             SeaMQSConfig(**base)),
        ])
    # Probe locations are strictly outside both sealed boxes for every
    # configuration and far from the artificial outer boundary.
    probes = np.array([
        [0.0, 0.0, 0.0],
        [0.13, 0.0, 0.0],
        [0.0, 0.13, 0.0],
        [0.0, 0.0, 0.22],
    ])
    currents = np.array([1.0 + 0.2j, -0.4 + 0.5j])
    assert all(domain.in_seawater(probes).all() for _, domain, *_ in configs)
    results = {}
    source_cache = {}
    for name, domain, face_modes, face_refinement, config in configs:
        # Air/port source quadrature does not change when refining the sea
        # trial space or Green kernel, so do not recompute it unnecessarily.
        key = (domain.outer_radius, domain.outer_center)
        if key not in source_cache:
            source_cache[key] = build_bfzi_source(
                domain, 200_000., order=4,
            )
        result = solve_seawater_reference(
            source_cache[key],
            basis=build_sea_basis(
                domain, modes_per_face=face_modes,
                face_refinement=face_refinement,
            ),
            config=config,
        )
        results[name] = (
            result.impedance_seawater, result.seawater_loss(probes, currents)
        )
        print(f"{name:>21}: Q={len(result.integration_points):6d}, "
              f"modes={result.resistance.shape[0]:3d}, "
              f"Zsw={result.impedance_seawater.tolist()}", flush=True)
    pairs = [
        ("one_wall_layer", "ungraded"),
        ("two_wall_layers", "one_wall_layer"),
        ("two_face_modes", "two_wall_layers"),
        ("refined_cubature", "two_face_modes"),
        ("larger_domain", "two_face_modes"),
        ("smaller_softening", "two_face_modes"),
    ]
    if full:
        pairs.extend([
            ("refined_face_basis", "two_face_modes"),
            ("higher_cubature", "refined_cubature"),
            ("smaller_softening_2", "smaller_softening"),
            ("larger_domain_2", "larger_domain"),
        ])
    print("\nOne-axis relative changes (Zsw / fixed-point seawater Joule):")
    for refined, coarse in pairs:
        z1, q1 = results[refined]
        z0, q0 = results[coarse]
        dz = np.linalg.norm(z1 - z0) / max(np.linalg.norm(z0), 1e-30)
        dq = np.linalg.norm(q1 - q0) / max(np.linalg.norm(q0), 1e-30)
        print(f"{refined:>21} vs {coarse:<20}: "
              f"dZ={dz:.5g}, dq={dq:.5g}")
    print("NOT CERTIFIED: confirm successive refinements and independent truth.")


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--full"):
        raise SystemExit("usage: python -m examples.bfzi_convergence [--full]")
    main(full=len(sys.argv) == 2)
