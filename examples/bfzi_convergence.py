"""BFZI one-axis-at-a-time convergence probe (not a certification).

Run from the repository root:
    python -m examples.bfzi_convergence

Compares the same two-port seawater reaction impedance under basis, angular/
radial quadrature, outer-radius and kernel-softening refinements. An individual
small difference is NOT proof of infinite-domain or singular-kernel convergence.
"""
from dataclasses import replace
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import (
    SeaMQSConfig, build_bfzi_source, build_sea_basis,
    solve_seawater_reference,
)


def main():
    scene = build_demo_scene()
    configurations = {
        "ungraded_wall": (scene, 2, SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025, wall_layers=0,
        )),
        "one_wall_layer": (scene, 2, SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025, wall_layers=1,
        )),
        "baseline": (scene, 1, SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025,
        )),
        "enriched_modes": (scene, 2, SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025,
        )),
        "refined_cubature": (scene, 2, SeaMQSConfig(
            radial_order=6, polar_order=10, azimuthal_order=16,
            field_softening=0.025,
        )),
        "larger_domain": (replace(scene, outer_radius=0.7), 2, SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025,
        )),
        "smaller_softening": (scene, 2, SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.018,
        )),
    }
    impedances = {}
    for label, (domain, n_face, config) in configurations.items():
        sources = build_bfzi_source(domain, 200_000.0, order=4)
        result = solve_seawater_reference(
            sources, basis=build_sea_basis(domain, modes_per_face=n_face),
            config=config,
        )
        impedances[label] = result.impedance_seawater
        print(
            f"{label:>19}: nodes={len(result.integration_points):5d},"
            f" resolved_modes={result.resistance.shape[0]:2d},"
            f" reaction={result.impedance_seawater.tolist()}"
        )
    for label, baseline in (
        ("one_wall_layer", "ungraded_wall"),
        ("enriched_modes", "one_wall_layer"),
        ("enriched_modes", "baseline"),
        ("refined_cubature", "enriched_modes"),
        ("larger_domain", "enriched_modes"),
        ("smaller_softening", "enriched_modes"),
    ):
        change = np.linalg.norm(impedances[label] - impedances[baseline])
        scale = max(np.linalg.norm(impedances[baseline]), 1e-30)
        print(f"{label:>19} relative reaction change: {change / scale:.6g}")
    print("NOT CERTIFIED: continue independent refinements until changes stabilize.")


if __name__ == "__main__":
    main()
