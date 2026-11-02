"""BFZI reference convergence exploration, not a certification.

Compare the legacy angular-ray quadrature with disjoint near-box Gauss
shells; then refine the shell normal/tangential rules, far-field rays, trial
basis, softening, and outer domain *independently*.

    python run.py --mode bfzi-convergence
    python run.py --mode bfzi-convergence --full
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
    base = dict(
        radial_order=4, polar_order=6, azimuthal_order=8,
        field_softening=0.025, shell_normal_order=3,
        shell_tangential_order=4,
    )
    # Format: label, scene, face basis count, face refinement, configuration.
    configurations = [
        ("ray", scene, 1, 1, SeaMQSConfig(
            **base, quadrature_scheme="ray", wall_layers=2,
        )),
        ("box_shell", scene, 1, 1, SeaMQSConfig(**base)),
        ("normal_4", scene, 1, 1, SeaMQSConfig(
            **{**base, "shell_normal_order": 4},
        )),
        ("tangent_5", scene, 1, 1, SeaMQSConfig(
            **{**base, "shell_tangential_order": 5},
        )),
        ("outer_cubature", scene, 1, 1, SeaMQSConfig(
            **{**base, "radial_order": 5, "polar_order": 8,
               "azimuthal_order": 12},
        )),
        ("face_modes_2", scene, 2, 1, SeaMQSConfig(**base)),
        ("outer_radius_070", replace(scene, outer_radius=0.70),
         1, 1, SeaMQSConfig(**base)),
        ("softening_018", scene, 1, 1, SeaMQSConfig(
            **{**base, "field_softening": 0.018},
        )),
    ]
    if full:
        configurations.extend([
            ("tangent_6", scene, 1, 1, SeaMQSConfig(
                **{**base, "shell_tangential_order": 6},
            )),
            ("normal_5", scene, 1, 1, SeaMQSConfig(
                **{**base, "shell_normal_order": 5},
            )),
            ("finer_outer", scene, 1, 1, SeaMQSConfig(
                **{**base, "radial_order": 6, "polar_order": 12,
                   "azimuthal_order": 16},
            )),
            ("face_refined", scene, 2, 2, SeaMQSConfig(**base)),
            ("outer_radius_085", replace(scene, outer_radius=0.85),
             1, 1, SeaMQSConfig(**base)),
            ("softening_012", scene, 1, 1, SeaMQSConfig(
                **{**base, "field_softening": 0.012},
            )),
        ])

    probes = np.array([
        [0.0, 0.0, 0.0],
        [0.13, 0.0, 0.0],
        [0.0, 0.13, 0.0],
        [0.0, 0.0, 0.22],
    ])
    currents = np.array([1.0 + 0.2j, -0.4 + 0.5j])
    results = {}
    sources = {}
    for name, domain, face_modes, face_refinement, config in configurations:
        if not domain.in_seawater(probes).all():
            raise ValueError("fixed convergence probes must be in seawater")
        key = (domain.outer_radius, domain.outer_center)
        if key not in sources:
            sources[key] = build_bfzi_source(domain, 200_000., order=4)
        result = solve_seawater_reference(
            sources[key],
            basis=build_sea_basis(
                domain, modes_per_face=face_modes,
                face_refinement=face_refinement,
            ),
            config=config,
        )
        box_volume = sum(
            8.0 * np.prod(a.enclosure.half_extents)
            for a in domain.assemblies
        )
        volume = 4.0 * np.pi * domain.outer_radius**3 / 3.0 - box_volume
        volume_error = abs(result.integration_weights.sum() / volume - 1.0)
        results[name] = (
            result.impedance_seawater,
            result.seawater_loss(probes, currents),
        )
        print(
            f"{name:>18}: nodes={len(result.integration_points):6d}, "
            f"modes={result.resistance.shape[0]:3d}, "
            f"volume_err={volume_error:.3g}, "
            f"Zsw={result.impedance_seawater.tolist()}",
            flush=True,
        )

    pairs = [
        ("box_shell", "ray"),
        ("normal_4", "box_shell"),
        ("tangent_5", "box_shell"),
        ("outer_cubature", "box_shell"),
        ("face_modes_2", "box_shell"),
        ("outer_radius_070", "box_shell"),
        ("softening_018", "box_shell"),
    ]
    if full:
        pairs.extend([
            ("tangent_6", "tangent_5"),
            ("normal_5", "normal_4"),
            ("finer_outer", "outer_cubature"),
            ("face_refined", "face_modes_2"),
            ("outer_radius_085", "outer_radius_070"),
            ("softening_012", "softening_018"),
        ])
    print("\nRelative changes in seawater Z and fixed-point Joule field:")
    for fine, coarse in pairs:
        z1, q1 = results[fine]
        z0, q0 = results[coarse]
        dz = np.linalg.norm(z1 - z0) / max(np.linalg.norm(z0), 1e-30)
        dq = np.linalg.norm(q1 - q0) / max(np.linalg.norm(q0), 1e-30)
        print(f"{fine:>18} vs {coarse:<18}  dZ={dz:.5g}  dq={dq:.5g}")

    if full:
        air = sources[(scene.outer_radius, scene.outer_center)]
        for name, new in (
            ("air_self_panels", build_bfzi_source(
                scene, 200_000., order=4,
                air_panel_length_factor=2.0, air_panel_gauss_order=4,
            )),
            ("air_mutual_source", build_bfzi_source(
                scene, 200_000., order=6,
            )),
        ):
            change = np.linalg.norm(
                new.inductance_air - air.inductance_air
            ) / max(np.linalg.norm(air.inductance_air), 1e-30)
            print(f"{name:>18} relative air-L change: {change:.5g}")
    print("NOT CERTIFIED: resolve independent refinement sequences before using absolute port predictions.")


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--full"):
        raise SystemExit("usage: python -m examples.bfzi_convergence [--full]")
    main(full=len(sys.argv) == 2)
