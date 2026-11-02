"""BFZI reference convergence exploration, not a certification.

Compare the legacy angular-ray quadrature with disjoint near-box Gauss
shells; then refine the shell normal/tangential rules, far-field rays, trial
basis, softening, and outer domain *independently*.

    python run.py --mode bfzi-convergence
    python run.py --mode bfzi-convergence --full
    python run.py --mode bfzi-convergence --basis
"""
from __future__ import annotations

from dataclasses import replace
import sys
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import (
    SeaMQSConfig, build_bfzi_source, build_sea_basis,
    solve_seawater_reference, restrict_seawater_response,
)


def _indices_in_enriched_basis(enriched, subset):
    """Identify nested physical curl trial modes without reordering."""
    indices = []
    for candidate in subset.modes:
        matches = [
            index for index, mode in enumerate(enriched.modes)
            if np.array_equal(mode.center, candidate.center)
            and np.array_equal(mode.direction, candidate.direction)
            and mode.width == candidate.width
            and mode.source_port == candidate.source_port
        ]
        if len(matches) != 1:
            raise ValueError("trial spaces are not a strictly nested hierarchy")
        indices.append(matches[0])
    return np.asarray(indices, dtype=int)


def basis_convergence():
    """Four nested Galerkin spaces with ONE shared cubature and Green solve.

    Target the broad faces and actual closed-port field curls before escalating to
    all-face patch refinement. Only the enriched teacher's expensive
    Coulomb R/L/K assembly is computed.
    """
    scene = build_demo_scene()
    config = SeaMQSConfig(
        radial_order=5, polar_order=8, azimuthal_order=12,
        shell_normal_order=3, shell_tangential_order=4,
        field_softening=0.025,
    )
    source = build_bfzi_source(scene, 200_000., order=4)
    enriched = build_sea_basis(
        scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True, sources=source,
        include_source_guided=True,
    )
    print("Building enriched thin-face Galerkin operator...",
          f"raw modes={enriched.size}", flush=True)
    teacher = solve_seawater_reference(
        source, basis=enriched, config=config,
    )
    currents = np.array([1.0 + 0.2j, -0.4 + 0.5j])
    probes = np.array([
        [0.0, 0.0, 0.0],
        [0.13, 0.0, 0.0],
        [0.0, 0.13, 0.0],
        [0.0, 0.0, 0.22],
    ])
    stages = []
    for label, modes_per_face, face_refinement, thin in (
        ("one_tangent", 1, 1, False),
        ("two_tangents", 2, 1, False),
        ("thin_face_patches", 2, 2, True),
        ("guided_port_curls", 2, 2, True),
    ):
        guided = label == "guided_port_curls"
        subset = build_sea_basis(
            scene, modes_per_face=modes_per_face,
            face_refinement=face_refinement,
            refine_thin_faces_only=thin,
            sources=source if guided else None,
            include_source_guided=guided,
        )
        solution = (
            teacher if guided else restrict_seawater_response(
                teacher, _indices_in_enriched_basis(enriched, subset),
            )
        )
        z = solution.impedance_seawater
        q = solution.seawater_loss(probes, currents)
        p = solution.port_seawater_power(currents)
        stages.append((label, z, q, p))
        print(
            f"{label:>20}: modes={solution.resistance.shape[0]:3d}, "
            f"sea_power={p:.8g} W, Zsw={z.tolist()}",
            flush=True,
        )
        print(f"{'fixed_probe_q':>20}: {q.tolist()}", flush=True)
    print("\nNested-basis changes (identical Q, kernel, domain and source):")
    for coarse, fine in zip(stages[:-1], stages[1:]):
        dz = np.linalg.norm(fine[1] - coarse[1]) / max(
            np.linalg.norm(coarse[1]), 1e-30,
        )
        dq = np.linalg.norm(fine[2] - coarse[2]) / max(
            np.linalg.norm(coarse[2]), 1e-30,
        )
        dp = abs(fine[3] - coarse[3]) / max(abs(coarse[3]), 1e-30)
        print(
            f"{coarse[0]} -> {fine[0]}: "
            f"dZ={dz:.6g}, dq={dq:.6g}, dP={dp:.6g}"
        )
    print("NOT CERTIFIED: source-guided and spatial refinements require independent cubature checks.")


def main(full: bool = False, basis_only: bool = False):
    if basis_only:
        return basis_convergence()
    scene = build_demo_scene()
    base = dict(
        radial_order=4, polar_order=6, azimuthal_order=8,
        field_softening=0.025, shell_normal_order=3,
        shell_tangential_order=4,
    )
    # Format: label, scene, face basis count, face refinement, configuration.
    configurations = [
        ("ray", scene, 2, 1, SeaMQSConfig(
            **base, quadrature_scheme="ray", wall_layers=2,
        )),
        ("box_shell", scene, 2, 1, SeaMQSConfig(**base)),
        ("normal_4", scene, 2, 1, SeaMQSConfig(
            **{**base, "shell_normal_order": 4},
        )),
        ("tangent_5", scene, 2, 1, SeaMQSConfig(
            **{**base, "shell_tangential_order": 5},
        )),
        ("outer_radial", scene, 2, 1, SeaMQSConfig(
            **{**base, "radial_order": 6},
        )),
        ("outer_angles", scene, 2, 1, SeaMQSConfig(
            **{**base, "polar_order": 9, "azimuthal_order": 12},
        )),
        ("outer_cubature", scene, 2, 1, SeaMQSConfig(
            **{**base, "radial_order": 5, "polar_order": 8,
               "azimuthal_order": 12},
        )),
        ("shell_width_16", scene, 2, 1, SeaMQSConfig(
            **{**base, "shell_width_factor": 16.0},
        )),
        ("face_modes_1", scene, 1, 1, SeaMQSConfig(**base)),
        ("outer_radius_070", replace(scene, outer_radius=0.70),
         2, 1, SeaMQSConfig(**base)),
        ("softening_018", scene, 2, 1, SeaMQSConfig(
            **{**base, "field_softening": 0.018},
        )),
    ]
    if full:
        configurations.extend([
            ("tangent_6", scene, 2, 1, SeaMQSConfig(
                **{**base, "shell_tangential_order": 6},
            )),
            ("normal_5", scene, 2, 1, SeaMQSConfig(
                **{**base, "shell_normal_order": 5},
            )),
            ("finer_outer", scene, 2, 1, SeaMQSConfig(
                **{**base, "radial_order": 6, "polar_order": 12,
                   "azimuthal_order": 16},
            )),
            ("face_refined", scene, 2, 2, SeaMQSConfig(**base)),
            ("outer_radius_085", replace(scene, outer_radius=0.85),
             2, 1, SeaMQSConfig(**base)),
            ("softening_012", scene, 2, 1, SeaMQSConfig(
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
        ("outer_radial", "box_shell"),
        ("outer_angles", "box_shell"),
        ("outer_cubature", "box_shell"),
        ("shell_width_16", "box_shell"),
        ("box_shell", "face_modes_1"),
        ("outer_radius_070", "box_shell"),
        ("softening_018", "box_shell"),
    ]
    if full:
        pairs.extend([
            ("tangent_6", "tangent_5"),
            ("normal_5", "normal_4"),
            ("finer_outer", "outer_cubature"),
            ("face_refined", "box_shell"),
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
    if len(sys.argv) > 2 or (
        len(sys.argv) == 2 and sys.argv[1] not in ("--full", "--basis")
    ):
        raise SystemExit(
            "usage: python -m examples.bfzi_convergence [--full|--basis]"
        )
    main(
        full="--full" in sys.argv[1:],
        basis_only="--basis" in sys.argv[1:],
    )
