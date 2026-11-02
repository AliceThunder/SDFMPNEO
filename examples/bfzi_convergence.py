"""BFZI reference convergence exploration, not a certification.

Compare the legacy angular-ray quadrature with disjoint near-box Gauss
shells; then refine the shell normal/tangential rules, far-field rays, trial
basis, softening, and outer domain *independently*.

    python run.py --mode bfzi-convergence
    python run.py --mode bfzi-convergence --full
    python run.py --mode bfzi-convergence --basis
    python run.py --mode bfzi-convergence --basis-green  # independent Green L and softening
    python run.py --mode bfzi-convergence --basis-domain  # independent outer domain
    python run.py --mode bfzi-convergence --basis-local   # same-L R/K ladder
    python run.py --mode bfzi-convergence --basis-angular # composite sphere angle tests
    python run.py --mode bfzi-convergence --basis-geometry # geometry-guided mu panels
    python run.py --mode bfzi-convergence --basis-verify  # expensive fine Green L
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


def basis_convergence(*, verify: bool = False, green: bool = False,
                      domain: bool = False, local: bool = False,
                      angular: bool = False, geometry: bool = False):
    """Four nested Galerkin spaces with shared fine R/K and coarse L cubatures.

    Target the broad faces and actual closed-port field curls before escalating to
    all-face patch refinement. Only the enriched teacher's expensive
    Coulomb R/L/K assembly is computed.
    """
    scene = build_demo_scene()
    # Resolve the locally sharp curl-mode resistance and true source
    # coupling on a finer rule. The costly nonlocal L double integral uses
    # the earlier 5/8/12/3/4 cubature with the *same* trial functions
    # and whitening; this isolates the measured 11-27% near-wall R/K
    # aliasing without raising Green cost quadratically.
    config = SeaMQSConfig(
        radial_order=6, polar_order=10, azimuthal_order=16,
        shell_normal_order=4, shell_tangential_order=5,
        field_softening=0.025,
        inductance_quadrature=(5, 8, 12, 3, 4),
        max_radial_panel_width=0.18 if domain else None,
    )
    source = build_bfzi_source(scene, 200_000., order=4)
    enriched = build_sea_basis(
        scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True, sources=source,
        include_source_guided=True,
    )
    print("Building enriched 63-mode Galerkin operator...",
          f"raw modes={enriched.size}, "
          "fine R/K/Joule cubature; independently coarse L", flush=True)
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
    # A posteriori residual in the same enriched physical Galerkin space.
    # This distinguishes a genuine missing trial direction from merely
    # redistributing energy among coarse basis functions.
    system = (
        teacher.resistance + 1j * source.omega * teacher.inductance
    )
    forcing = 1j * source.omega * teacher.coupling
    inverse_resistance_forcing = np.linalg.solve(
        teacher.resistance, forcing,
    )
    forcing_norm = np.sqrt(max(
        float(np.real(np.vdot(forcing, inverse_resistance_forcing))),
        1e-60,
    ))
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
        lift, *_ = np.linalg.lstsq(
            teacher.transform,
            solution.transform @ solution.unit_current_modes,
            rcond=None,
        )
        residual = system @ lift + forcing
        relative_residual = np.sqrt(max(
            float(np.real(np.vdot(
                residual, np.linalg.solve(teacher.resistance, residual),
            ))), 0.0,
        )) / forcing_norm
        stages.append((label, z, q, p))
        print(
            f"{label:>20}: modes={solution.resistance.shape[0]:3d}, "
            f"sea_power={p:.8g} W, "
            f"relative_Rinv_residual={relative_residual:.5g}, "
            f"Zsw={z.tolist()}",
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
    # The enriched solution was assembled on the fine R/K cubature. Now
    # reintegrate that *same solution* using independent near-wall and
    # far-field rules. This is much cheaper than a second O(Q^2) Green
    # assembly and catches deceptively well-closing aliased R/K matrices.
    print("\nHeld-out R/K integration and re-solved response (same Green L):")
    holdout_configs = (
        ("wall_heldout", replace(
            config, shell_normal_order=5, shell_tangential_order=6,
        )),
        ("far_heldout", replace(
            config, radial_order=7, polar_order=12, azimuthal_order=20,
        )),
    )
    for name, cfg in holdout_configs:
        points, weights = scene.box_shell_quadrature(
            cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
            wall_scale=enriched.wall_scale,
            shell_width_factor=cfg.shell_width_factor,
            shell_normal_order=cfg.shell_normal_order,
            shell_tangential_order=cfg.shell_tangential_order,
            outer_taper_width=enriched.outer_taper_width,
            max_radial_panel_width=cfg.max_radial_panel_width,
            polar_panel_count=cfg.polar_panel_count,
            azimuthal_panel_order=cfg.azimuthal_panel_order,
        )
        check = teacher.independent_cubature_metrics(
            points, weights, currents, probes=probes,
        )
        print(
            f"{name:>20}: nodes={len(points):6d}, "
            f"dR={check['relative_resistance_difference']:.6g}, "
            f"dK={check['relative_coupling_difference']:.6g}, "
            f"frozen_dP={check['relative_power_difference']:.6g}, "
            f"residual={check['relative_heldout_residual']:.6g}",
            flush=True,
        )
        print(
            f"{'re_solved':>20}: "
            f"dZ={check['relative_resolved_impedance_change']:.6g}, "
            f"dq={check['relative_resolved_probe_change']:.6g}, "
            f"dP={check['relative_resolved_power_change']:.6g}, "
            f"sea_power={check['resolved_power']:.8g} W",
            flush=True,
        )
    if local:
        print("\nR/K-only refinement ladder (fixed Green L and physical basis):")
        refinements = (
            ("radial_only", replace(config, radial_order=7)),
            ("angles_only", replace(
                config, polar_order=12, azimuthal_order=20,
            )),
            ("wall_only", replace(
                config, shell_normal_order=5, shell_tangential_order=6,
            )),
            ("combined", replace(
                config, radial_order=7, polar_order=12,
                azimuthal_order=20, shell_normal_order=5,
                shell_tangential_order=6,
            )),
            ("combined_finer", replace(
                config, radial_order=8, polar_order=14,
                azimuthal_order=24, shell_normal_order=6,
                shell_tangential_order=7,
            )),
        )
        original_z = teacher.impedance_seawater
        original_q = teacher.seawater_loss(probes, currents)
        original_p = teacher.port_seawater_power(currents)
        preceding_combined = teacher
        for label, cfg in refinements:
            local_points, local_weights = scene.box_shell_quadrature(
                cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=cfg.shell_width_factor,
                shell_normal_order=cfg.shell_normal_order,
                shell_tangential_order=cfg.shell_tangential_order,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=cfg.max_radial_panel_width,
                polar_panel_count=cfg.polar_panel_count,
                azimuthal_panel_order=cfg.azimuthal_panel_order,
            )
            updated = teacher.reintegrate_local_response(
                local_points, local_weights,
            )
            new_z = updated.impedance_seawater
            new_q = updated.seawater_loss(probes, currents)
            new_p = updated.port_seawater_power(currents)
            dz = np.linalg.norm(new_z - original_z) / max(
                np.linalg.norm(original_z), 1e-30,
            )
            dq = np.linalg.norm(new_q - original_q) / max(
                np.linalg.norm(original_q), 1e-30,
            )
            dp = abs(new_p - original_p) / max(abs(original_p), 1e-30)
            previous_z = preceding_combined.impedance_seawater
            previous_q = preceding_combined.seawater_loss(probes, currents)
            previous_p = preceding_combined.port_seawater_power(currents)
            step_dz = np.linalg.norm(new_z - previous_z) / max(
                np.linalg.norm(previous_z), 1e-30,
            )
            step_dq = np.linalg.norm(new_q - previous_q) / max(
                np.linalg.norm(previous_q), 1e-30,
            )
            step_dp = abs(new_p - previous_p) / max(abs(previous_p), 1e-30)
            print(
                f"{label:>20}: nodes={len(local_points):6d}, "
                f"base_dZ={dz:.6g}, base_dq={dq:.6g}, "
                f"base_dP={dp:.6g}, "
                f"step_dZ={step_dz:.6g}, "
                f"step_dq={step_dq:.6g}, "
                f"step_dP={step_dp:.6g}, "
                f"sea_power={new_p:.8g} W",
                flush=True,
            )
            # Single-axis cases compare with the original reference.
            # The combined cases alone form the sequential refinement.
            if label in ("combined", "combined_finer"):
                preceding_combined = updated

    if angular:
        # A comparison with equal total mu nodes but shifted Gauss panels
        # is a consistency check, not convergence. Hold phi sectors,
        # radial cubature, wall Gauss, basis, and L fixed while increasing
        # *only* the Gaussian order within each of two mu panels.
        print("\nSame-family polar Gauss order ladder (fixed phi/radial/wall/L):")
        polar_trials = (
            ("phi2_global10", replace(
                config, polar_panel_count=1, azimuthal_panel_order=2,
            )),
            ("polar2_mu10", replace(
                config, polar_panel_count=2, azimuthal_panel_order=2,
            )),
            ("polar2_mu14", replace(
                config, polar_order=14, polar_panel_count=2,
                azimuthal_panel_order=2,
            )),
            ("polar2_mu18", replace(
                config, polar_order=18, polar_panel_count=2,
                azimuthal_panel_order=2,
            )),
            ("polar2_mu22", replace(
                config, polar_order=22, polar_panel_count=2,
                azimuthal_panel_order=2,
            )),
            ("polar2_mu22_phi3", replace(
                config, polar_order=22, polar_panel_count=2,
                azimuthal_panel_order=3,
            )),
        )
        base_z = teacher.impedance_seawater
        base_q = teacher.seawater_loss(probes, currents)
        base_p = teacher.port_seawater_power(currents)
        previous = None
        for label, cfg in polar_trials:
            points, weights = scene.box_shell_quadrature(
                cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=cfg.shell_width_factor,
                shell_normal_order=cfg.shell_normal_order,
                shell_tangential_order=cfg.shell_tangential_order,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=cfg.max_radial_panel_width,
                polar_panel_count=cfg.polar_panel_count,
                azimuthal_panel_order=cfg.azimuthal_panel_order,
            )
            resolved = teacher.reintegrate_local_response(points, weights)
            z = resolved.impedance_seawater
            q = resolved.seawater_loss(probes, currents)
            p = resolved.port_seawater_power(currents)
            # Update K alone, using the same independent nodes already
            # evaluated for the full R/K solve. Avoid recomputing source
            # fields or any Green L quadrature.
            k_only = teacher.resolve_with_coupling(resolved.coupling)
            k_q = k_only.seawater_loss(probes, currents)
            k_p = k_only.port_seawater_power(currents)
            def relative(a, b):
                return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
            dr = relative(resolved.resistance, teacher.resistance)
            dk = relative(resolved.coupling, teacher.coupling)
            dz = relative(z, base_z)
            dq = relative(q, base_q)
            dp = abs(p - base_p) / max(abs(base_p), 1e-30)
            print(
                f"{label:>20}: nodes={len(points):6d}, "
                f"dR={dr:.6g}, dK={dk:.6g}, "
                f"base_dZ={dz:.6g}, base_dq={dq:.6g}, "
                f"base_dP={dp:.6g}, sea_power={p:.8g} W",
                flush=True,
            )
            print(
                f"{'K_only_re_solved':>20}: "
                f"dZ={relative(k_only.impedance_seawater, base_z):.6g}, "
                f"dq={relative(k_q, base_q):.6g}, "
                f"dP={abs(k_p - base_p) / max(abs(base_p), 1e-30):.6g}",
                flush=True,
            )
            print(
                f"{'probe_q':>20}: "
                + " ".join(f"{value:.8g}" for value in q),
                flush=True,
            )
            if previous is not None:
                old = previous[1]
                old_q = old.seawater_loss(probes, currents)
                old_p = old.port_seawater_power(currents)
                point_relative = np.abs(q - old_q) / np.maximum(
                    np.abs(old_q), 1e-30,
                )
                print(
                    f"{'step_from_previous':>20}: "
                    f"from={previous[0]}, "
                    f"dZ={relative(z, old.impedance_seawater):.6g}, "
                    f"dq={relative(q, old_q):.6g}, "
                    f"max_probe_dq={np.max(point_relative):.6g}, "
                    f"dP={abs(p - old_p) / max(abs(old_p), 1e-30):.6g}",
                    flush=True,
                )
            previous = label, resolved
        print(
            "The polar2_mu10/14/18/22 rows increase ONLY mu Gauss "
            "degree on the same two panels; the last phi3 row "
            "separately checks azimuthal resolution.",
            flush=True,
        )

    if geometry:
        # Refine Gauss order within fixed package-projected mu panels.
        # The same source, raw basis, R-whitening and Green L are retained.
        print("\nGeometry-aligned polar R/K ladder (fixed Green L):")
        previous = None
        stable_geometry = None
        for label, mu_order, phi_order in (
            ("geometry_mu3", 3, 2),
            ("geometry_mu4", 4, 2),
            ("geometry_mu5", 5, 2),
            ("geometry_mu6", 6, 2),
            ("geometry_mu8", 8, 2),
            ("geometry_mu8_phi3", 8, 3),
        ):
            cfg = replace(
                config, polar_geometry_order=mu_order,
                azimuthal_panel_order=phi_order,
            )
            points, weights = scene.box_shell_quadrature(
                cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=cfg.shell_width_factor,
                shell_normal_order=cfg.shell_normal_order,
                shell_tangential_order=cfg.shell_tangential_order,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=cfg.max_radial_panel_width,
                polar_geometry_order=cfg.polar_geometry_order,
                azimuthal_panel_order=cfg.azimuthal_panel_order,
            )
            full = teacher.reintegrate_local_response(points, weights)
            k_only = teacher.resolve_with_coupling(full.coupling)
            r_only = full.resolve_with_coupling(teacher.coupling)
            z = full.impedance_seawater
            q = full.seawater_loss(probes, currents)
            power = full.port_seawater_power(currents)
            base_q = teacher.seawater_loss(probes, currents)
            base_p = teacher.port_seawater_power(currents)
            def relative(a, b):
                return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
            print(
                f"{label:>20}: nodes={len(points):6d}, "
                f"dR={relative(full.resistance, teacher.resistance):.6g}, "
                f"dK={relative(full.coupling, teacher.coupling):.6g}, "
                f"base_dZ={relative(z, teacher.impedance_seawater):.6g}, "
                f"base_dq={relative(q, base_q):.6g}, "
                f"base_dP={abs(power-base_p)/max(abs(base_p),1e-30):.6g}, "
                f"P={power:.8g} W",
                flush=True,
            )
            print(
                f"{'K_only':>20}: "
                f"dZ={relative(k_only.impedance_seawater, teacher.impedance_seawater):.6g}, "
                f"dq={relative(k_only.seawater_loss(probes, currents), base_q):.6g}; "
                f"R_only_dZ={relative(r_only.impedance_seawater, teacher.impedance_seawater):.6g}, "
                f"R_only_dq={relative(r_only.seawater_loss(probes, currents), base_q):.6g}",
                flush=True,
            )
            print(
                f"{'probe_q':>20}: "
                + " ".join(f"{value:.8g}" for value in q),
                flush=True,
            )
            if previous is not None:
                old_label, old = previous
                old_q = old.seawater_loss(probes, currents)
                old_p = old.port_seawater_power(currents)
                print(
                    f"{'step_from_previous':>20}: from={old_label}, "
                    f"dZ={relative(z, old.impedance_seawater):.6g}, "
                    f"dq={relative(q, old_q):.6g}, "
                    f"max_probe_dq={np.max(np.abs(q-old_q)/np.maximum(np.abs(old_q),1e-30)):.6g}, "
                    f"dP={abs(power-old_p)/max(abs(old_p),1e-30):.6g}",
                    flush=True,
                )
            if label == "geometry_mu8":
                stable_geometry = full
            previous = label, full

        # With the converged-in-angle geometry_mu8 rule as the *new*
        # comparator, independently perturb radial and wall cubature.
        # These are not successive refinements of the same axis.
        if stable_geometry is None:
            raise RuntimeError("missing geometry_mu8 comparator")
        print("\nHeld-out radial / near-wall R/K at geometry_mu8 (same L):")
        geometry_q = stable_geometry.seawater_loss(probes, currents)
        geometry_p = stable_geometry.port_seawater_power(currents)
        geometry_z = stable_geometry.impedance_seawater
        for label, changes in (
            ("radial7", {"radial_order": 7}),
            ("wall5_6", {"shell_normal_order": 5, "shell_tangential_order": 6}),
            ("radial7_wall5_6", {
                "radial_order": 7, "shell_normal_order": 5,
                "shell_tangential_order": 6,
            }),
        ):
            cfg = replace(
                config, polar_geometry_order=8,
                azimuthal_panel_order=2, **changes,
            )
            points, weights = scene.box_shell_quadrature(
                cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=cfg.shell_width_factor,
                shell_normal_order=cfg.shell_normal_order,
                shell_tangential_order=cfg.shell_tangential_order,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=cfg.max_radial_panel_width,
                polar_geometry_order=cfg.polar_geometry_order,
                azimuthal_panel_order=cfg.azimuthal_panel_order,
            )
            modified = teacher.reintegrate_local_response(points, weights)
            now_q = modified.seawater_loss(probes, currents)
            now_p = modified.port_seawater_power(currents)
            dz = np.linalg.norm(
                modified.impedance_seawater - geometry_z,
            ) / max(np.linalg.norm(geometry_z), 1e-30)
            dq = np.linalg.norm(
                now_q - geometry_q,
            ) / max(np.linalg.norm(geometry_q), 1e-30)
            dr = np.linalg.norm(
                modified.resistance - stable_geometry.resistance,
            ) / max(np.linalg.norm(stable_geometry.resistance), 1e-30)
            dk = np.linalg.norm(
                modified.coupling - stable_geometry.coupling,
            ) / max(np.linalg.norm(stable_geometry.coupling), 1e-30)
            max_dq = np.max(np.abs(
                now_q - geometry_q,
            ) / np.maximum(np.abs(geometry_q), 1e-30))
            print(
                f"{label:>20}: nodes={len(points):6d}, "
                f"vs_geom8_dR={dr:.6g}, vs_geom8_dK={dk:.6g}, "
                f"vs_geom8_dZ={dz:.6g}, vs_geom8_dq={dq:.6g}, "
                f"max_probe_dq={max_dq:.6g}, "
                f"vs_geom8_dP={abs(now_p - geometry_p) / max(abs(geometry_p), 1e-30):.6g}, "
                f"P={now_p:.8g} W",
                flush=True,
            )

    if green:
        # Keep the original 63-mode R/K, basis, whitening, power cubature,
        # and port source fixed. Vary only the nonlocal L double integration.
        # Comparisons for different softenings use IDENTICAL Green nodes.
        # These moderate-size sweeps are cheaper than --basis-verify, but
        # still scale quadratically in the Green node count.
        print("\nIndependent Green L cubature / softening re-solves (fixed R/K):")
        green_trials = (
            ("green_wall", (5, 8, 12, 3, 5), (config.field_softening,)),
            ("green_outer", (5, 10, 16, 3, 4), (config.field_softening,)),
            ("green_softening", (5, 8, 12, 3, 4), (0.018, 0.012)),
        )
        for label, orders, softenings in green_trials:
            radial, polar, azimuthal, normal, tangential = orders
            green_points, green_weights = scene.box_shell_quadrature(
                radial, polar, azimuthal,
                wall_scale=enriched.wall_scale,
                shell_width_factor=config.shell_width_factor,
                shell_normal_order=normal,
                shell_tangential_order=tangential,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=config.max_radial_panel_width,
                polar_panel_count=config.polar_panel_count,
                azimuthal_panel_order=config.azimuthal_panel_order,
            )
            checks = teacher.independent_green_metrics(
                green_points, green_weights, currents, probes=probes,
                softenings=softenings,
                kernel_chunk_size=config.kernel_chunk_size,
            )
            for check in checks:
                print(
                    f"{label:>20}: nodes={len(green_points):6d}, "
                    f"a={check['softening']:.5g} m, "
                    f"dL={check['relative_inductance_difference']:.6g}, "
                    f"self_trace={check['green_diagonal_trace_fraction']:.6g}, "
                    f"re_solved_dZ={check['relative_resolved_impedance_change']:.6g}, "
                    f"dq={check['relative_resolved_probe_change']:.6g}, "
                    f"dP={check['relative_resolved_power_change']:.6g}, "
                    f"residual={check['relative_heldout_residual']:.6g}",
                    flush=True,
                )

    if domain:
        # Unlike a held-out Green integral, a larger physical seawater
        # domain changes the boundary-conditioned trial functions. These
        # are fresh complete R/K/L Galerkin solves at FIXED mode hierarchy,
        # softening, frequency and cubature orders, not a Green-only test.
        print("\nOuter-domain sequence (independent 63-mode solves):")
        baseline_z, baseline_q = stages[-1][1:3]
        baseline_power = stages[-1][3]
        previous_z, previous_q, previous_power = (
            baseline_z, baseline_q, baseline_power
        )
        for radius in (0.70, 0.85, 1.05, 1.30):
            larger_scene = replace(scene, outer_radius=radius)
            larger_source = build_bfzi_source(
                larger_scene, 200_000., order=4,
            )
            larger_basis = build_sea_basis(
                larger_scene, modes_per_face=2, face_refinement=2,
                refine_thin_faces_only=True, sources=larger_source,
                include_source_guided=True,
            )
            larger = solve_seawater_reference(
                larger_source, basis=larger_basis, config=config,
            )
            # Identical raw trial fields at physical near-coil probes
            # distinguish boundary-position effects from basis renormalization.
            near_mode_change = (
                np.linalg.norm(
                    larger_basis.evaluate(probes) - enriched.evaluate(probes)
                ) / max(np.linalg.norm(enriched.evaluate(probes)), 1e-30)
            )
            z = larger.impedance_seawater
            q = larger.seawater_loss(probes, currents)
            power = larger.port_seawater_power(currents)
            dz = np.linalg.norm(z - previous_z) / max(
                np.linalg.norm(previous_z), 1e-30
            )
            dq = np.linalg.norm(q - previous_q) / max(
                np.linalg.norm(previous_q), 1e-30
            )
            dp = abs(power - previous_power) / max(
                abs(previous_power), 1e-30
            )
            print(
                f"radius={radius:.3g} m: "
                f"nodes={len(larger.integration_points):6d}, "
                f"modes={larger.resistance.shape[0]:3d}, "
                f"raw_mode_dW={near_mode_change:.3g}, "
                f"step_dZ={dz:.6g}, step_dq={dq:.6g}, "
                f"step_dP={dp:.6g}, sea_power={power:.8g} W",
                flush=True,
            )
            previous_z, previous_q, previous_power = z, q, power

    if verify:
        # Optional expensive second REFERENCE solve. Unlike the cheap
        # holdouts above, this changes the full nonlocal Green L operator
        # and recomputes the coupled 63-mode current state.
        fine_config = replace(config, inductance_quadrature=None)
        print(
            "\nFull 63-mode nonlocal Green L verification on fine "
            "R/K/Joule cubature (expensive)...",
            flush=True,
        )
        fine = solve_seawater_reference(
            source, basis=enriched, config=fine_config,
        )
        fine_z = fine.impedance_seawater
        fine_q = fine.seawater_loss(probes, currents)
        base_z, base_q, base_p = stages[-1][1:]
        fine_p = fine.port_seawater_power(currents)
        dz = np.linalg.norm(fine_z - base_z) / max(
            np.linalg.norm(base_z), 1e-30,
        )
        dq = np.linalg.norm(fine_q - base_q) / max(
            np.linalg.norm(base_q), 1e-30,
        )
        dp = abs(fine_p - base_p) / max(abs(base_p), 1e-30)
        print(
            f"refined_63: nodes={len(fine.integration_points):6d}, "
            f"retained_modes={fine.resistance.shape[0]}, "
            f"dZ={dz:.6g}, dq={dq:.6g}, dP={dp:.6g}, "
            f"sea_power={fine_p:.8g} W",
            flush=True,
        )
    print(
        "NOT CERTIFIED: separate outer-domain, basis-completeness, "
        "and Green-regularization convergence are still required."
    )


def main(
    full: bool = False, basis_only: bool = False,
    basis_verify: bool = False, basis_green: bool = False,
    basis_domain: bool = False, basis_local: bool = False,
    basis_angular: bool = False, basis_geometry: bool = False,
):
    if (basis_only or basis_verify or basis_green or basis_domain
            or basis_local or basis_angular or basis_geometry):
        return basis_convergence(
            verify=basis_verify, green=basis_green, domain=basis_domain,
            local=basis_local, angular=basis_angular,
            geometry=basis_geometry,
        )
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
        len(sys.argv) == 2
        and sys.argv[1] not in ("--full", "--basis", "--basis-verify", "--basis-green", "--basis-domain", "--basis-local", "--basis-angular", "--basis-geometry")
    ):
        raise SystemExit(
            "usage: python -m examples.bfzi_convergence "
            "[--full|--basis|--basis-local|--basis-angular|--basis-geometry|--basis-green|--basis-domain|--basis-verify]"
        )
    main(
        full="--full" in sys.argv[1:],
        basis_only="--basis" in sys.argv[1:],
        basis_verify="--basis-verify" in sys.argv[1:],
        basis_green="--basis-green" in sys.argv[1:],
        basis_domain="--basis-domain" in sys.argv[1:],
        basis_local="--basis-local" in sys.argv[1:],
        basis_angular="--basis-angular" in sys.argv[1:],
        basis_geometry="--basis-geometry" in sys.argv[1:],
    )
