"""BFZI reference convergence exploration, not a certification.

Compare the legacy angular-ray quadrature with disjoint near-box Gauss
shells; then refine the shell normal/tangential rules, far-field rays, trial
basis, softening, and outer domain *independently*.

    python run.py --mode bfzi-convergence
    python run.py --mode bfzi-convergence --full
    python run.py --mode bfzi-convergence --basis
    python run.py --mode bfzi-convergence --basis-green  # independent Green L and softening
    python run.py --mode bfzi-convergence --basis-green-refined # converged-wall R/K, independent L
    python run.py --mode bfzi-convergence --basis-green-cross # noncoincident magnetic cross-rule test
    python run.py --mode bfzi-convergence --basis-green-gaussian # PSD Gaussian volume Green test
    python run.py --mode bfzi-convergence --basis-green-ball # exact uniform-ball Coulomb volume kernel
    python run.py --mode bfzi-convergence --basis-ball # nested modes with uniform-ball L
    python run.py --mode bfzi-convergence --basis-ball-check-l # expensive 159-mode magnetic A/B cross-rule
    python run.py --mode bfzi-convergence --basis-domain-ball # outer-domain sequence with uniform-ball L
    python run.py --mode bfzi-convergence --basis-domain  # independent outer domain
    python run.py --mode bfzi-convergence --basis-local   # same-L R/K ladder
    python run.py --mode bfzi-convergence --basis-angular # composite sphere angle tests
    python run.py --mode bfzi-convergence --basis-geometry # geometry-guided mu panels
    python run.py --mode bfzi-convergence --basis-radial   # fixed-geometry radial panels and two K forms
    python run.py --mode bfzi-convergence --basis-kparts   # sphere versus near-wall weak-form K gap
    python run.py --mode bfzi-convergence --basis-wallpanels # composite physical face Gauss
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
            and (
                (mode.gaussian_metric is None
                 and candidate.gaussian_metric is None)
                or (mode.gaussian_metric is not None
                    and candidate.gaussian_metric is not None
                    and np.array_equal(
                        mode.gaussian_metric,candidate.gaussian_metric,
                    ))
            )
        ]
        if len(matches) != 1:
            raise ValueError("trial spaces are not a strictly nested hierarchy")
        indices.append(matches[0])
    return np.asarray(indices, dtype=int)


def _relative_dissipation_norm(resistance, current, reference):
    """Full-volume weighted current error in a common Galerkin basis.

    R_ij = integral_sea (J_i dot J_j)/sigma dV; unlike a handful of
    point probes, sqrt(delta_u^H R delta_u / u_ref^H R u_ref)
    compares the whole sampled seawater domain. Complex arrays may
    contain either one driven field or both independent port fields.
    """
    delta=current-reference
    numerator=float(np.real(np.vdot(delta,resistance @ delta)))
    denominator=float(np.real(np.vdot(
        reference,resistance @ reference,
    )))
    return float(np.sqrt(max(numerator,0.0)/max(denominator,1e-60)))


def _parent_galerkin_energy_bound(resistance, system, forcing, coefficients):
    """A posteriori upper bound on relative R-energy current error.

    For the fixed finite Galerkin parent, A=R+i*omega*L with R positive
    definite and L real symmetric: Re(u^H A u)=u^H R u.
    Thus ||u_exact-u_trial||_R <= ||A*u_trial+forcing||_(R^-1).
    Normalize by the unknown exact norm using the reverse triangle
    inequality: ||u_exact||_R >= ||u_trial||_R - residual_bound.
    An infinite result means this estimate is inconclusive, not that
    the actual parent-space error is infinite.
    """
    residual=system @ coefficients+forcing
    residual_energy=max(float(np.real(np.vdot(
        residual,np.linalg.solve(resistance,residual),
    ))),0.0)
    trial_energy=max(float(np.real(np.vdot(
        coefficients,resistance @ coefficients,
    ))),0.0)
    absolute_bound=np.sqrt(residual_energy)
    trial_norm=np.sqrt(trial_energy)
    if trial_norm <= absolute_bound:
        return float("inf")
    return float(absolute_bound/(trial_norm-absolute_bound))


def basis_convergence(*, verify: bool = False, green: bool = False,
                      domain: bool = False, local: bool = False,
                      angular: bool = False, geometry: bool = False,
                      radial: bool = False, kparts: bool = False,
                      wallpanels: bool = False,
                      green_refined: bool = False,
                      green_cross: bool = False,
                      green_gaussian: bool = False,
                      green_ball: bool = False,
                      ball_reference: bool = False,
                      ball_l_check: bool = False,
                      rk_normal_check: bool = False,
                      mode_ablation: bool = False,
                      source_face_guided: bool = False,
                      edge_refinement: bool = False,
                      anisotropic_refinement: bool = False):
    """Nested Galerkin spaces and spatial holdouts on independent R/K and L.

    The experimental uniform-ball modal route includes nested 159-,
    167-, 199- and optional 207-mode physical parents, and residual-driven
    sparse-mode enrichment inside the selected parent.
    Only that parent's expensive nonlocal Green L is assembled; nested
    subspaces re-solve projected R/L/K, not separate Green double integrals.
    """
    if edge_refinement and not source_face_guided:
        raise ValueError("edge refinement requires source-face guided parent")
    if anisotropic_refinement and not edge_refinement:
        raise ValueError("anisotropic refinement requires edge-refined parent")
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
        inductance_kernel="uniform_ball" if ball_reference else "plummer",
        inductance_quadrature=(5, 8, 12, 3, 4),
        max_radial_panel_width=0.18 if domain else None,
    )
    source = build_bfzi_source(scene, 200_000., order=4)
    spatial_holdout = ball_reference and not domain and not (
        radial or green_refined or green_cross or green_gaussian or green_ball
    )
    enriched = build_sea_basis(
        scene, modes_per_face=2,
        face_refinement=(5 if edge_refinement else 4) if spatial_holdout else 2,
        refine_thin_faces_only=True, sources=source,
        include_source_guided=True,
        include_source_face_guided=source_face_guided,
        include_anisotropic_face_guided=anisotropic_refinement,
    )
    print(f"Building enriched {enriched.size}-mode Galerkin operator...",
          f"raw modes={enriched.size}, "
          f"kernel={config.inductance_kernel}; "
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
    if ball_reference and domain:
        # Rebuild the *complete* physical MQS reference for each radius.
        # The magnetic quadrature is independent and uses uniform-ball L;
        # critically, R/K is independently re-integrated on the accepted
        # high-resolution wall/radial rule for EVERY domain, so apparent
        # convergence cannot be due to the old coarse dissipative rule.
        print("\nOuter-domain sequence: full ball L plus refined R/K",
              flush=True)
        previous = None
        for radius in (0.52, 0.70, 0.85, 1.05, 1.30):
            current_scene = (
                scene if radius == scene.outer_radius
                else replace(scene, outer_radius=radius)
            )
            current_source = (
                source if current_scene is scene
                else build_bfzi_source(
                    current_scene, 200_000., order=4,
                )
            )
            current_basis = (
                enriched if current_scene is scene
                else build_sea_basis(
                    current_scene, modes_per_face=2,
                    face_refinement=2, refine_thin_faces_only=True,
                    sources=current_source, include_source_guided=True,
                )
            )
            # Parent L is assembled separately for each outer domain
            # using the same physical angular/near-wall magnetic rule.
            parent = (
                teacher if current_scene is scene
                else solve_seawater_reference(
                    current_source, basis=current_basis, config=config,
                )
            )
            r_cfg = replace(
                config, polar_geometry_order=8,
                azimuthal_panel_order=2,
                radial_geometry_cuts=True, max_radial_panel_width=0.06,
                shell_normal_order=5, shell_tangential_order=6,
                shell_tangential_max_panel_width=0.0167,
            )
            rx, rw = current_scene.box_shell_quadrature(
                r_cfg.radial_order, r_cfg.polar_order, r_cfg.azimuthal_order,
                wall_scale=current_basis.wall_scale,
                shell_width_factor=r_cfg.shell_width_factor,
                shell_normal_order=r_cfg.shell_normal_order,
                shell_tangential_order=r_cfg.shell_tangential_order,
                shell_tangential_max_panel_width=r_cfg.shell_tangential_max_panel_width,
                outer_taper_width=current_basis.outer_taper_width,
                max_radial_panel_width=r_cfg.max_radial_panel_width,
                radial_geometry_cuts=r_cfg.radial_geometry_cuts,
                polar_geometry_order=r_cfg.polar_geometry_order,
                azimuthal_panel_order=r_cfg.azimuthal_panel_order,
            )
            r, k_curl, k_dual = parent.reintegrate_local_operators(
                rx, rw,
            )
            n_rk = len(rx)
            del rx, rw
            query_modes = np.einsum(
                "nmc,mr->nrc", current_basis.evaluate(probes),
                parent.transform, optimize=True,
            )

            def response_for(k):
                coeff = np.linalg.solve(
                    r + 1j*current_source.omega*parent.inductance,
                    -1j*current_source.omega*k,
                )
                z = 1j*current_source.omega*k.T @ coeff
                z = 0.5*(z + z.T)
                v = coeff @ currents
                j = np.einsum(
                    "nmc,m->nc", query_modes, v, optimize=True,
                )
                q = np.sum(np.abs(j)**2, axis=1) / (
                    2*current_scene.seawater.conductivity
                )
                power = 0.5*float(np.real(np.vdot(v, r @ v)))
                return z, q, power

            z, q, power = response_for(k_curl)
            zd, qd, pd = response_for(k_dual)

            def relative(a, b):
                return float(np.linalg.norm(a-b) /
                             max(np.linalg.norm(b), 1e-30))

            print(
                f"  radius={radius:.3g} m, fine_RK_nodes={n_rk}, "
                f"Green_L=uniform_ball, modes={len(r)}, "
                f"min_L_eig={np.linalg.eigvalsh(parent.inductance)[0]:.6g} H, "
                f"sea_power={power:.8g} W",
                flush=True,
            )
            print(
                f"  weak_form_gap: dK={relative(k_dual,k_curl):.6g}, "
                f"dZ={relative(zd,z):.6g}, "
                f"dq={relative(qd,q):.6g}, "
                f"dP={abs(pd-power)/max(abs(power),1e-30):.6g}",
                flush=True,
            )
            if previous is not None:
                old_radius, old_z, old_q, old_p = previous
                print(
                    f"  domain_step: {old_radius:.3g}->{radius:.3g} m, "
                    f"dZ={relative(z,old_z):.6g}, "
                    f"dq={relative(q,old_q):.6g}, "
                    f"dP={abs(power-old_p)/max(abs(old_p),1e-30):.6g}",
                    flush=True,
                )
            previous = radius, z, q, power
        print(
            "Outer-domain screening uses independently refined R/K "
            "at every radius and physical ball L. Port/probe/power "
            "stability still needs a validated radius plateau.",
            flush=True,
        )
        return

    if spatial_holdout:
        # Extend the four established bulk probes with physical near-wall
        # locations on both broad faces of each oriented enclosure. This
        # catches sharp local Joule features masked by a small port dZ.
        near_wall = []
        for assembly in scene.assemblies:
            half = np.asarray(assembly.enclosure.half_extents)
            axis = int(np.argmin(half))
            tangents = [direction for direction in range(3) if direction != axis]
            for sign in (-1.0, 1.0):
                for u, v in ((0.0, 0.0), (0.65, 0.0), (0.0, 0.65)):
                    local = np.zeros(3)
                    local[axis] = sign*(
                        half[axis] + max(0.003, 3.0*enriched.wall_scale)
                    )
                    local[tangents] = np.asarray((u, v))*half[tangents]
                    near_wall.append(assembly.pose.position(local))
        probes = np.concatenate((probes, np.asarray(near_wall)))
        if not scene.in_seawater(probes).all():
            raise RuntimeError("modal wall probes must lie in seawater")

        # An independent face-normal sweep catches localization shifts
        # which the original 16 fixed probes can miss. It is evaluated
        # only after the common parent operator has been assembled;
        # no extra Green integral or solver mesh is created.
        normal_line = []
        for assembly in scene.assemblies:
            half=np.asarray(assembly.enclosure.half_extents)
            axis=int(np.argmin(half))
            tangents=[j for j in range(3) if j!=axis]
            for sign in (-1.0,1.0):
                for u,v in ((0.0,0.0),(0.65,0.0)):
                    for step in (0.0,0.010,0.025,0.050,0.075):
                        local=np.zeros(3)
                        local[axis]=sign*(
                            half[axis]
                            +max(0.003,3.0*enriched.wall_scale)+step
                        )
                        local[tangents]=np.array((u,v))*half[tangents]
                        normal_line.append(assembly.pose.position(local))
        normal_line=np.asarray(normal_line)
        if not scene.in_seawater(normal_line).all():
            raise RuntimeError("normal sweep probes must lie in seawater")
        # Genuine nested-mode comparison with the already established
        # independently refined near-wall R/K, NOT the preliminary coarse
        # teacher R/K. The uniform-ball L was assembled on magnetic A
        # by the teacher and is held fixed for every nested Galerkin solve.
        print(f"\nRefined R/K + uniform-ball L: independent 63-to-95-to-{enriched.size} modal holdout",
              flush=True)
        r_cfg = replace(
            config, polar_geometry_order=8, azimuthal_panel_order=2,
            radial_geometry_cuts=True, max_radial_panel_width=0.06,
            shell_normal_order=5, shell_tangential_order=6,
            shell_tangential_max_panel_width=0.0167,
        )
        rx, rw = scene.box_shell_quadrature(
            r_cfg.radial_order, r_cfg.polar_order, r_cfg.azimuthal_order,
            wall_scale=enriched.wall_scale,
            shell_width_factor=r_cfg.shell_width_factor,
            shell_normal_order=r_cfg.shell_normal_order,
            shell_tangential_order=r_cfg.shell_tangential_order,
            shell_tangential_max_panel_width=r_cfg.shell_tangential_max_panel_width,
            outer_taper_width=enriched.outer_taper_width,
            max_radial_panel_width=r_cfg.max_radial_panel_width,
            radial_geometry_cuts=r_cfg.radial_geometry_cuts,
            polar_geometry_order=r_cfg.polar_geometry_order,
            azimuthal_panel_order=r_cfg.azimuthal_panel_order,
        )
        r, k_curl, k_dual = teacher.reintegrate_local_operators(rx, rw)
        print(
            f"  refined R/K nodes={len(rx)}; L=A uniform_ball; "
            "no 461648-node heat tensor retained",
            flush=True,
        )
        del rx, rw
        probe_modes = np.einsum(
            "nmc,mr->nrc", enriched.evaluate(probes),
            teacher.transform, optimize=True,
        )
        normal_modes = np.einsum(
            "nmc,mr->nrc", enriched.evaluate(normal_line),
            teacher.transform, optimize=True,
        )
        omega = source.omega
        m = len(r)
        full_system = r + 1j * omega * teacher.inductance
        full_forcing = 1j * omega * k_curl
        forcing_norm = np.sqrt(max(float(np.real(np.vdot(
            full_forcing, np.linalg.solve(r, full_forcing),
        ))), 1e-60))

        def relative(a, b):
            return float(np.linalg.norm(a-b) /
                         max(np.linalg.norm(b), 1e-30))

        def solve_nested(coords, coupling):
            reduced_r = coords.T @ r @ coords
            reduced_l = coords.T @ teacher.inductance @ coords
            reduced_k = coords.T @ coupling
            amplitudes = np.linalg.solve(
                reduced_r + 1j * omega * reduced_l,
                -1j * omega * reduced_k,
            )
            z = 1j * omega * reduced_k.T @ amplitudes
            z = 0.5 * (z + z.T)
            driven = amplitudes @ currents
            j_ports = np.einsum(
                "nmc,mp->ncp",probe_modes,coords @ amplitudes,
                optimize=True,
            )
            j = np.einsum("ncp,p->nc",j_ports,currents,optimize=True)
            q = np.sum(np.abs(j)**2, axis=1) / (
                2 * scene.seawater.conductivity
            )
            p = 0.5 * float(np.real(
                np.vdot(driven, reduced_r @ driven)
            ))
            residual = full_system @ (coords @ amplitudes) + full_forcing
            relative_residual = np.sqrt(max(float(np.real(np.vdot(
                residual, np.linalg.solve(r, residual),
            ))), 0.0)) / forcing_norm
            normal_j=np.einsum(
                "nmc,m->nc",normal_modes,coords @ driven,optimize=True,
            )
            normal_q=np.sum(np.abs(normal_j)**2,axis=1)/(
                2*scene.seawater.conductivity
            )
            return (z,q,p,relative_residual,j,j_ports,
                    coords @ amplitudes,normal_q,
                    _parent_galerkin_energy_bound(
                        r,full_system,full_forcing,coords @ amplitudes,
                    ))

        previous = None
        levels = [
            ("one_tangent", 1, 1, False, False, False),
            ("two_tangents", 2, 1, False, False, False),
            ("thin_face_patches", 2, 2, True, False, False),
            ("guided_port_curls", 2, 2, True, True, False),
            ("edge_face_patches", 2, 3, True, True, False),
            ("corner_face_patches", 2, 4, True, True, False),
        ]
        if source_face_guided:
            levels.append(("source_face_guided", 2, 4, True, True, True))
        if edge_refinement:
            levels.append(("edge_lip_patches", 2, 5, True, True, True))
        if anisotropic_refinement:
            levels.append(("anisotropic_face_curls", 2, 5, True, True, True))
        for label, modes_per_face, face_refinement, thin, guided, face_guided in levels:
            subset = build_sea_basis(
                scene, modes_per_face=modes_per_face,
                face_refinement=face_refinement,
                refine_thin_faces_only=thin,
                sources=source if guided else None,
                include_source_guided=guided,
                include_source_face_guided=face_guided,
                include_anisotropic_face_guided=(
                    label=="anisotropic_face_curls"
                ),
            )
            indices = _indices_in_enriched_basis(enriched, subset)
            selection = np.eye(enriched.size)[:, indices]
            coords, _, _, _ = np.linalg.lstsq(
                teacher.transform, selection, rcond=None,
            )
            if not np.allclose(
                teacher.transform @ coords, selection,
                rtol=1e-8, atol=1e-9,
            ):
                raise RuntimeError(
                    "nested current modes were discarded by R whitening"
                )
            (z,q,p,residual,j,j_ports,unit_modes,
             normal_q,parent_energy_bound) = solve_nested(coords,k_curl)
            print(
                f"  {label}: modes={coords.shape[1]}, "
                f"relative_Rinv_residual={residual:.6g}, "
                f"parent_dJ_energy_upper={parent_energy_bound:.6g}, "
                f"sea_power={p:.8g} W, Zsw={z.tolist()}",
                flush=True,
            )
            print(f"  {label}_probes: q={q.tolist()}", flush=True)
            if previous is not None:
                (old_label, old_z, old_q, old_p, old_j,
                 old_j_ports,old_unit_modes,old_normal_q) = previous
                print(
                    f"  {old_label}_to_{label}: "
                    f"dZ={relative(z,old_z):.6g}, "
                    f"dJ={relative(j,old_j):.6g}, "
                    f"dJ_ports={relative(j_ports,old_j_ports):.6g}, "
                    f"dJ_volume={_relative_dissipation_norm(r,unit_modes @ currents,old_unit_modes @ currents):.6g}, "
                    f"dJ_ports_volume={_relative_dissipation_norm(r,unit_modes,old_unit_modes):.6g}, "
                    f"dJ_port0={relative(j_ports[:,:,0],old_j_ports[:,:,0]):.6g}, "
                    f"dJ_port1={relative(j_ports[:,:,1],old_j_ports[:,:,1]):.6g}, "
                    f"dq={relative(q,old_q):.6g}, "
                    f"max_probe_dq={np.max(np.abs(q-old_q)/np.maximum(np.abs(old_q),1e-10)):.6g}, "
                    f"dP={abs(p-old_p)/max(abs(old_p),1e-30):.6g}",
                    flush=True,
                )
            if label in (
                "source_face_guided","edge_lip_patches",
                "anisotropic_face_curls",
            ):
                # Distinguish a large relative change in a cool location
                # from a comparable change in a heating hotspot.
                old_q = previous[2] if previous is not None else q
                delta = np.abs(q-old_q)
                relative_probe = delta/np.maximum(old_q,1e-10)
                peak = max(float(np.max(old_q)),1e-30)
                worst = int(np.argmax(relative_probe))
                hot = old_q>=0.1*peak
                normal_delta=np.abs(normal_q-old_normal_q)
                normal_peak=max(float(np.max(old_normal_q)),1e-30)
                normal_worst=int(np.argmax(normal_delta))
                print(
                    f"  {label}_normal_sweep: "
                    f"dq={relative(normal_q,old_normal_q):.6g}, "
                    f"peak_scaled_max_dq={np.max(normal_delta)/normal_peak:.6g}, "
                    f"max_abs_point={normal_line[normal_worst].tolist()}, "
                    f"reference_q={old_normal_q[normal_worst]:.8g}, "
                    f"enriched_q={normal_q[normal_worst]:.8g}",
                    flush=True,
                )
                print(
                    f"  {label}_heat_detail: "
                    f"worst_probe={worst}, "
                    f"reference_q={old_q[worst]:.8g}, "
                    f"enriched_q={q[worst]:.8g}, "
                    f"worst_point={probes[worst].tolist()}, "
                    f"hotspot_max_dq={np.max(relative_probe[hot]):.6g}, "
                    f"peak_scaled_max_dq={np.max(delta)/peak:.6g}",
                    flush=True,
                )
            previous = label, z, q, p, j, j_ports, unit_modes, normal_q

        # Residual-driven sparse trial selection from the
        # strictly nested physical current family. This is an
        # offline diagnostic, not an online approximation: the full parent
        # R/L/K must first be assembled, so no continuum accuracy is claimed.
        # It records how many additional
        # modes are actually needed for Z, Joule probes and port power.
        print("\nResidual-driven selective enrichment from 63 modes:",
              flush=True)
        raw_directions, _, _, _ = np.linalg.lstsq(
            teacher.transform, np.eye(enriched.size), rcond=None,
        )
        if not np.allclose(
            teacher.transform @ raw_directions, np.eye(enriched.size),
            atol=1e-8, rtol=1e-8,
        ):
            raise RuntimeError(
                "enriched hierarchy contains discarded physical modes; "
                "need better-resolved primary whitening before greedy selection"
            )
        raw_r = raw_directions.T @ r @ raw_directions
        raw_l = raw_directions.T @ teacher.inductance @ raw_directions
        raw_k = raw_directions.T @ k_curl
        raw_a = raw_r + 1j*omega*raw_l
        raw_f = 1j*omega*raw_k
        raw_probes = np.einsum(
            "nmc,mr->nrc", probe_modes, raw_directions, optimize=True,
        )
        def report_sparse(selected):
            chosen = np.asarray(selected, dtype=int)
            rr = raw_r[np.ix_(chosen,chosen)]
            kk = raw_k[chosen]
            coeff = np.linalg.solve(raw_a[np.ix_(chosen,chosen)],
                                    -raw_f[chosen])
            impedance = 1j*omega*kk.T @ coeff
            impedance = 0.5*(impedance + impedance.T)
            driven = coeff @ currents
            j = np.einsum(
                "nmc,m->nc", raw_probes[:,chosen,:], driven, optimize=True,
            )
            heat = np.sum(np.abs(j)**2, axis=1) / (
                2*scene.seawater.conductivity
            )
            power = 0.5*float(np.real(np.vdot(driven,rr @ driven)))
            return impedance, heat, power, coeff
        full_z, full_q, full_power, full_coeff = report_sparse(
            np.arange(enriched.size)
        )

        # A two-port phasor solution has at most four REAL response
        # directions regardless of the size of the physical trial space.
        # The parent MUST be converged separately; this only proves that
        # the already computed frequency/geometry response is compressible.
        from sdfmpneo_bfzi.sea_snapshot import BFZISnapshotROM
        print("\nStructure-preserving fixed-scene response compression:",
              flush=True)
        for rank in (2, 4):
            snapshot = BFZISnapshotROM.from_operators(
                source, enriched, teacher.transform,
                r, teacher.inductance, k_curl,
                rank=rank, inductance_kernel="uniform_ball",
            )
            reduced_z = snapshot.impedance_seawater
            reduced_q = snapshot.seawater_loss(probes, currents)
            reduced_power = snapshot.port_seawater_power(currents)
            modal_power = snapshot.modal_seawater_power(currents)
            print(
                f"  snapshot_rank_{rank}: "
                f"dZ={relative(reduced_z,full_z):.6g}, "
                f"dq={relative(reduced_q,full_q):.6g}, "
                f"max_probe_dq={np.max(np.abs(reduced_q-full_q)/np.maximum(full_q,1e-10)):.6g}, "
                f"dP={abs(reduced_power-full_power)/max(abs(full_power),1e-30):.6g}, "
                f"power_closure={abs(reduced_power-modal_power)/max(abs(modal_power),1e-30):.6g}",
                flush=True,
            )
        print(
            "  snapshot_note: rank-4 exactness is ONLY relative to "
            f"the unconverged {enriched.size}-mode parent at this scene/frequency. "
            "It does not validate a cross-geometry model or local heat truth.",
            flush=True,
        )

        if ball_l_check:
            # Smaller-space magnetic A/B agreement cannot be promoted to
            # a larger physical trial space. This independently
            # resolves magnetic integration without changing the 461k-node
            # fine resistance, both port couplings, or physical probes.
            print(f"\nIndependent {enriched.size}-mode uniform-ball magnetic L B rule:",
                  flush=True)
            bx,bw = scene.box_shell_quadrature(
                5,9,14,wall_scale=enriched.wall_scale,
                shell_width_factor=config.shell_width_factor,
                shell_normal_order=4,shell_tangential_order=5,
                outer_taper_width=enriched.outer_taper_width,
            )
            print(f"  assembling B nodes={len(bx)}",flush=True)
            magnetic_b, _ = teacher.reintegrate_ball_green_matrix(
                bx,bw,kernel_chunk_size=config.kernel_chunk_size,
            )
            del bx,bw
            def full_with_l(l_matrix, k):
                coefficient = np.linalg.solve(
                    r + 1j*omega*l_matrix,-1j*omega*k,
                )
                z = 1j*omega*k.T @ coefficient
                z = 0.5*(z+z.T)
                field_ports = np.einsum(
                    "nmc,mp->ncp",probe_modes,coefficient,
                    optimize=True,
                )
                field = np.einsum(
                    "ncp,p->nc",field_ports,currents,optimize=True,
                )
                q = np.sum(np.abs(field)**2,axis=1)/(
                    2*scene.seawater.conductivity
                )
                power = 0.5*float(np.real(np.vdot(
                    coefficient @ currents,r @ (coefficient @ currents)
                )))
                return z,q,power,field,coefficient,field_ports

            # Only the magnetic cubature changes: same refined R,
            # curl-current K, whitened parent space and physical probes.
            z_b,q_b,p_b,j_b,u_b,j_b_ports = full_with_l(magnetic_b,k_curl)
            z_a,q_a,p_a,j_a,u_a,j_a_ports = full_with_l(
                teacher.inductance,k_curl,
            )
            dL = relative(magnetic_b,teacher.inductance)
            dZ = relative(z_b,z_a)
            # The seawater impedance is mostly resistive here. A small
            # full-complex dZ can obscure a much larger relative error
            # in its physically important reactive component.
            dReZ = relative(z_b.real,z_a.real)
            dImZ = relative(z_b.imag,z_a.imag)
            dJ = relative(j_b,j_a)
            dJ_ports=relative(j_b_ports,j_a_ports)
            dJ_volume=_relative_dissipation_norm(
                r,u_b @ currents,u_a @ currents,
            )
            dJ_ports_volume=_relative_dissipation_norm(r,u_b,u_a)
            dq = relative(q_b,q_a)
            dP = abs(p_b-p_a)/max(abs(p_a),1e-30)
            # Test the actual A solution in the *independently assembled*
            # B variational operator. A small response delta alone is not
            # enough if the two magnetic rules share a discretization bias.
            b_residual=(
                (r+1j*omega*magnetic_b) @ u_a + 1j*omega*k_curl
            )
            b_relative_residual=np.sqrt(max(float(np.real(np.vdot(
                b_residual,np.linalg.solve(r,b_residual),
            ))),0.0))/forcing_norm
            operator_pass=(dL<=0.05)
            response_pass=(
                dZ<=0.002 and dJ<=0.01 and dJ_ports<=0.01
                and dJ_volume<=0.01 and dJ_ports_volume<=0.01
                and dq<=0.005 and dP<=0.002
                and b_relative_residual<=0.01
                and np.max(np.abs(q_b-q_a)/np.maximum(q_a,1e-10))<=0.05
            )
            passed=operator_pass and response_pass
            print(
                f"  ball_{enriched.size}_B_vs_A: dL={dL:.6g}, "
                f"dZ={dZ:.6g}, dReZ={dReZ:.6g}, "
                f"dImZ={dImZ:.6g}, "
                f"dJ={dJ:.6g}, dJ_ports={dJ_ports:.6g}, "
                f"dJ_volume={dJ_volume:.6g}, "
                f"dJ_ports_volume={dJ_ports_volume:.6g}, "
                f"dJ_port0={relative(j_b_ports[:,:,0],j_a_ports[:,:,0]):.6g}, "
                f"dJ_port1={relative(j_b_ports[:,:,1],j_a_ports[:,:,1]):.6g}, "
                f"dq={dq:.6g}, "
                f"max_probe_dq={np.max(np.abs(q_b-q_a)/np.maximum(q_a,1e-10)):.6g}, "
                f"dP={dP:.6g}, "
                f"B_on_A_Rinv_residual={b_relative_residual:.6g}, "
                f"min_L_B_eig={np.linalg.eigvalsh(magnetic_b)[0]:.6g}, "
                f"operator_gate={'PASS' if operator_pass else 'FAIL'}, "
                f"response_gate={'PASS' if response_pass else 'FAIL'}, "
                f"combined_gate={'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            za_dual,qa_dual,pa_dual,ja_dual,_,_ = full_with_l(
                teacher.inductance,k_dual,
            )
            zb_dual,qb_dual,pb_dual,jb_dual,_,_ = full_with_l(
                magnetic_b,k_dual,
            )
            print(
                f"  ball_{enriched.size}_dual_B_vs_A: "
                f"dZ={relative(zb_dual,za_dual):.6g}, "
                f"dJ={relative(jb_dual,ja_dual):.6g}, "
                f"dq={relative(qb_dual,qa_dual):.6g}, "
                f"dP={abs(pb_dual-pa_dual)/max(abs(pa_dual),1e-30):.6g}",
                flush=True,
            )
            if not passed:
                print(
                    "  magnetic_gate_failed: "
                    f"operator={'FAIL' if not operator_pass else 'PASS'}, "
                    f"response={'FAIL' if not response_pass else 'PASS'}; "
                    "response agreement does not establish operator "
                    "convergence, and neither certifies continuum accuracy.",
                    flush=True,
                )
            print("  magnetic_note: a resistive-dominated full dZ can "
                  "hide reactive impedance error; inspect dImZ separately. "
                  "Also compare successive modal changes before declaring "
                  "a spatial teacher.",flush=True)

        if rk_normal_check:
            # Independently resolve the face-normal one-volume R/K rule
            # without recomputing the expensive Green L, changing basis,
            # or retaining a giant heat tensor. Hold all tangential,
            # angular, radial and source parameters fixed.
            b_cfg=replace(r_cfg,shell_normal_order=r_cfg.shell_normal_order+1)
            nx,nw=scene.box_shell_quadrature(
                b_cfg.radial_order,b_cfg.polar_order,b_cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=b_cfg.shell_width_factor,
                shell_normal_order=b_cfg.shell_normal_order,
                shell_tangential_order=b_cfg.shell_tangential_order,
                shell_tangential_max_panel_width=b_cfg.shell_tangential_max_panel_width,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=b_cfg.max_radial_panel_width,
                radial_geometry_cuts=b_cfg.radial_geometry_cuts,
                polar_geometry_order=b_cfg.polar_geometry_order,
                azimuthal_panel_order=b_cfg.azimuthal_panel_order,
            )
            print(
                f"\nIndependent {enriched.size}-mode R/K face-normal cubature: "
                f"A_normal={r_cfg.shell_normal_order}, "
                f"B_normal={b_cfg.shell_normal_order}, "
                f"B_nodes={len(nx)}",flush=True,
            )
            r_b,k_b,kdual_b=teacher.reintegrate_local_operators(nx,nw)
            del nx,nw
            def rk_response(local_r,local_k):
                coeff=np.linalg.solve(
                    local_r+1j*omega*teacher.inductance,
                    -1j*omega*local_k,
                )
                z=1j*omega*local_k.T @ coeff
                z=0.5*(z+z.T)
                field=np.einsum(
                    "nmc,m->nc",probe_modes,coeff @ currents,
                    optimize=True,
                )
                heat=np.sum(np.abs(field)**2,axis=1)/(
                    2*scene.seawater.conductivity
                )
                power=0.5*float(np.real(np.vdot(
                    coeff @ currents,local_r @ (coeff @ currents),
                )))
                return coeff,z,field,heat,power

            u_ka,z_ka,j_ka,q_ka,p_ka=rk_response(r,k_curl)
            u_kb,z_kb,j_kb,q_kb,p_kb=rk_response(r_b,k_b)
            rk_field_energy=_relative_dissipation_norm(
                r,u_kb @ currents,u_ka @ currents,
            )
            rk_ports_energy=_relative_dissipation_norm(r,u_kb,u_ka)
            rk_response_pass=(
                relative(z_kb,z_ka)<=0.002
                and relative(j_kb,j_ka)<=0.01
                and rk_field_energy<=0.01 and rk_ports_energy<=0.01
                and relative(q_kb,q_ka)<=0.005
                and np.max(np.abs(q_kb-q_ka)/np.maximum(q_ka,1e-10))<=0.05
                and abs(p_kb-p_ka)/max(abs(p_ka),1e-30)<=0.002
            )
            print(
                f"  rk_normal_{enriched.size}_B_vs_A: "
                f"dR={relative(r_b,r):.6g}, "
                f"dK={relative(k_b,k_curl):.6g}, "
                f"dK_dual={relative(kdual_b,k_dual):.6g}, "
                f"dZ={relative(z_kb,z_ka):.6g}, "
                f"dReZ={relative(z_kb.real,z_ka.real):.6g}, "
                f"dImZ={relative(z_kb.imag,z_ka.imag):.6g}, "
                f"dJ={relative(j_kb,j_ka):.6g}, "
                f"dJ_volume={rk_field_energy:.6g}, "
                f"dJ_ports_volume={rk_ports_energy:.6g}, "
                f"dq={relative(q_kb,q_ka):.6g}, "
                f"max_probe_dq={np.max(np.abs(q_kb-q_ka)/np.maximum(q_ka,1e-10)):.6g}, "
                f"dP={abs(p_kb-p_ka)/max(abs(p_ka),1e-30):.6g}, "
                f"response_gate={'PASS' if rk_response_pass else 'FAIL'}",
                flush=True,
            )
            print(
                "  rk_normal_note: B changes only face-normal integration "
                "order. The reported full-volume J metric uses common "
                "A-rule resistance; passing this comparison alone does "
                "not certify tangential, far-field or Green convergence.",
                flush=True,
            )

        initial_basis = build_sea_basis(
            scene, modes_per_face=2, face_refinement=2,
            refine_thin_faces_only=True, sources=source,
            include_source_guided=True,
        )
        selected = list(_indices_in_enriched_basis(enriched, initial_basis))
        diag_r = np.maximum(np.diag(raw_r), 1e-30)
        iteration = 0
        while True:
            # Do not silently stop at the historical 159-mode iteration
            # count: an enriched 167-mode parent needs one further batch.
            # The loop is finite because each step adds independent modes
            # and stops at either a match or exhaustion of the parent.
            z, q, power, coeff = report_sparse(selected)
            dz = relative(z, full_z)
            dq = relative(q, full_q)
            dp = abs(power-full_power)/max(abs(full_power),1e-30)
            sparse_coeff=np.zeros_like(full_coeff)
            sparse_coeff[selected]=coeff
            dj_volume=_relative_dissipation_norm(
                raw_r,sparse_coeff @ currents,full_coeff @ currents,
            )
            dj_ports_volume=_relative_dissipation_norm(
                raw_r,sparse_coeff,full_coeff,
            )
            heat_change = np.abs(q-full_q)
            per_probe_rel = heat_change/np.maximum(full_q,1e-10)
            worst_probe = int(np.argmax(per_probe_rel))
            max_dq = float(per_probe_rel[worst_probe])
            peak_q = max(float(np.max(full_q)),1e-30)
            max_peak_scaled_dq = float(np.max(heat_change)/peak_q)
            hotspot = full_q >= 0.1*peak_q
            hotspot_max_dq = float(np.max(per_probe_rel[hotspot]))
            hotspot_match = (
                dz<=0.002 and dq<=0.005 and dp<=0.002
                and hotspot_max_dq<=0.05
                and max_peak_scaled_dq<=0.01
            )
            # Full near-wall agreement remains a stricter, separate gate.
            # A hotspot match is a diagnostic, never a certificate for
            # cooler but potentially thermally sensitive locations.
            sparse = np.zeros_like(raw_f)
            sparse[selected] = coeff
            residual = raw_a @ sparse + raw_f
            # Use the same well-scaled parent-coordinate R for the
            # residual energy estimator, not the potentially ill-scaled
            # unwhitened raw physical Gaussian basis.
            white_coeff=raw_directions @ sparse
            energy_bound=_parent_galerkin_energy_bound(
                r,full_system,full_forcing,white_coeff,
            )
            if len(selected)==enriched.size:
                score = np.full(enriched.size,-np.inf)
            else:
                # R-orthogonalize each candidate against selected modes.
                # This suppresses near-duplicate Gaussian directions.
                cross = raw_r[:, selected]
                schur = np.diag(raw_r) - np.einsum(
                    "ij,ji->i",cross,
                    np.linalg.solve(
                        raw_r[np.ix_(selected,selected)],cross.T,
                    ),optimize=True,
                )
                independent = schur > 1e-10*diag_r
                score = np.full(enriched.size,-np.inf)
                score[independent] = (
                    np.linalg.norm(residual[independent],axis=1)
                    /np.sqrt(schur[independent])
                )
                score[selected] = -np.inf
            print(
                f"  adaptive_{iteration:02d}: modes={len(selected)}, "
                f"dZ={dz:.6g}, dq={dq:.6g}, "
                f"dJ_volume={dj_volume:.6g}, "
                f"dJ_ports_volume={dj_ports_volume:.6g}, "
                f"parent_dJ_energy_upper={energy_bound:.6g}, "
                f"max_probe_dq={max_dq:.6g}, "
                f"hotspot_max_dq={hotspot_max_dq:.6g}, "
                f"peak_scaled_max_dq={max_peak_scaled_dq:.6g}, "
                f"worst_probe={worst_probe}, "
                f"dP={dp:.6g}, "
                f"max_residual_score={np.max(score):.6g}, "
                f"hotspot_match={hotspot_match}",
                flush=True,
            )
            if (dz<=0.002 and dq<=0.005 and dp<=0.002
                    and max_dq<=0.05):
                status = (
                    "SPARSE_MATCH" if len(selected)<enriched.size
                    else "FULL_PARENT_ONLY"
                )
                print(
                    f"  adaptive_screen: {status}; "
                    f"selected={len(selected)}/{enriched.size}, "
                    "matched all fixed heat probes to <=5% plus port "
                    "and aggregate heat criteria relative to this parent; "
                    "NOT continuum-certified",
                    flush=True,
                )
                break
            if len(selected)==enriched.size:
                print("  adaptive_screen: FULL_PARENT_ONLY; "
                      "all available directions selected, "
                      "NOT continuum-certified",flush=True)
                break
            available = np.flatnonzero(np.isfinite(score))
            if not len(available):
                print("  adaptive_screen: no independent candidate "
                      "direction remains; NOT continuum-certified",
                      flush=True)
                break
            # Select the eight most violated remaining Galerkin equations,
            # then re-solve. The final batch may contain fewer than eight.
            ranked = available[np.argsort(-score[available],kind="stable")]
            selected.extend(int(i) for i in ranked[:8])
            iteration += 1

        final = solve_nested(np.eye(m), k_curl)
        dual = solve_nested(np.eye(m), k_dual)
        print(
            "  full_weak_form: "
            f"dK={relative(k_dual,k_curl):.6g}, "
            f"dZ={relative(dual[0],final[0]):.6g}, "
            f"dq={relative(dual[1],final[1]):.6g}, "
            f"dP={abs(dual[2]-final[2])/max(abs(final[2]),1e-30):.6g}",
            flush=True,
        )
        print(
            "Modal screen uses refined streamed R/K and A uniform-ball L. "
            f"It measures missing trial directions within the {enriched.size}-mode "
            "parent, not a continuum accuracy certificate.",
            flush=True,
        )
        return

    if radial:
        # Do not build the redundant nested basis and global angular
        # studies: this experiment holds those axes fixed and isolates
        # spherical radial panels, including the two weak forms of K.
        print("\nRadial geometry/cap convergence (fixed 63 modes and Green L):",
              flush=True)
        print(
            "All rows use geometry polar Gauss order 8, azimuth sectors "
            "16 with 2-point Gauss, fixed box-wall (4,5), and "
            "the same R-whitened physical Galerkin space.",
            flush=True,
        )
        experiments = (
            ("geo8_original", False, None, 6),
            ("geo8_radial_cuts", True, None, 6),
            ("radial_cap_012", True, 0.12, 6),
            ("radial_cap_006", True, 0.06, 6),
            ("radial_cap_003", True, 0.03, 6),
            ("radial_cap_006_g7", True, 0.06, 7),
        )

        def relative(a, b):
            return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30))

        previous = None
        fixed_cap006 = None
        for label, use_cuts, width, order in experiments:
            cfg = replace(
                config, radial_order=order, radial_geometry_cuts=use_cuts,
                max_radial_panel_width=width, polar_geometry_order=8,
                azimuthal_panel_order=2,
            )
            x, weights = scene.box_shell_quadrature(
                cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=cfg.shell_width_factor,
                shell_normal_order=cfg.shell_normal_order,
                shell_tangential_order=cfg.shell_tangential_order,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=cfg.max_radial_panel_width,
                radial_geometry_cuts=cfg.radial_geometry_cuts,
                polar_geometry_order=cfg.polar_geometry_order,
                azimuthal_panel_order=cfg.azimuthal_panel_order,
            )
            curl = teacher.reintegrate_local_response(x, weights)
            dual_k = teacher.reintegrate_source_coupling(
                x, weights, coupling_form="curl_dual",
            )
            dual = curl.resolve_with_coupling(dual_k)
            curl_q = curl.seawater_loss(probes, currents)
            dual_q = dual.seawater_loss(probes, currents)
            curl_p = curl.port_seawater_power(currents)
            dual_p = dual.port_seawater_power(currents)
            print(
                f"{label:>22}: nodes={len(x):6d}, "
                f"P_curl={curl_p:.8g} W, P_dual={dual_p:.8g} W",
                flush=True,
            )
            print(
                f"{'weak_form_gap':>22}: "
                f"dK={relative(dual.coupling,curl.coupling):.6g}, "
                f"dZ={relative(dual.impedance_seawater,curl.impedance_seawater):.6g}, "
                f"dq={relative(dual_q,curl_q):.6g}, "
                f"max_probe_dq={np.max(np.abs(dual_q-curl_q)/np.maximum(np.abs(curl_q),1e-30)):.6g}, "
                f"dP={abs(dual_p-curl_p)/max(abs(curl_p),1e-30):.6g}",
                flush=True,
            )
            if previous is not None:
                # The final Gauss-degree cross-check must compare with
                # the SAME 0.06 m radial panels, not preceding 0.03 m.
                reference = (
                    fixed_cap006 if label == "radial_cap_006_g7"
                    else previous
                )
                old_label, old_curl, old_dual = reference
                for form, response, old in (
                    ("curl_K", curl, old_curl),
                    ("dual_K", dual, old_dual),
                ):
                    q = response.seawater_loss(probes, currents)
                    p = response.port_seawater_power(currents)
                    old_q = old.seawater_loss(probes, currents)
                    old_p = old.port_seawater_power(currents)
                    print(
                        f"{form:>22}: from={old_label}, "
                        f"dR={relative(response.resistance,old.resistance):.6g}, "
                        f"dK={relative(response.coupling,old.coupling):.6g}, "
                        f"dZ={relative(response.impedance_seawater,old.impedance_seawater):.6g}, "
                        f"dq={relative(q,old_q):.6g}, "
                        f"max_probe_dq={np.max(np.abs(q-old_q)/np.maximum(np.abs(old_q),1e-30)):.6g}, "
                        f"dP={abs(p-old_p)/max(abs(old_p),1e-30):.6g}",
                        flush=True,
                    )
            if label == "radial_cap_006":
                fixed_cap006 = label, curl, dual
            previous = label, curl, dual
        print(
            "The cap 0.03 vs 0.06 comparison refines physical "
            "radial panels. The g7 row changes only the Gauss degree "
            "at cap 0.06. Green L and the 63-mode trial space stay fixed.",
            flush=True,
        )
        return

    if green_refined or green_cross or green_gaussian or green_ball:
        # This comparison changes ONLY Green L after a single high-quality
        # streamed R/K solve. It avoids the original teacher's inaccurate
        # global-angle/near-wall R/K while retaining that teacher's modes,
        # whitening, and separate Green rule.
        print("\nIndependent Green L / softening with refined 63-mode R/K:",
              flush=True)
        r_cfg = replace(
            config, polar_geometry_order=8, azimuthal_panel_order=2,
            radial_geometry_cuts=True, max_radial_panel_width=0.06,
            shell_normal_order=5, shell_tangential_order=6,
            shell_tangential_max_panel_width=0.0167,
        )
        x, w = scene.box_shell_quadrature(
            r_cfg.radial_order, r_cfg.polar_order, r_cfg.azimuthal_order,
            wall_scale=enriched.wall_scale,
            shell_width_factor=r_cfg.shell_width_factor,
            shell_normal_order=r_cfg.shell_normal_order,
            shell_tangential_order=r_cfg.shell_tangential_order,
            shell_tangential_max_panel_width=r_cfg.shell_tangential_max_panel_width,
            outer_taper_width=enriched.outer_taper_width,
            max_radial_panel_width=r_cfg.max_radial_panel_width,
            radial_geometry_cuts=r_cfg.radial_geometry_cuts,
            polar_geometry_order=r_cfg.polar_geometry_order,
            azimuthal_panel_order=r_cfg.azimuthal_panel_order,
        )
        resistance, k_current, k_dual = (
            teacher.reintegrate_local_operators(x, w)
        )
        print(f"  refined R/K nodes={len(x)}; operator-only, no retained "
              "volume Joule tensor", flush=True)
        del x, w
        probe_modes = np.einsum(
            "nmc,mr->nrc", enriched.evaluate(probes),
            teacher.transform, optimize=True,
        )

        def metrics(l, k):
            coefficients = np.linalg.solve(
                resistance + 1j * source.omega * l,
                -1j * source.omega * k,
            )
            z = 1j * source.omega * k.T @ coefficients
            z = 0.5 * (z + z.T)
            amplitude = coefficients @ currents
            j = np.einsum(
                "nmc,m->nc", probe_modes, amplitude, optimize=True,
            )
            q = np.sum(np.abs(j)**2, axis=1) / (
                2 * scene.seawater.conductivity
            )
            p = 0.5 * float(
                np.real(np.vdot(amplitude, resistance @ amplitude))
            )
            return (z, q, p)

        def relative(a, b):
            return float(np.linalg.norm(a - b)
                         / max(np.linalg.norm(b), 1e-30))

        base_l = teacher.inductance
        original_base = (
            metrics(base_l, k_current), metrics(base_l, k_dual),
        )
        print(
            f"  initial independent L from teacher: "
            f"trace={np.trace(base_l):.8g} H; "
            f"P_curl={original_base[0][2]:.8g} W, "
            f"P_dual={original_base[1][2]:.8g} W",
            flush=True,
        )
        if green_cross:
            # A and B cover the same box-shell partition of unity, using
            # independently shifted Gauss-degree/sector nodes. Off-diagonal
            # cross-pairs no longer contain artificial i=j self terms.
            # This is an error-diagnostic estimator, NOT a PSD teacher.
            print("\nCross versus same-rule Green cubature "
                  "(identical refined R/K):", flush=True)
            softenings = (0.025, 0.018, 0.012)
            magnetic_rules = (
                ("A", (5, 8, 12, 3, 4)),
                ("B", (5, 9, 14, 4, 5)),
            )
            (ra, pa, aa, na, ta) = magnetic_rules[0][1]
            (rb, pb, ab, nb, tb) = magnetic_rules[1][1]
            ax, aw = scene.box_shell_quadrature(
                ra, pa, aa, wall_scale=enriched.wall_scale,
                shell_width_factor=config.shell_width_factor,
                shell_normal_order=na, shell_tangential_order=ta,
                outer_taper_width=enriched.outer_taper_width,
            )
            bx, bw = scene.box_shell_quadrature(
                rb, pb, ab, wall_scale=enriched.wall_scale,
                shell_width_factor=config.shell_width_factor,
                shell_normal_order=nb, shell_tangential_order=tb,
                outer_taper_width=enriched.outer_taper_width,
            )
            print(
                f"  rule A: {len(ax)} nodes; rule B: {len(bx)} nodes; "
                f"softenings={softenings}; same physical seawater domain",
                flush=True,
            )
            self_matrices, diagonal_numerator = (
                teacher.reintegrate_green_matrices(
                    ax, aw, softenings=softenings,
                    kernel_chunk_size=config.kernel_chunk_size,
                )
            )
            print("  self-rule Green L assembled; now cross-rule pairs",
                  flush=True)
            cross_matrices, overlap = (
                teacher.reintegrate_cross_green_matrices(
                    ax, aw, bx, bw, softenings=softenings,
                    kernel_chunk_size=config.kernel_chunk_size,
                )
            )
            print(f"  exactly coincident cross-pairs={overlap}",
                  flush=True)
            for a, ls, lc in zip(
                softenings, self_matrices, cross_matrices,
            ):
                old = metrics(ls, k_current)
                cross = metrics(lc, k_current)
                old_dual = metrics(ls, k_dual)
                cross_dual = metrics(lc, k_dual)
                eigen = np.linalg.eigvalsh(lc)
                print(
                    f"{'cross_vs_self':>20}: a={a:.4g} m, "
                    f"dL={relative(lc,ls):.6g}, "
                    f"self_diag_trace={diagonal_numerator/(a*max(np.trace(ls),1e-30)):.6g}, "
                    f"min_L_cross_eig={eigen[0]:.7g} H, "
                    f"dZ={relative(cross[0],old[0]):.6g}, "
                    f"dq={relative(cross[1],old[1]):.6g}, "
                    f"dP={abs(cross[2]-old[2])/max(abs(old[2]),1e-30):.6g}, "
                    f"dual_dZ={relative(cross_dual[0],old_dual[0]):.6g}, "
                    f"dual_dq={relative(cross_dual[1],old_dual[1]):.6g}, "
                    f"dual_dP={abs(cross_dual[2]-old_dual[2])/max(abs(old_dual[2]),1e-30):.6g}",
                    flush=True,
                )
            print(
                "Cross rule is symmetric but not guaranteed PSD; its "
                "lack of diagonal does not prove accuracy. Compare "
                "both quadrature rules jointly before changing the "
                "production Green operator.",
                flush=True,
            )
            return

        if green_ball:
            # Same refined R/K and unchanged Galerkin trial modes as the
            # Gaussian screen. Uniform *solid* clouds interact with the
            # exact unsoftened Coulomb kernel: K=1/d for disjoint balls,
            # analytic finite overlap/self values and a PSD Gram L.
            print("\nUniform-ball Coulomb Green L (PSD alternative):",
                  flush=True)
            magnetic_rules = (
                ("A", (5, 8, 12, 3, 4)),
                ("B", (5, 9, 14, 4, 5)),
            )
            results = {}
            gaussian_a = None
            for label, orders in magnetic_rules:
                radial, polar, azimuthal, normal, tangent = orders
                gx, gw = scene.box_shell_quadrature(
                    radial, polar, azimuthal,
                    wall_scale=enriched.wall_scale,
                    shell_width_factor=config.shell_width_factor,
                    shell_normal_order=normal,
                    shell_tangential_order=tangent,
                    outer_taper_width=enriched.outer_taper_width,
                )
                print(
                    f"  assembling uniform-ball rule {label}: "
                    f"{len(gx)} nodes, no blur or fitted width",
                    flush=True,
                )
                l, diagonal = teacher.reintegrate_ball_green_matrix(
                    gx, gw, kernel_chunk_size=config.kernel_chunk_size,
                )
                current = metrics(l, k_current)
                dual = metrics(l, k_dual)
                eigen = float(np.linalg.eigvalsh(l)[0])
                results[label] = (l, current, dual, eigen)
                print(
                    f"{'ball_L':>20}: rule={label}, "
                    f"trace={np.trace(l):.8g} H, "
                    f"diag_share={diagonal/max(np.trace(l),1e-30):.6g}, "
                    f"min_L_eig={eigen:.8g} H, "
                    f"vs_old_dL={relative(l,base_l):.6g}, "
                    f"vs_old_dZ={relative(current[0],original_base[0][0]):.6g}, "
                    f"vs_old_dq={relative(current[1],original_base[0][1]):.6g}, "
                    f"vs_old_dP={abs(current[2]-original_base[0][2])/max(abs(original_base[0][2]),1e-30):.6g}",
                    flush=True,
                )
                if label == "A":
                    print("  assembling Gaussian A at matched nodes for "
                          "shape-model discrepancy", flush=True)
                    gauss, _ = teacher.reintegrate_gaussian_green_matrices(
                        gx, gw, extra_blurs=(0.0,),
                        width_factor=1.0,
                        kernel_chunk_size=config.kernel_chunk_size,
                    )
                    gaussian_a = gauss[0]
                del gx, gw

            def report_ball(label, candidate, target):
                cl, cc, cd, _ = candidate
                tl, tc, td, _ = target
                dL = relative(cl, tl)
                dZ = relative(cc[0], tc[0])
                dq = relative(cc[1], tc[1])
                dP = abs(cc[2]-tc[2])/max(abs(tc[2]),1e-30)
                dual_dZ = relative(cd[0], td[0])
                dual_dq = relative(cd[1], td[1])
                dual_dP = abs(cd[2]-td[2])/max(abs(td[2]),1e-30)
                passed = (
                    dL <= 0.05 and max(dZ, dual_dZ) <= 0.002
                    and max(dq, dual_dq) <= 0.005
                    and max(dP, dual_dP) <= 0.002
                )
                print(
                    f"{label:>20}: dL={dL:.6g}, dZ={dZ:.6g}, "
                    f"dq={dq:.6g}, dP={dP:.6g}, "
                    f"dual_dZ={dual_dZ:.6g}, "
                    f"dual_dq={dual_dq:.6g}, "
                    f"dual_dP={dual_dP:.6g}, "
                    f"gate={'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
                return passed

            a = results["A"]
            b = results["B"]
            cross_pass = report_ball("ball_B_vs_A", b, a)
            gaussian_current = metrics(gaussian_a, k_current)
            gaussian_dual = metrics(gaussian_a, k_dual)
            shape_pass = report_ball(
                "ball_vs_gaussian_A",
                a, (gaussian_a, gaussian_current, gaussian_dual, 0.0),
            )
            psd_pass = all(item[3] >= -1e-12 for item in results.values())
            print(
                "  ball_screen: "
                f"cross_rule={'PASS' if cross_pass else 'FAIL'}, "
                f"shape_agreement={'PASS' if shape_pass else 'FAIL'}, "
                f"PSD={'PASS' if psd_pass else 'FAIL'}; "
                "promotion=HOLD until near-cell geometry/continuum "
                "and mode/domain/electrothermal checks.",
                flush=True,
            )
            return

        if green_gaussian:
            # A/B tests cubature at fixed physical width prescription.
            # A width-factor ladder tests that prescription separately,
            # sharing each pair-distance block rather than increasing
            # cubature orders or discarding the positive self energy.
            print("\nFinite-volume Gaussian Green L (PSD candidate):",
                  flush=True)
            rules = (
                ("A", (5, 8, 12, 3, 4)),
                ("B", (5, 9, 14, 4, 5)),
            )
            width_ladder = (0.75, 1.0, 1.25)
            reference = {}
            for label, orders in rules:
                radial, polar, azimuthal, normal, tangent = orders
                gx, gw = scene.box_shell_quadrature(
                    radial, polar, azimuthal,
                    wall_scale=enriched.wall_scale,
                    shell_width_factor=config.shell_width_factor,
                    shell_normal_order=normal,
                    shell_tangential_order=tangent,
                    outer_taper_width=enriched.outer_taper_width,
                )
                factors = width_ladder if label == "A" else (1.0,)
                print(
                    f"  assembling Gaussian rule {label}: {len(gx)} "
                    f"nodes, width factors={factors}, extra blur=0 m",
                    flush=True,
                )
                matrices, diagonals = (
                    teacher.reintegrate_gaussian_green_matrices(
                        gx, gw, extra_blurs=(0.0,),
                        width_factor=factors,
                        kernel_chunk_size=config.kernel_chunk_size,
                    )
                )
                del gx, gw
                for factor, l, diag in zip(factors, matrices, diagonals):
                    curl = metrics(l, k_current)
                    dual = metrics(l, k_dual)
                    eigen = float(np.linalg.eigvalsh(l)[0])
                    reference[(label, factor)] = (l, curl, dual, eigen)
                    print(
                        f"{'gaussian_L':>20}: rule={label}, "
                        f"width_factor={factor:.4g}, a=0 m, "
                        f"trace={np.trace(l):.8g} H, "
                        f"diag_share={diag/max(np.trace(l),1e-30):.6g}, "
                        f"min_L_eig={eigen:.8g} H, "
                        f"vs_old_dL={relative(l,base_l):.6g}, "
                        f"vs_old_dZ={relative(curl[0],original_base[0][0]):.6g}, "
                        f"vs_old_dq={relative(curl[1],original_base[0][1]):.6g}, "
                        f"vs_old_dP={abs(curl[2]-original_base[0][2])/max(abs(original_base[0][2]),1e-30):.6g}",
                        flush=True,
                    )

            # Fixed gates: 5% magnetic L, 0.2% port Z,
            # 0.5% local heat, 0.2% integrated dissipative power.
            thresholds = (0.05, 0.002, 0.005, 0.002)

            def compare_gaussian(candidate, target):
                lc, cc, dc, _ = candidate
                lr, cr, dr, _ = target
                differences = (
                    relative(lc, lr),
                    relative(cc[0], cr[0]),
                    relative(cc[1], cr[1]),
                    abs(cc[2]-cr[2])/max(abs(cr[2]),1e-30),
                )
                dual_differences = (
                    differences[0], relative(dc[0], dr[0]),
                    relative(dc[1], dr[1]),
                    abs(dc[2]-dr[2])/max(abs(dr[2]),1e-30),
                )
                passed = all(
                    value <= bound
                    for values in (differences, dual_differences)
                    for value, bound in zip(values, thresholds)
                )
                return differences, dual_differences, passed

            def report(label, candidate, target):
                values, dual_values, passed = compare_gaussian(
                    candidate, target,
                )
                print(
                    f"{label:>20}: "
                    f"dL={values[0]:.6g}, dZ={values[1]:.6g}, "
                    f"dq={values[2]:.6g}, dP={values[3]:.6g}, "
                    f"dual_dZ={dual_values[1]:.6g}, "
                    f"dual_dq={dual_values[2]:.6g}, "
                    f"dual_dP={dual_values[3]:.6g}, "
                    f"gate={'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
                return passed

            a = reference[("A", 1.0)]
            cross_passed = report(
                "gaussian_B_vs_A", reference[("B", 1.0)], a,
            )
            widths_passed = True
            for factor in width_ladder:
                if factor == 1.0:
                    continue
                passed = report(
                    f"width_{factor:g}_vs_A1", reference[("A", factor)], a,
                )
                widths_passed = widths_passed and passed
            psd_passed = all(
                result[3] >= -1e-12 for result in reference.values()
            )
            print(
                "  gaussian_screen: "
                f"cross_rule={'PASS' if cross_passed else 'FAIL'}, "
                f"width_sensitivity={'PASS' if widths_passed else 'FAIL'}, "
                f"PSD={'PASS' if psd_passed else 'FAIL'}; "
                "promotion=HOLD pending mode/domain/electrothermal checks "
                "and independent continuum corroboration.",
                flush=True,
            )
            return

        # At fixed Green nodes, a=25/18/12 mm isolates regularization.
        # Other rows hold a=25 mm and isolate independently wall/outer
        # magnetic cubature. Never compare different axes as a sequence.
        green_trials = (
            ("green_same_nodes", (5, 8, 12, 3, 4), (0.025, 0.018, 0.012)),
            ("green_wall", (5, 8, 12, 3, 5), (0.025,)),
            ("green_outer", (5, 10, 16, 3, 4), (0.025,)),
        )
        for label, orders, softenings in green_trials:
            radial, polar, azimuthal, normal, tangent = orders
            gx, gw = scene.box_shell_quadrature(
                radial, polar, azimuthal,
                wall_scale=enriched.wall_scale,
                shell_width_factor=config.shell_width_factor,
                shell_normal_order=normal,
                shell_tangential_order=tangent,
                outer_taper_width=enriched.outer_taper_width,
            )
            n_green = len(gx)
            print(
                f"  assembling {label}: {n_green} Green nodes, "
                f"{len(softenings)} regularization lengths",
                flush=True,
            )
            matrices, diagonal_numerator = (
                teacher.reintegrate_green_matrices(
                    gx, gw, softenings=softenings,
                    kernel_chunk_size=config.kernel_chunk_size,
                )
            )
            del gx, gw
            for a, l in zip(softenings, matrices):
                old_curl, old_dual = original_base
                z, q, p = metrics(l, k_current)
                zd, qd, pd = metrics(l, k_dual)
                print(
                    f"{label:>20}: nodes={n_green:6d}, "
                    f"a={a:.4g} m, "
                    f"dL={relative(l,base_l):.6g}, "
                    f"diag_trace={diagonal_numerator/(a*max(np.trace(l),1e-30)):.6g}, "
                    f"dZ={relative(z,old_curl[0]):.6g}, "
                    f"dq={relative(q,old_curl[1]):.6g}, "
                    f"dP={abs(p-old_curl[2])/max(abs(old_curl[2]),1e-30):.6g}, "
                    f"dual_dZ={relative(zd,old_dual[0]):.6g}, "
                    f"dual_dq={relative(qd,old_dual[1]):.6g}, "
                    f"dual_dP={abs(pd-old_dual[2])/max(abs(old_dual[2]),1e-30):.6g}",
                    flush=True,
                )
        print(
            "Independent magnetic cubature and softening are NOT "
            "a combined continuum limit; verify both axes separately.",
            flush=True,
        )
        return

    if wallpanels:
        # Fixed sphere and Green L: refine only oriented-box slab panels.
        # Stream R/K to avoid retaining hundreds of thousands of 63-mode
        # integration tensors. Full Joule power follows from c* R c.
        print("\nStreaming near-wall face/edge R/K/Joule convergence "
              "(fixed 63 modes, sphere and Green L):", flush=True)
        scheme = (
            ("cap050", 0.05, 5, 6),
            ("cap025", 0.025, 5, 6),
            ("cap020", 0.020, 5, 6),
            ("cap0167", 0.0167, 5, 6),
            ("cap0167_t7", 0.0167, 5, 7),
            ("cap0167_n6", 0.0167, 6, 6),
        )
        fixed = dict(
            wall_scale=enriched.wall_scale,
            shell_width_factor=config.shell_width_factor,
            outer_taper_width=enriched.outer_taper_width,
            max_radial_panel_width=0.06,
            radial_geometry_cuts=True,
            polar_geometry_order=8,
            azimuthal_panel_order=2,
        )
        # The shell blend and radial/angular rule are identical across
        # all experiments. Assemble the large sphere part only once.
        sphere_r = sphere_k = sphere_k_dual = None
        sphere_count = None
        probe_modes = np.einsum(
            "nmc,mr->nrc", enriched.evaluate(probes), teacher.transform,
            optimize=True,
        )
        omega = source.omega
        conductivity = scene.seawater.conductivity

        def summary(r, k):
            coefficients = np.linalg.solve(
                r + 1j * omega * teacher.inductance,
                -1j * omega * k,
            )
            impedance = 1j * omega * k.T @ coefficients
            impedance = 0.5 * (impedance + impedance.T)
            port_modes = coefficients @ currents
            j = np.einsum("nmc,m->nc", probe_modes, port_modes)
            heat = np.sum(np.abs(j)**2, axis=1) / (2 * conductivity)
            power = 0.5 * float(np.real(
                np.vdot(port_modes, r @ port_modes),
            ))
            return dict(R=r, K=k, Z=impedance, q=heat, P=power)

        def rel(a, b):
            return float(np.linalg.norm(a - b)
                         / max(np.linalg.norm(b), 1e-30))

        earlier = None
        fine_baseline = None
        for label, cap, normal, tangent in scheme:
            x, weights, split = scene.box_shell_quadrature(
                config.radial_order, config.polar_order,
                config.azimuthal_order, shell_normal_order=normal,
                shell_tangential_order=tangent,
                shell_tangential_max_panel_width=cap,
                return_sphere_count=True, **fixed,
            )
            if sphere_r is None:
                sphere_count = split
                sphere_r, sphere_k, sphere_k_dual = (
                    teacher.reintegrate_local_operators(
                        x[:split], weights[:split],
                    )
                )
                print(f"  fixed sphere nodes={sphere_count}; "
                      "assembled only once", flush=True)
            elif split != sphere_count:
                raise RuntimeError(
                    "spherical quadrature changed during wall-only refinement"
                )
            slab_r, slab_k, slab_k_dual = (
                teacher.reintegrate_local_operators(
                    x[split:], weights[split:],
                )
            )
            r = sphere_r + slab_r
            current = summary(r, sphere_k + slab_k)
            dual = summary(r, sphere_k_dual + slab_k_dual)
            print(
                f"{label:>20}: slab_nodes={len(x)-split:6d}, "
                f"P={current['P']:.8g} W, "
                f"dual_P={dual['P']:.8g} W, "
                f"weak_dK={rel(dual['K'],current['K']):.6g}, "
                f"weak_dZ={rel(dual['Z'],current['Z']):.6g}",
                flush=True,
            )
            baseline = (
                fine_baseline if label in ("cap0167_t7", "cap0167_n6")
                else earlier
            )
            if baseline is not None:
                for form, now, old in (
                    ("curl_K_step", current, baseline[1]),
                    ("dual_K_step", dual, baseline[2]),
                ):
                    now_q, old_q = now["q"], old["q"]
                    print(
                        f"{form:>20}: from={baseline[0]}, "
                        f"dR={rel(now['R'],old['R']):.6g}, "
                        f"dK={rel(now['K'],old['K']):.6g}, "
                        f"dZ={rel(now['Z'],old['Z']):.6g}, "
                        f"dq={rel(now_q,old_q):.6g}, "
                        f"max_probe_dq={np.max(np.abs(now_q-old_q)/np.maximum(np.abs(old_q),1e-30)):.6g}, "
                        f"dP={abs(now['P']-old['P'])/max(abs(old['P']),1e-30):.6g}",
                        flush=True,
                    )
            if label == "cap0167":
                fine_baseline = label, current, dual
            earlier = label, current, dual
            del x, weights
        print(
            "Cap refinements are successive; t7/n6 compare independently "
            "against cap0167 at fixed panels. Green L is unchanged.",
            flush=True,
        )
        return

    if kparts:
        # The two continuous curl-integration forms agree only after
        # integrating BOTH pieces of the partition-of-unity quadrature.
        # Isolate the sphere and box-wall contributions on identical
        # positive points; angular/radial resolution is already high.
        print("\nSphere/slab source K weak-form gap localization:",
              flush=True)
        print("Fixed radial geometry cuts and 0.06 m maximum panel "
              "width; same 63 modes and Green L.", flush=True)
        previous = None

        def relative(a, b):
            return float(np.linalg.norm(a - b)
                         / max(np.linalg.norm(b), 1e-30))

        schemes = (
            ("wall4_5", 4, 5, 8, 2),
            ("wall5_6", 5, 6, 8, 2),
            ("wall6_7", 6, 7, 8, 2),
            ("wall8_9", 8, 9, 8, 2),
            ("wall8_9_mu10", 8, 9, 10, 2),
        )
        for label, normal, tangential, mu, phi in schemes:
            cfg = replace(
                config, polar_geometry_order=mu,
                azimuthal_panel_order=phi, radial_geometry_cuts=True,
                max_radial_panel_width=0.06,
                shell_normal_order=normal,
                shell_tangential_order=tangential,
            )
            points, weights, sphere_count = scene.box_shell_quadrature(
                cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
                wall_scale=enriched.wall_scale,
                shell_width_factor=cfg.shell_width_factor,
                shell_normal_order=cfg.shell_normal_order,
                shell_tangential_order=cfg.shell_tangential_order,
                shell_tangential_max_panel_width=cfg.shell_tangential_max_panel_width,
                outer_taper_width=enriched.outer_taper_width,
                max_radial_panel_width=cfg.max_radial_panel_width,
                radial_geometry_cuts=cfg.radial_geometry_cuts,
                polar_geometry_order=cfg.polar_geometry_order,
                azimuthal_panel_order=cfg.azimuthal_panel_order,
                return_sphere_count=True,
            )
            full = teacher.reintegrate_local_response(points, weights)
            sphere_points = points[:sphere_count]
            sphere_weights = weights[:sphere_count]
            slab_points = points[sphere_count:]
            slab_weights = weights[sphere_count:]
            k_sphere = teacher.reintegrate_source_coupling(
                sphere_points, sphere_weights,
                coupling_form="current_potential",
            )
            k_sphere_dual = teacher.reintegrate_source_coupling(
                sphere_points, sphere_weights, coupling_form="curl_dual",
            )
            k_slab_dual = teacher.reintegrate_source_coupling(
                slab_points, slab_weights, coupling_form="curl_dual",
            )
            k_slab = full.coupling - k_sphere
            k_dual = k_sphere_dual + k_slab_dual
            response_dual = full.resolve_with_coupling(k_dual)
            delta_sphere = k_sphere_dual - k_sphere
            delta_slab = k_slab_dual - k_slab
            delta_total = delta_sphere + delta_slab
            norm_k = max(np.linalg.norm(full.coupling), 1e-30)
            alignment = float(np.vdot(
                delta_sphere, delta_slab,
            ).real / max(
                np.linalg.norm(delta_sphere) * np.linalg.norm(delta_slab),
                1e-30,
            ))
            q_curl = full.seawater_loss(probes, currents)
            q_dual = response_dual.seawater_loss(probes, currents)
            p_curl = full.port_seawater_power(currents)
            p_dual = response_dual.port_seawater_power(currents)
            print(
                f"{label:>20}: sphere_nodes={sphere_count:6d}, "
                f"slab_nodes={len(points)-sphere_count:6d}, "
                f"normK={norm_k:.8g}",
                flush=True,
            )
            print(
                f"{'K_gap_by_region':>20}: "
                f"sphere={np.linalg.norm(delta_sphere)/norm_k:.6g}, "
                f"slab={np.linalg.norm(delta_slab)/norm_k:.6g}, "
                f"total={np.linalg.norm(delta_total)/norm_k:.6g}, "
                f"alignment={alignment:.6g}, "
                f"curl_slab_share={np.linalg.norm(k_slab)/norm_k:.6g}",
                flush=True,
            )
            print(
                f"{'weak_form_response':>20}: "
                f"dZ={relative(response_dual.impedance_seawater,full.impedance_seawater):.6g}, "
                f"dq={relative(q_dual,q_curl):.6g}, "
                f"max_probe_dq={np.max(np.abs(q_dual-q_curl)/np.maximum(np.abs(q_curl),1e-30)):.6g}, "
                f"dP={abs(p_dual-p_curl)/max(abs(p_curl),1e-30):.6g}",
                flush=True,
            )
            if previous is not None:
                old_label, old_curl, old_dual = previous
                for name, response, old in (
                    ("curl_K_step", full, old_curl),
                    ("dual_K_step", response_dual, old_dual),
                ):
                    q = response.seawater_loss(probes, currents)
                    old_q = old.seawater_loss(probes, currents)
                    p = response.port_seawater_power(currents)
                    old_p = old.port_seawater_power(currents)
                    print(
                        f"{name:>20}: from={old_label}, "
                        f"dR={relative(response.resistance,old.resistance):.6g}, "
                        f"dK={relative(response.coupling,old.coupling):.6g}, "
                        f"dZ={relative(response.impedance_seawater,old.impedance_seawater):.6g}, "
                        f"dq={relative(q,old_q):.6g}, "
                        f"max_probe_dq={np.max(np.abs(q-old_q)/np.maximum(np.abs(old_q),1e-30)):.6g}, "
                        f"dP={abs(p-old_p)/max(abs(old_p),1e-30):.6g}",
                        flush=True,
                    )
            previous = label, full, response_dual
        print(
            "Only the final mu10 row changes angular order. "
            "All earlier rows change the near-wall Gauss rule alone.",
            flush=True,
        )
        return

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
        stable_geometry_dual = None
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
                # Compare curl-mode C coupling with its boundary-dual
                # primitive-mode B coupling on exactly identical nodes.
                k_dual = teacher.reintegrate_source_coupling(
                    points, weights, coupling_form="curl_dual",
                )
                stable_geometry_dual = full.resolve_with_coupling(k_dual)
                dual_q = stable_geometry_dual.seawater_loss(probes, currents)
                print(
                    f"{'geometry_mu8_dual_K':>20}: "
                    f"vs_curl_dK={relative(k_dual, full.coupling):.6g}, "
                    f"vs_curl_dZ={relative(stable_geometry_dual.impedance_seawater, z):.6g}, "
                    f"vs_curl_dq={relative(dual_q, q):.6g}, "
                    f"vs_curl_dP={abs(stable_geometry_dual.port_seawater_power(currents)-power)/max(abs(power),1e-30):.6g}",
                    flush=True,
                )
            previous = label, full

        # With the converged-in-angle geometry_mu8 rule as the *new*
        # comparator, independently perturb radial and wall cubature.
        # These are not successive refinements of the same axis.
        if stable_geometry is None or stable_geometry_dual is None:
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
            k_dual = teacher.reintegrate_source_coupling(
                points, weights, coupling_form="curl_dual",
            )
            dual = modified.resolve_with_coupling(k_dual)
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
            dual_q = dual.seawater_loss(probes, currents)
            base_dual_q = stable_geometry_dual.seawater_loss(
                probes, currents,
            )
            dual_p = dual.port_seawater_power(currents)
            base_dual_p = stable_geometry_dual.port_seawater_power(currents)
            print(
                f"{'curl_dual_K':>20}: "
                f"vs_geom8_dZ={np.linalg.norm(dual.impedance_seawater-stable_geometry_dual.impedance_seawater)/max(np.linalg.norm(stable_geometry_dual.impedance_seawater),1e-30):.6g}, "
                f"vs_geom8_dq={np.linalg.norm(dual_q-base_dual_q)/max(np.linalg.norm(base_dual_q),1e-30):.6g}, "
                f"max_probe_dq={np.max(np.abs(dual_q-base_dual_q)/np.maximum(np.abs(base_dual_q),1e-30)):.6g}, "
                f"vs_geom8_dP={abs(dual_p-base_dual_p)/max(abs(base_dual_p),1e-30):.6g}, "
                f"P={dual_p:.8g} W",
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
    basis_radial: bool = False, basis_kparts: bool = False,
    basis_wallpanels: bool = False,
    basis_green_refined: bool = False,
    basis_green_cross: bool = False,
    basis_green_gaussian: bool = False,
    basis_green_ball: bool = False,
    basis_ball: bool = False,
    basis_ball_check_l: bool = False,
    basis_ball_check_rk: bool = False,
    basis_ball_ablate: bool = False,
    basis_ball_face: bool = False,
    basis_ball_edge: bool = False,
    basis_ball_aniso: bool = False,
    basis_domain_ball: bool = False,
):
    if (basis_only or basis_verify or basis_green or basis_domain
            or basis_local or basis_angular or basis_geometry
            or basis_radial or basis_kparts or basis_wallpanels
            or basis_green_refined or basis_green_cross
            or basis_green_gaussian or basis_green_ball
            or basis_ball or basis_domain_ball or basis_ball_check_l
            or basis_ball_check_rk or basis_ball_ablate
            or basis_ball_face or basis_ball_edge or basis_ball_aniso):
        return basis_convergence(
            verify=basis_verify, green=basis_green,
            domain=basis_domain or basis_domain_ball,
            local=basis_local, angular=basis_angular,
            geometry=basis_geometry, radial=basis_radial,
            kparts=basis_kparts, wallpanels=basis_wallpanels,
            green_refined=basis_green_refined,
            green_cross=basis_green_cross,
            green_gaussian=basis_green_gaussian,
            green_ball=basis_green_ball,
            ball_reference=basis_ball or basis_domain_ball or basis_ball_check_l
            or basis_ball_check_rk or basis_ball_ablate
            or basis_ball_face or basis_ball_edge or basis_ball_aniso,
            ball_l_check=basis_ball_check_l,
            rk_normal_check=basis_ball_check_rk,
            mode_ablation=basis_ball_ablate,
            source_face_guided=(
                basis_ball_face or basis_ball_edge or basis_ball_aniso
            ),
            edge_refinement=basis_ball_edge or basis_ball_aniso,
            anisotropic_refinement=basis_ball_aniso,
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
        and sys.argv[1] not in ("--full", "--basis", "--basis-verify", "--basis-green", "--basis-domain", "--basis-local", "--basis-angular", "--basis-geometry", "--basis-radial", "--basis-kparts", "--basis-wallpanels", "--basis-green-refined", "--basis-green-cross", "--basis-green-gaussian", "--basis-green-ball", "--basis-ball", "--basis-ball-check-l", "--basis-ball-face", "--basis-domain-ball")
    ):
        raise SystemExit(
            "usage: python -m examples.bfzi_convergence "
            "[--full|--basis|--basis-local|--basis-angular|--basis-geometry|--basis-radial|--basis-kparts|--basis-wallpanels|--basis-green-refined|--basis-green-cross|--basis-green-gaussian|--basis-green-ball|--basis-ball|--basis-ball-check-l|--basis-ball-face|--basis-domain-ball|--basis-green|--basis-domain|--basis-verify]"
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
        basis_radial="--basis-radial" in sys.argv[1:],
        basis_kparts="--basis-kparts" in sys.argv[1:],
        basis_wallpanels="--basis-wallpanels" in sys.argv[1:],
        basis_green_refined="--basis-green-refined" in sys.argv[1:],
        basis_green_cross="--basis-green-cross" in sys.argv[1:],
        basis_green_gaussian="--basis-green-gaussian" in sys.argv[1:],
        basis_green_ball="--basis-green-ball" in sys.argv[1:],
        basis_ball="--basis-ball" in sys.argv[1:],
        basis_ball_check_l="--basis-ball-check-l" in sys.argv[1:],
        basis_ball_face="--basis-ball-face" in sys.argv[1:],
        basis_domain_ball="--basis-domain-ball" in sys.argv[1:],
    )
