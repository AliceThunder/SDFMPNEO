"""P0 only: check the *physical source* without assembling seawater R/K/L.

Run through run.py --mode bfzi-p0-source --with-tests to auto-publish the
short source contract, source-order convergence and MQS regime estimates.
"""
from __future__ import annotations

import math
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.source import LineSource, build_line_source, MU0
from sdfmpneo_bfzi.source_reference import (
    continuous_port_field, polygonal_port_field,
)


def probes_for_scene(scene):
    """Reproducible near-wall and remote wet points, in world coordinates."""
    xs = []
    for assembly in scene.assemblies:
        half = np.asarray(assembly.enclosure.half_extents)
        for direction in range(3):
            for side in (-1.0, 1.0):
                for distance in (0.0025, 0.025):
                    local = np.zeros(3)
                    local[direction] = side*(half[direction]+distance)
                    xs.append(assembly.pose.position(local))
    # Real OBB edges and corners, not just face-center rays.
    # Each local point is transported with its *own* assembly pose.
    # Every wall query remains 2.5 or 25 mm outside the sealed housing.
    for assembly in scene.assemblies:
        half = np.asarray(assembly.enclosure.half_extents)
        for distance in (0.0025, 0.025):
            for i, j in ((0, 1), (0, 2), (1, 2)):
                for sign_i in (-1.0, 1.0):
                    for sign_j in (-1.0, 1.0):
                        local = np.zeros(3)
                        local[i] = sign_i * (half[i] + distance)
                        local[j] = sign_j * (half[j] + distance)
                        xs.append(assembly.pose.position(local))
            for sign in (-1.0, 1.0):
                local = sign * (half + distance)
                xs.append(assembly.pose.position(local))
    for p in (
        (0.28, 0.07, 0.19),
        (-0.24, 0.19, -0.05),
        (0.11, -0.26, 0.13),
        (-0.16, -0.11, -0.30),
    ):
        xs.append(p)
    points = np.asarray(xs, dtype=float)
    if not scene.in_seawater(points).all():
        raise ValueError("P0 source probe is not in conducting seawater")
    return points


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)), 1e-30))


def peak_scaled_max(a, b):
    """Worst *absolute* vector error divided by the reference peak."""
    return float(
        np.max(np.linalg.norm(a-b, axis=-1))
        / max(float(np.max(np.linalg.norm(b, axis=-1))), 1e-30)
    )


def main():
    scene = build_demo_scene()
    positions = probes_for_scene(scene)
    f = 200000.0
    omega = 2.0*math.pi*f
    sigma = scene.seawater.conductivity
    delta = math.sqrt(2.0/(omega*MU0*sigma))
    # eps_r=80 is a conditional seawater example, NOT a material
    # parameter in the current MQS solver and NOT a dielectric certificate.
    eps_rel_example=80.0
    eps0=8.8541878128e-12
    print(
        "p0_physics: model=linear_MQS, source=prescribed_closed_ideal_line, "
        f"f_hz={f:.7g},sigma_s={sigma:.7g},mu_rel=1, "
        f"sea_skin_depth_m={delta:.7g}, "
        f"omega_eps_over_sigma_if_epsr80={omega*eps0*eps_rel_example/sigma:.7g}, "
        f"air_k0_R={omega*scene.outer_radius/299792458.0:.7g}, "
        "insulating_boxes=2,outer_sphere=artificial_no_normal_current, "
        "finite_wire_proximity=excluded,displacement_current=excluded",
        flush=True,
    )
    for port, assembly in enumerate(scene.assemblies):
        # The axis-aligned local box is convex. Bounding the entire
        # continuous spiral by A0+beta and every return vertex by the
        # enclosure half-extents proves their entire radius-r sweep
        # remains inside the sealed box. This does NOT prove self-clearance.
        half=np.asarray(assembly.enclosure.half_extents)
        r=assembly.wire_radius
        coil_xy_margin=(
            min(half[:2])-assembly.coil.outer_half_size-assembly.coil.beta-r
        )
        coil_z_margin=half[2]-r
        outer,inner=assembly.coil.endpoints()
        return_vertices=np.vstack((
            inner,np.asarray(assembly.return_waypoints),outer,
        ))
        route_wall_margin=float(
            np.min(half-np.abs(return_vertices))-r
        )
        print(
            f"p0_geometry: port={port},"
            f"coil_xy_margin_m={coil_xy_margin:.7g},"
            f"coil_z_margin_m={coil_z_margin:.7g},"
            f"return_wall_margin_m={route_wall_margin:.7g},"
            "finite_wire_self_clearance=not_certified",
            flush=True,
        )
        order_metrics = {}
        ref_c, ref_b = continuous_port_field(
            assembly, positions, corner_order=32,
        )
        ref_c24, ref_b24 = continuous_port_field(
            assembly, positions, corner_order=24,
        )
        print(
            f"p0_reference: port={port},probe_count={len(positions)}, "
            f"corner_24_to_32_dC={relative(ref_c24,ref_c):.6g}, "
            f"corner_24_to_32_dB={relative(ref_b24,ref_b):.6g}",
            flush=True,
        )
        for panels in (16, 32, 64):
            pc,pb,pdiv=polygonal_port_field(
                assembly,positions,panels_per_corner=panels,
            )
            print(
                f"p0_polygon: port={port},corner_panels={panels}, "
                f"dC={relative(pc,ref_c):.6g}, "
                f"dB={relative(pb,ref_b):.6g}, "
                f"divC_scaled={0.1*np.linalg.norm(pdiv)/max(np.linalg.norm(pc),1e-30):.6g}",
                flush=True,
            )
        for order in (3,4,6,8,12):
            line = build_line_source(assembly, order=order)
            approx_c = line.potential_kernel(positions)
            approx_b = np.real(line.magnetic_field(positions))
            # Historical all-Gauss source on the *same* physical path.
            # This isolates the effect of the analytic straight+return
            # integration from corner-order or geometry changes.
            old = LineSource(line.points, line.line_elements)
            old_c = old.potential_kernel(positions)
            old_b = np.real(old.magnetic_field(positions))
            closure = np.linalg.norm(line.line_elements.sum(axis=0))
            arc = np.linalg.norm(line.line_elements, axis=1).sum()
            dC = relative(approx_c,ref_c)
            dB = relative(approx_b,ref_b)
            order_metrics[order]=(dC,dB)
            div = line.divergence_kernel(positions)
            # div C has units 1/metre. Normalize with enclosure
            # characteristic length; an exact closed filament has div C=0.
            characteristic=0.10
            div_scaled = (
                characteristic*np.linalg.norm(div)
                / max(np.linalg.norm(approx_c),1e-30)
            )
            print(
                f"p0_source: port={port},order={order}, "
                f"dC={dC:.6g},dB={dB:.6g}, "
                f"dC_faces={relative(approx_c[:24],ref_c[:24]):.6g}, "
                f"dB_faces={relative(approx_b[:24],ref_b[:24]):.6g}, "
                f"dC_edges={relative(approx_c[24:-4],ref_c[24:-4]):.6g}, "
                f"dB_edges={relative(approx_b[24:-4],ref_b[24:-4]):.6g}, "
                f"dB_peak_max={peak_scaled_max(approx_b,ref_b):.6g}, "
                f"divC_scaled={div_scaled:.6g}, "
                f"legacy_dC={relative(old_c,ref_c):.6g}, "
                f"legacy_dB={relative(old_b,ref_b):.6g}, "
                f"line_closure_over_length={closure/max(arc,1e-30):.6g}",
                flush=True,
            )
        for order in (4,12):
            dc,db=order_metrics[order]
            print(
                f"p0_gate: port={port},source_order={order}, "
                f"source_dC_le_0p1pct={dc<0.001}, "
                f"source_dB_le_0p1pct={db<0.001}",
                flush=True,
            )
    print(
        "p0_note: C/B fields are checked against analytic straight lines "
        "+ independent high-order curved-corner Gauss, NOT against a "
        "full-wave/copper-proximity or infinite-seawater solution. "
        "P0 polygon uses actual closed finite segments and analytic kernels: "
        "its divC vanishes by telescoping but chord-shape error still needs "
        "panel refinement. Neither source method certifies finite-radius wire "
        "self-clearance, full Maxwell fields, or continuous seawater R/K/L.",
        flush=True,
    )


if __name__ == "__main__":
    main()
