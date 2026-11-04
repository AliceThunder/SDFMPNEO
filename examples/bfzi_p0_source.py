"""P0 only: check the *physical source* without assembling seawater R/K/L.

Run through run.py --mode bfzi-p0-source --with-tests to auto-publish the
short source contract, source-order convergence and MQS regime estimates.
"""
from __future__ import annotations

import math
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.source import build_line_source, MU0
from sdfmpneo_bfzi.source_reference import continuous_port_field


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
        for order in (3,4,6,8,12):
            line = build_line_source(assembly, order=order)
            approx_c = line.potential_kernel(positions)
            approx_b = np.real(line.magnetic_field(positions))
            closure = np.linalg.norm(line.line_elements.sum(axis=0))
            arc = np.linalg.norm(line.line_elements, axis=1).sum()
            print(
                f"p0_source: port={port},order={order}, "
                f"dC={relative(approx_c,ref_c):.6g}, "
                f"dB={relative(approx_b,ref_b):.6g}, "
                f"line_closure_over_length={closure/max(arc,1e-30):.6g}",
                flush=True,
            )
    print(
        "p0_note: C/B fields are checked against analytic straight lines "
        "+ independent high-order curved-corner Gauss, NOT against a "
        "full-wave/copper-proximity or infinite-seawater solution. "
        "No source order is certified until the measured differences "
        "meet a predeclared accuracy target.",
        flush=True,
    )


if __name__ == "__main__":
    main()
