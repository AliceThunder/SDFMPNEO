"""M1: real two-port UMFC + dual sealed boxes + seawater A-phi direct solve.

Run from repository root:
  python -m examples.bfzi_m1_direct --output runs/m1.json

Uses a modest, conforming axis-aligned mesh for an END-TO-END prototype.
For independent arbitrary SE(3) CAD/mesh validation, use bfzi_d2_aphi
--generate-msh first, then pass --mesh with the same physical scene.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.geometry import RigidPose
from sdfmpneo_bfzi.direct3d_mesh import bfzi_mesh_from_gmsh22
from sdfmpneo_bfzi.direct3d_structured_bfzi import axis_aligned_bfzi_tetra_mesh
from sdfmpneo_bfzi.m1_direct import solve_m1


def pair(z):
    return np.stack((np.asarray(z).real, np.asarray(z).imag), axis=-1).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frequency", type=float, default=200000.)
    ap.add_argument("--mesh", type=Path, help="Physical-tag Gmsh 2.2 mesh")
    ap.add_argument("--far-step", type=float, default=.50)
    ap.add_argument("--panels", type=int, default=3)
    ap.add_argument("--rx-rotvec", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--probe", type=float, nargs=3, action="append")
    ap.add_argument("--currents", type=float, nargs=4, default=(1, 0, .4, .3),
                    metavar=("I0_RE", "I0_IM", "I1_RE", "I1_IM"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)

    scene = build_demo_scene()
    rot = Rotation.from_rotvec(args.rx_rotvec).as_matrix()
    rx = scene.receiver
    rx = replace(rx, pose=RigidPose(rot@rx.pose.rotation, rx.pose.center))
    scene = replace(scene, receiver=rx)
    if args.mesh:
        mesh = bfzi_mesh_from_gmsh22(args.mesh, scene)
        mesh_kind = "physical-tag Gmsh OCC SE(3)"
    else:
        mesh = axis_aligned_bfzi_tetra_mesh(
            scene, far_step_m=args.far_step, box_layers=1)
        mesh_kind = "axis-aligned conforming smoke"
    result = solve_m1(scene, mesh, args.frequency,
                      panels_per_corner=args.panels)
    i = np.asarray([complex(args.currents[0], args.currents[1]),
                    complex(args.currents[2], args.currents[3])])
    x = np.asarray(args.probe or [[0, 0, 0], [.12, 0, 0]], dtype=float)
    sample = result.fields(x, i)
    pos, dw, q = result.heat_nodes(i)
    p_volume = result.volume_power(i)
    p_port = result.port_sea_power(i)
    report = {
        "status": "M1 functional prototype; D2 independent accuracy OPEN",
        "scene": "two real closed continuous BFZI spirals / dual insulating boxes",
        "mesh": mesh_kind,
        "tetrahedra": len(mesh.tetrahedra),
        "frequency_hz": args.frequency,
        "currents_A": pair(i),
        "unit_port_Zsea_ohm": pair(result.z_sea),
        "Zem_ohm": pair(result.z_em),
        "sea_dissipation_channel_ohm": pair(result.z_loss),
        "probe_m": x.tolist(),
        "unit_E_V_m": pair(sample["unit_E_V_m"]),
        "unit_B_T": pair(sample["unit_B_T"]),
        "E_V_m": pair(sample["E_V_m"]),
        "B_T": pair(sample["B_T"]),
        "J_A_m2": pair(sample["J_A_m2"]),
        "q_W_m3": sample["q_W_m3"].tolist(),
        "seawater_power_volume_W": p_volume,
        "seawater_power_heat_integral_W": float(dw@q),
        "seawater_power_port_W": p_port,
        "port_volume_power_gap_W": abs(p_port-p_volume),
        "port_reciprocity_gap_ohm": float(abs(result.z_sea[0,1]-result.z_sea[1,0])),
        "linear_residual": result.physical.reaction.linear_relative_residual,
        "limitations": "Finite outer cube, first-order Nedelec, polygonal source; "
                       "h/R/port-power independence still OPEN; not a certified neural teacher.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "tetrahedra": report["tetrahedra"],
        "Zsea_ohm": report["unit_port_Zsea_ohm"],
        "P_volume_W": p_volume,
        "P_heat_W": report["seawater_power_heat_integral_W"],
        "P_port_W": p_port,
        "q_probe_W_m3": report["q_W_m3"],
        "linear_residual": report["linear_residual"],
        "status": report["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
