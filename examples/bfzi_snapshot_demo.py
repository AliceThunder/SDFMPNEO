"""BFZI operator-only snapshot ROM: solve, save, load, and field inference.

This deliberately uses a *small* provisional Galerkin teacher so the
end-to-end mechanism can be exercised cheaply. It is NOT the converged
electromagnetic heat reference and not a trained cross-geometry network.

Run: python run.py --mode bfzi-snapshot --out bfzi_response.npz
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import numpy as np

from .bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import (
    BFZISnapshotROM, SeaMQSConfig, RigidPose, build_bfzi_source,
    build_sea_basis, solve_seawater_reference,
    BFZIResponseLabel, save_response_labels,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="bfzi_response.npz",
                        help="output .npz snapshot artifact (numeric, no pickle)")
    parser.add_argument("--dataset-out",default=None,
                        help="also write two separate provisional geometry labels")
    args = parser.parse_args(argv)
    scene = build_demo_scene()
    source = build_bfzi_source(scene,200_000.0,order=4)
    basis = build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    cfg = SeaMQSConfig(
        radial_order=4,polar_order=6,azimuthal_order=8,
        shell_normal_order=2,shell_tangential_order=3,
        field_softening=0.025,inductance_kernel="uniform_ball",
        inductance_quadrature=(4,5,8,2,3),
    )
    parent = solve_seawater_reference(source,basis=basis,config=cfg)
    rom = BFZISnapshotROM.from_operators(
        source,basis,parent.transform,
        parent.resistance,parent.inductance,parent.coupling,
        inductance_kernel=cfg.inductance_kernel,
    )
    rom.save(args.out)
    loaded = BFZISnapshotROM.load(args.out,source,basis)
    current = np.array([1.0+0.2j,-0.4+0.5j])
    probes = np.array([
        [0.0,0.0,0.0],
        [0.13,0.0,0.0],
        [0.0,0.13,0.0],
        [0.0,0.0,0.22],
    ])
    z_error = np.linalg.norm(
        loaded.impedance_seawater-parent.impedance_seawater
    )/max(np.linalg.norm(parent.impedance_seawater),1e-30)
    q_ref = parent.seawater_loss(probes,current)
    q_rom = loaded.seawater_loss(probes,current)
    q_error = np.linalg.norm(q_ref-q_rom)/max(np.linalg.norm(q_ref),1e-30)
    power_error = abs(
        loaded.port_seawater_power(current)
        - loaded.modal_seawater_power(current)
    )
    print(f"Saved snapshot: {args.out}",flush=True)
    print(f"Physical parent={parent.resistance.shape[0]} modes; "
          f"real snapshot rank={rom.rank}; kernel={rom.inductance_kernel}",
          flush=True)
    print(f"Same-parent relative port Z error={z_error:.8g}, "
          f"probe Joule error={q_error:.8g}, "
          f"power closure error={power_error:.8g} W",flush=True)
    print("UNCERTIFIED DEMO: rank-4 reproduces ONLY the provisional "
          "fixed-scene parent. No cross-geometry training or spatial "
          "continuum accuracy is claimed.",flush=True)
    if args.dataset_out is not None:
        # Second geometry is a genuinely different receiver rigid pose,
        # not a re-sampled probe from the same reference response.
        modified_receiver = replace(
            scene.receiver,
            pose=RigidPose(
                scene.receiver.pose.rotation,
                np.asarray(scene.receiver.pose.center) + [0.008,0.0,0.015],
            ),
        )
        varied_scene = replace(scene, receiver=modified_receiver)
        varied_source = build_bfzi_source(
            varied_scene, 200_000.0, order=4,
        )
        varied_basis = build_sea_basis(
            varied_scene, modes_per_face=2, face_refinement=1,
        )
        varied_parent = solve_seawater_reference(
            varied_source, basis=varied_basis, config=cfg,
        )
        varied_snapshot = BFZISnapshotROM.from_operators(
            varied_source, varied_basis, varied_parent.transform,
            varied_parent.resistance, varied_parent.inductance,
            varied_parent.coupling, inductance_kernel=cfg.inductance_kernel,
        )
        save_response_labels(
            args.dataset_out, (
                BFZIResponseLabel.from_snapshot(rom),
                BFZIResponseLabel.from_snapshot(varied_snapshot),
            )
        )
        print(f"Saved 2 independent PROVISIONAL geometry labels: "
              f"{args.dataset_out}",flush=True)
        print("Dataset records contain geometry features, two-port raw "
              "current response coefficients and reciprocal Zsea; "
              "neither the labels nor the tiny dataset have been "
              "certified for neural training.",flush=True)


if __name__ == "__main__":
    main()
