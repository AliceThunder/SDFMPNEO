"""Sample independent BFZI geometry/pose references for experimental learning.

Every sample runs a fresh complete-port MQS solve. The default 27-mode
uniform-ball reference is PROVISIONAL and must NOT be used to claim
physical accuracy. Only a lightweight engineering pipeline test.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from .bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import (
    BFZIResponseLabel, BFZISnapshotROM, RigidPose,
    SeaMQSConfig, Seawater, build_bfzi_source, build_sea_basis,
    save_response_labels, solve_seawater_reference,
)


def make_scene(seed_scene, rng):
    """Small, reproducible two-pose, package, coil-shape and medium perturbations."""
    coil = seed_scene.receiver.coil
    coil_new = replace(
        coil,
        outer_half_size=coil.outer_half_size * rng.uniform(0.985, 1.015),
        corner_radius=coil.corner_radius * rng.uniform(0.985, 1.015),
        alpha=float(np.clip(coil.alpha + rng.uniform(-0.04, 0.04), 0.1, 0.9)),
    )
    pose = seed_scene.receiver.pose
    rot = Rotation.from_rotvec(
        rng.uniform(-0.10, 0.10, size=3)
    ).as_matrix() @ pose.rotation
    center = np.asarray(pose.center) + rng.uniform(
        -0.012, 0.012, size=3,
    )
    receiver = replace(
        seed_scene.receiver,coil=coil_new,
        pose=RigidPose(rot, center),
        enclosure=replace(
            seed_scene.receiver.enclosure,
            half_extents=tuple(
                np.asarray(seed_scene.receiver.enclosure.half_extents)
                *rng.uniform(0.98,1.02,size=3)
            ),
        ),
    )
    transmitter = seed_scene.transmitter
    tx_coil = replace(
        transmitter.coil,
        alpha=float(np.clip(
            transmitter.coil.alpha+rng.uniform(-0.04,0.04),0.1,0.9,
        )),
    )
    tx_pose=transmitter.pose
    tx_rot=Rotation.from_rotvec(
        rng.uniform(-0.06,0.06,size=3)
    ).as_matrix() @ tx_pose.rotation
    transmitter = replace(
        transmitter,coil=tx_coil,
        pose=RigidPose(
            tx_rot,np.asarray(tx_pose.center)+rng.uniform(
                -0.006,0.006,size=3
            ),
        ),
        enclosure=replace(
            transmitter.enclosure,
            half_extents=tuple(
                np.asarray(transmitter.enclosure.half_extents)
                *rng.uniform(0.98,1.02,size=3)
            ),
        ),
    )
    return replace(
        seed_scene, transmitter=transmitter, receiver=receiver,
        seawater=Seawater(
            seed_scene.seawater.conductivity*rng.uniform(0.92,1.08)
        ),
    )


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples",type=int,default=12)
    parser.add_argument("--seed",type=int,default=23)
    parser.add_argument("--out",default="bfzi_geometry_labels.npz")
    parser.add_argument("--source-order",type=int,default=4)
    args=parser.parse_args(argv)
    if args.samples<4:
        parser.error("--samples must be >=4 for a whole-geometry holdout")
    rng=np.random.default_rng(args.seed)
    baseline=build_demo_scene()
    config=SeaMQSConfig(
        radial_order=4,polar_order=6,azimuthal_order=8,
        shell_normal_order=2,shell_tangential_order=3,
        inductance_kernel="uniform_ball",
        inductance_quadrature=(4,5,8,2,3),
    )
    labels=[]
    for idx in range(args.samples):
        scene=baseline if idx==0 else make_scene(baseline,rng)
        sources=build_bfzi_source(
            scene, 200_000.0,order=args.source_order,
        )
        # Fixed physical mode topology/order across the whole dataset.
        basis=build_sea_basis(
            scene, modes_per_face=2,face_refinement=1,
        )
        response=solve_seawater_reference(
            sources,basis=basis,config=config,
        )
        snapshot=BFZISnapshotROM.from_operators(
            sources,basis,response.transform,
            response.resistance,response.inductance,
            response.coupling,rank=4,
            inductance_kernel=config.inductance_kernel,
        )
        labels.append(BFZIResponseLabel.from_snapshot(snapshot))
        print(
            f"  geometry {idx+1}/{args.samples}: "
            f"basis={basis.size}, conductivity={scene.seawater.conductivity:.4g}, "
            f"R_rank={len(response.resistance)}, status=provisional",
            flush=True,
        )
    save_response_labels(args.out,labels)
    print(f"Saved {len(labels)} geometry-distinct provisional labels: "
          f"{args.out}",flush=True)
    print("REFERENCE NOT CONVERGED: these records exercise training, "
          "saving, geometry holdouts and inference ONLY.",flush=True)


if __name__=="__main__":
    main()
