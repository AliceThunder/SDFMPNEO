"""Train or query the *experimental* BFZI geometry-conditional neural path.

No pretrained/certified model or FEM-equivalent accuracy is implied.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from .bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import build_bfzi_source, build_sea_basis
from sdfmpneo_bfzi.sea_neural import (
    BFZINeuralSurrogate, train_response_surrogate,
)


def train(argv=None):
    parser=argparse.ArgumentParser(description="Train a provisional BFZI MLP")
    parser.add_argument("--dataset",default="bfzi_geometry_labels.npz")
    parser.add_argument("--model",default="bfzi_neural.pt")
    parser.add_argument("--epochs",type=int,default=400)
    parser.add_argument("--hidden",type=int,default=96)
    parser.add_argument("--seed",type=int,default=13)
    args=parser.parse_args(argv)
    result=train_response_surrogate(
        args.dataset,args.model,epochs=args.epochs,
        hidden=args.hidden,seed=args.seed,
    )
    print(json.dumps(result,indent=2),flush=True)
    print("Saved PROVISIONAL model:",args.model,flush=True)
    print("Held-out errors are against provisional small-mode labels; "
          "no physical accuracy or cross-domain generalization "
          "has been certified.",flush=True)


def infer(argv=None):
    parser=argparse.ArgumentParser(description="Fast BFZI neural field query")
    parser.add_argument("--model",default="bfzi_neural.pt")
    parser.add_argument(
        "--raw-port-loss",action="store_true",
        help="skip positive local R closure for the fastest but inconsistent output",
    )
    args=parser.parse_args(argv)
    scene=build_demo_scene()
    sources=build_bfzi_source(scene,200_000.0,order=4)
    basis=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    model=BFZINeuralSurrogate.load(args.model)
    pred=model.predict(
        sources,basis,power_consistent=not args.raw_port_loss,
    )
    points=np.array([
        [0.,0.,0.],
        [0.13,0.,0.],
        [0.,0.13,0.],
        [0.,0.,0.22],
    ])
    currents=np.array([1.0+0.2j,-0.4+0.5j])
    print("Predicted seawater port Z (ohms):",
          pred.impedance_seawater,flush=True)
    print("Predicted complete port Z (ohms):",
          pred.port_impedance,flush=True)
    print("Predicted sampled Joule heat (W/m^3):",
          pred.seawater_loss(points,currents),flush=True)
    print("Predicted seawater port power (W):",
          pred.port_seawater_power(currents),flush=True)
    print("Finite conductor powers (W):",
          pred.port_conductor_power(currents),flush=True)
    print("Total port dissipated power (W):",
          pred.port_total_power(currents),flush=True)
    print("Port loss minimum eigenvalue:",
          np.linalg.eigvalsh(pred.impedance_seawater.real)[0],flush=True)
    print("Outside observed whole-feature training range:",
          pred.out_of_training_range,flush=True)
    if pred.power_closure_enabled:
        integrated=pred.integrated_seawater_power(currents)
        from_port=pred.port_seawater_power(currents)
        print("Inference local-R closure nodes:",
              len(pred.closure_points),flush=True)
        print("Port vs spatial Joule quadrature (W):",
              from_port,integrated,flush=True)
        print("Relative port/spatial power mismatch:",
              abs(from_port-integrated)/max(abs(integrated),1e-30),
              flush=True)
        print("Physical current response correction (relative):",
              pred.response_correction_relative,flush=True)
        print("Reciprocal energy Gram imaginary defect:",
              pred.reciprocal_gram_defect,flush=True)
    else:
        print("Uncorrected fast path: learned field and port loss "
              "are INDEPENDENT and need not close.",flush=True)
    print("PROVISIONAL: power closure applies only to the selected "
          "local-R cubature. It does not certify reference-field "
          "accuracy, local Joule resolution or learned MQS reactance.",
          flush=True)


if __name__=="__main__":
    train()
