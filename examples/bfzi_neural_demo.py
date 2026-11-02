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
    args=parser.parse_args(argv)
    scene=build_demo_scene()
    sources=build_bfzi_source(scene,200_000.0,order=4)
    basis=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    model=BFZINeuralSurrogate.load(args.model)
    pred=model.predict(sources,basis)
    points=np.array([
        [0.,0.,0.],
        [0.13,0.,0.],
        [0.,0.13,0.],
        [0.,0.,0.22],
    ])
    currents=np.array([1.0+0.2j,-0.4+0.5j])
    print("Predicted seawater port Z (ohms):",
          pred.impedance_seawater,flush=True)
    print("Predicted sampled Joule heat (W/m^3):",
          pred.seawater_loss(points,currents),flush=True)
    print("Predicted seawater port power (W):",
          pred.port_seawater_power(currents),flush=True)
    print("Port loss minimum eigenvalue:",
          np.linalg.eigvalsh(pred.impedance_seawater.real)[0],flush=True)
    print("Outside observed varying-feature range:",
          pred.out_of_training_range,flush=True)
    print("PROVISIONAL model: port passivity is parameterized, "
          "but independently learned J/Z need not satisfy exact "
          "volume-to-port Joule closure.",flush=True)


if __name__=="__main__":
    train()
