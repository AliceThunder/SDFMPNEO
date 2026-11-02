"""Optional PyTorch BFZI geometry-to-response network.

Experimental supervised learner for *provisional* raw-mode current responses
and passive reciprocal two-port seawater Z. It does NOT calculate an MQS
reference, certify data, guarantee local Joule/port loss closure, or represent
unseen material types beyond the feature schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np

from .sea_dataset import geometry_features, load_response_labels
from .sea_basis import SeaCurrentBasis
from .source import BFZISources


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "BFZI neural training requires PyTorch: pip install -e '.[neural]'"
        ) from exc
    return torch, nn


def _network(in_dim, out_dim, hidden):
    _, nn = _torch()
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def _encode_targets(labels):
    """Raw basis currents + Cholesky real Z + reciprocal imaginary Z."""
    targets = []
    for sample in labels:
        z = sample.impedance_seawater
        hermitian = 0.5*(z.real+z.real.T)
        eigenvalues = np.linalg.eigvalsh(hermitian)
        if eigenvalues[0] < -1e-10:
            raise ValueError("nonpassive training sample")
        tolerance = max(1e-15, 1e-10*max(eigenvalues[-1], 0.0))
        chol = np.linalg.cholesky(hermitian + tolerance*np.eye(2))
        target = np.r_[
            sample.raw_unit_modes.real.ravel(),
            sample.raw_unit_modes.imag.ravel(),
            chol[0,0], chol[1,0], chol[1,1],
            z.imag[0,0], z.imag[0,1], z.imag[1,1],
        ]
        targets.append(target)
    return np.stack(targets)


def _decode_target(vector, n_modes):
    n = n_modes*2
    raw = vector[:n].reshape(n_modes,2) + (
        1j*vector[n:2*n].reshape(n_modes,2)
    )
    lo, cross, hi, x, y, z = vector[2*n:2*n+6]
    cholesky = np.array([[lo, 0.0],[cross,hi]])
    re = cholesky @ cholesky.T
    im = np.array([[x,y],[y,z]])
    return raw, re + 1j*im


@dataclass(frozen=True)
class BFZINeuralPrediction:
    """Fast spatial J(x) and q(x) after one geometry-conditioned network pass.

    Impedance real part is PSD and imaginary part reciprocal by construction.
    Unlike a fully physical MQS solve, its learned J and Z are predicted
    separately: their *integrated* Joule losses need not close exactly.
    """

    basis: SeaCurrentBasis
    raw_unit_modes: np.ndarray
    impedance_seawater: np.ndarray
    out_of_training_range: bool

    def current_density(self, xyz, currents=(1.,0.)):
        i = np.asarray(currents,dtype=complex)
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        x = np.asarray(xyz,dtype=float)
        if x.shape[-1:] != (3,):
            raise ValueError("positions must have trailing dimension 3")
        modes = self.basis.evaluate(x.reshape(-1,3))
        field = np.einsum(
            "nmc,m->nc",modes,self.raw_unit_modes @ i,optimize=True,
        )
        return field.reshape(x.shape)

    def seawater_loss(self, xyz, currents=(1.,0.)):
        j=self.current_density(xyz,currents)
        return np.sum(np.abs(j)**2,axis=-1)/(
            2*self.basis.scene.seawater.conductivity
        )

    def port_seawater_power(self, currents=(1.,0.)):
        i=np.asarray(currents,dtype=complex)
        if i.shape!=(2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        return 0.5*float(np.real(np.vdot(
            i,self.impedance_seawater.real @ i,
        )))


class BFZINeuralSurrogate:
    """Portable, inference-only wrapper. Model uses train-only normalization."""

    def __init__(self, model, metadata):
        self._model=model.eval()
        self.metadata=metadata

    @classmethod
    def load(cls,path):
        torch,_=_torch()
        record=torch.load(path,map_location="cpu",weights_only=True)
        if (record.get("schema_version")!=1
                or record.get("reference_status")!="provisional"
                or record.get("mode_count",0)<1):
            raise ValueError("unsupported or untrusted BFZI neural model")
        model=_network(
            record["in_dim"],record["out_dim"],record["hidden"],
        )
        model.load_state_dict(record["state_dict"])
        model.eval()
        return cls(model,record)

    def predict(self,sources: BFZISources,basis: SeaCurrentBasis):
        torch,_=_torch()
        if basis.scene is not sources.scene:
            raise ValueError("query basis does not belong to source scene")
        meta=self.metadata
        if basis.size!=meta["mode_count"]:
            raise ValueError("query mode count differs from training topology")
        features=geometry_features(
            sources,max_return_waypoints=meta["max_return_waypoints"],
        )
        if len(features)!=meta["in_dim"]:
            raise ValueError("query geometry feature dimension mismatch")
        xmin=meta["train_min"].cpu().numpy()
        xmax=meta["train_max"].cpu().numpy()
        width=xmax-xmin
        moving=width>1e-12
        mean=meta["x_mean"].cpu().numpy()
        # Even a coordinate fixed in all training samples (e.g. coil
        # winding or transmitter pose) can change at inference.
        stationary_outside = (~moving) & (
            np.abs(features-mean) > 1e-8*np.maximum(np.abs(mean),1.0)
        )
        moving_outside = moving & (
            (features < xmin-1e-6*width)
            | (features > xmax+1e-6*width)
        )
        out_of_training_range=bool(np.any(
            stationary_outside | moving_outside
        ))
        scale=meta["x_scale"].cpu().numpy()
        x=(features-mean)/scale
        with torch.no_grad():
            pred=self._model(
                torch.tensor(x,dtype=torch.float32).unsqueeze(0)
            ).squeeze(0).cpu().numpy()
        output=pred*meta["y_scale"].cpu().numpy()+(
            meta["y_mean"].cpu().numpy()
        )
        raw,z=_decode_target(output,meta["mode_count"])
        if not np.isfinite(raw).all() or not np.isfinite(z).all():
            raise FloatingPointError("BFZI network returned nonfinite response")
        return BFZINeuralPrediction(
            basis,raw,z,out_of_training_range,
        )


def train_response_surrogate(
    dataset_path,model_path,*,epochs=400,hidden=96,
    learning_rate=1e-3,seed=13,validation_fraction=0.25,
    max_return_waypoints=8,
):
    """Train on independent geometry records, never split spatial points.

    Provisional labels are accepted only for a FUNCTIONAL pipeline smoke
    experiment. The validation errors measure agreement with these
    provisional Galerkin labels, not continuum electromagnetic accuracy.
    """
    torch,nn=_torch()
    labels=load_response_labels(dataset_path)
    n=len(labels)
    if n<4:
        raise ValueError("need >=4 independently solved geometries")
    if (not isinstance(epochs,int) or epochs<1
            or not isinstance(hidden,int) or hidden<4
            or not np.isfinite(learning_rate) or learning_rate<=0
            or not 0<validation_fraction<0.5):
        raise ValueError("invalid BFZI training hyperparameters")
    if any(s.reference_status!="provisional" for s in labels):
        raise ValueError("this experimental trainer expects provisional labels")
    x=np.stack([s.features for s in labels])
    y=_encode_targets(labels)
    mode_count=labels[0].raw_unit_modes.shape[0]
    permutation=np.random.default_rng(seed).permutation(n)
    nval=max(1,int(round(n*validation_fraction)))
    train_idx,val_idx=permutation[nval:],permutation[:nval]
    x_train,y_train=x[train_idx],y[train_idx]
    x_mean=x_train.mean(axis=0)
    x_scale=x_train.std(axis=0)
    x_scale=np.where(x_scale>1e-12,x_scale,1.0)
    y_mean=y_train.mean(axis=0)
    y_scale=y_train.std(axis=0)
    y_scale=np.maximum(y_scale, np.maximum(1e-9,1e-3*np.abs(y_mean)))
    xx=(x-x_mean)/x_scale
    yy=(y-y_mean)/y_scale
    torch.manual_seed(seed)
    model=_network(x.shape[1],y.shape[1],hidden)
    xx=torch.as_tensor(xx,dtype=torch.float32)
    yy=torch.as_tensor(yy,dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    lowest=float("inf")
    best=None
    best_epoch=0
    patience=max(30,min(120,epochs//4))
    for epoch in range(epochs):
        model.train()
        prediction=model(xx[train_idx])
        loss=nn.functional.mse_loss(prediction,yy[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid=nn.functional.mse_loss(
                model(xx[val_idx]),yy[val_idx],
            ).item()
        if valid<lowest:
            lowest=valid
            best=copy.deepcopy(model.state_dict())
            best_epoch=epoch+1
        elif epoch+1-best_epoch >= patience:
            break
    model.load_state_dict(best)
    record={
        "schema_version":1,
        "reference_status":"provisional",
        "in_dim":int(x.shape[1]),
        "out_dim":int(y.shape[1]),
        "hidden":int(hidden),
        "mode_count":int(mode_count),
        "max_return_waypoints":int(max_return_waypoints),
        "x_mean":torch.tensor(x_mean,dtype=torch.float32),
        "x_scale":torch.tensor(x_scale,dtype=torch.float32),
        "y_mean":torch.tensor(y_mean,dtype=torch.float32),
        "y_scale":torch.tensor(y_scale,dtype=torch.float32),
        "train_min":torch.tensor(x_train.min(axis=0),dtype=torch.float32),
        "train_max":torch.tensor(x_train.max(axis=0),dtype=torch.float32),
        "state_dict":best,
        "validation_signatures":[labels[i].scene_signature for i in val_idx],
        "training_count":int(len(train_idx)),
        "validation_count":int(len(val_idx)),
    }
    torch.save(record,model_path)
    # Report errors in the PHYSICAL outputs, not just normalized MSE.
    model.eval()
    with torch.no_grad():
        result=model(xx[val_idx]).cpu().numpy()
    result=result*y_scale+y_mean
    err_current=[]
    err_z=[]
    for row,j in zip(result,val_idx):
        estimate,z=_decode_target(row,mode_count)
        truth=labels[j]
        err_current.append(float(
            np.linalg.norm(estimate-truth.raw_unit_modes)/
            max(np.linalg.norm(truth.raw_unit_modes),1e-30)
        ))
        err_z.append(float(
            np.linalg.norm(z-truth.impedance_seawater)/
            max(np.linalg.norm(truth.impedance_seawater),1e-30)
        ))
    summary={
        "train_geometries":int(len(train_idx)),
        "heldout_geometries":int(len(val_idx)),
        "best_epoch":int(best_epoch),
        "normalized_val_mse":float(lowest),
        "heldout_raw_mode_relative_error":float(np.mean(err_current)),
        "heldout_port_z_relative_error":float(np.mean(err_z)),
    }
    return summary
