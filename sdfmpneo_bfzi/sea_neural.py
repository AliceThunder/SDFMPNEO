"""Optional PyTorch BFZI geometry-to-response network.

Experimental supervised learner for *provisional* raw-mode current responses
and passive reciprocal two-port seawater Z. At inference, the optional
positive local-R rule enforces discrete port/spatial Joule-power closure.
It does NOT calculate an MQS reference, certify local quadrature accuracy,
enforce the full electromagnetic weak form, or cover unseen material types.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np

from .sea_dataset import (
    geometry_features, load_response_labels, scene_from_geometry_features,
)
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


def _local_resistance(basis: SeaCurrentBasis, *, orders=(4, 6, 8, 2, 3),
                      chunk_size=768):
    """Positive local Joule Gram; NO inductance/Green assembly or full heat tensor.

    The inexpensive cubature is explicitly an inference *closure rule*;
    its resolution has not been physically certified. It should not be
    confused with the separate highly refined MQS reference R/K rule.
    """
    if (len(orders) != 5
            or any(not isinstance(v, int) or isinstance(v,bool) or v<2
                   for v in orders)):
        raise ValueError("closure orders must be five integers >=2")
    r, p, a, n, t = orders
    x,w=basis.scene.box_shell_quadrature(
        r,p,a, wall_scale=basis.wall_scale,
        shell_normal_order=n,shell_tangential_order=t,
        outer_taper_width=(
            basis.outer_taper_width
            if basis.outer_envelope=="boundary_layer" else None
        ),
    )
    if len(x)==0:
        raise ValueError("closure cubature returned no seawater samples")
    resistance=np.zeros((basis.size,basis.size))
    for start in range(0,len(x),chunk_size):
        end=min(start+chunk_size,len(x))
        modes=basis.evaluate(x[start:end])
        resistance+=np.einsum(
            "n,nic,njc->ij",w[start:end]/basis.scene.seawater.conductivity,
            modes,modes,optimize=True,
        )
    return 0.5*(resistance+resistance.T),x,w


def _power_consistent_response(basis,raw,learned_z,orders):
    """Project learned unit responses onto reciprocal two-port Joule closure.

    For two complex responses U=[u1,u2] in a *real* physical curl basis,
    H=U^H R U is PSD Hermitian. Reciprocity requires Im(H12)=0.
    A minimum one-column correction in the R metric,
        u2 <- u2 - j Im(H12)/H11 * u1,
    imposes that single identity without constructing magnetic L or
    performing a 27/159-mode complex MQS solve. Then H is real symmetric
    up to rounding and supplies Re(Z). The learned reciprocal Im(Z)
    is retained. Accuracy of the adjusted field is NOT certified.
    """
    resistance,points,weights=_local_resistance(basis,orders=orders)
    adjusted=np.array(raw,copy=True)
    gram=adjusted.conj().T @ resistance @ adjusted
    denom=float(np.real(gram[0,0]))
    phase_gap=float(np.imag(gram[0,1]))
    if denom>1e-30:
        adjusted[:,1]-=1j*(phase_gap/denom)*adjusted[:,0]
    elif abs(phase_gap)>1e-14:
        raise ValueError("inconsistent null first-port current in closure")
    fixed=adjusted.conj().T @ resistance @ adjusted
    fixed=0.5*(fixed+fixed.conj().T)
    magnitude=max(float(np.linalg.norm(fixed)),1e-30)
    defect=float(np.linalg.norm(fixed.imag)/magnitude)
    if defect>1e-8:
        raise RuntimeError(
            f"reciprocal Joule closure failed: imaginary Gram defect {defect:g}"
        )
    dissipative=0.5*(fixed.real+fixed.real.T)
    corrected_z=dissipative+1j*learned_z.imag
    shift=float(np.linalg.norm(adjusted-raw)/max(np.linalg.norm(raw),1e-30))
    return adjusted,corrected_z,points,weights,shift,defect


@dataclass(frozen=True)
class BFZINeuralPrediction:
    """Fast spatial J(x) and q(x) after one geometry-conditioned network pass.

    With explicit power_consistent=True the real impedance is reconstructed
    from predicted analytic curl fields on one POSITIVE closure cubature, and
    two-port phase reciprocity is enforced by a minimal field correction.
    Discrete power closure is exact at that rule, not a continuum certificate.
    By default (power_consistent=False), learned fields and Z remain
    independent: the measured held-out fields are not accurate enough to
    justify overwriting the better learned Z with a field-derived resistance.
    Closure is an optional energy-identity diagnostic, NOT an accuracy fix.
    """

    sources: BFZISources
    basis: SeaCurrentBasis
    raw_unit_modes: np.ndarray
    impedance_seawater: np.ndarray
    out_of_training_range: bool
    closure_points: np.ndarray | None = None
    closure_weights: np.ndarray | None = None
    response_correction_relative: float = 0.0
    reciprocal_gram_defect: float = 0.0

    @property
    def power_closure_enabled(self):
        return self.closure_points is not None

    def integrated_seawater_power(self,currents=(1.,0.)):
        """Independent spatial Joule integral on inference closure cubature."""
        if not self.power_closure_enabled:
            raise ValueError("spatial power closure is not enabled")
        return float(
            self.closure_weights @ self.seawater_loss(
                self.closure_points,currents,
            )
        )

    @property
    def port_impedance(self):
        return self.port_impedance_at()

    def port_impedance_at(self, temperatures=None):
        """Air L plus learned sea Z plus finite-conductor internal impedance."""
        return (
            1j*self.sources.omega*self.sources.inductance_air
            + self.impedance_seawater
            + np.diag(self.sources.conductor_internal_impedance(temperatures))
        )

    def port_conductor_power(self,currents=(1.,0.),temperatures=None):
        i=np.asarray(currents,dtype=complex)
        if i.shape!=(2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        return 0.5*np.abs(i)**2*self.sources.conductor_internal_impedance(
            temperatures,
        ).real

    def port_total_power(self,currents=(1.,0.),temperatures=None):
        return float(
            np.sum(self.port_conductor_power(currents,temperatures))
            + self.port_seawater_power(currents)
        )

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

    def predict(self,sources: BFZISources,basis: SeaCurrentBasis, *,
                power_consistent: bool = False,
                closure_orders=(4,6,8,2,3)):
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
        points=weights=None
        shift=defect=0.0
        if power_consistent:
            raw,z,points,weights,shift,defect=_power_consistent_response(
                basis,raw,z,closure_orders,
            )
        return BFZINeuralPrediction(
            sources,basis,raw,z,out_of_training_range,
            points,weights,shift,defect,
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


def evaluate_geometry_holdouts(
    dataset_path,model_path,*,source_order=4,
):
    """Test actual held-out geometries against SAVED provisional MQS labels.

    Rebuilds physical scenes from lossless geometry features; never reruns
    electromagnetic R/L/K or reads training source labels for predictions.
    Compares both original neural Z/J and local-R-closure-adjusted Z/J,
    including probe Joule and full-drive port power. A correct energy
    identity alone is NOT treated as an accuracy improvement.
    """
    from .source import build_bfzi_source
    from .sea_basis import build_sea_basis
    from .sea_snapshot import _scene_signature

    model=BFZINeuralSurrogate.load(model_path)
    samples=load_response_labels(dataset_path)
    heldout=set(model.metadata["validation_signatures"])
    selected=[s for s in samples if s.scene_signature in heldout]
    if not selected or len(selected)!=len(heldout):
        raise ValueError(
            "dataset is not the checkpoint's original geometry holdout"
        )
    if model.metadata["mode_count"]!=27:
        raise ValueError(
            "v1 holdout reconstruction supports the 27-mode basis only; "
            "a higher-mode checkpoint needs its basis specification"
        )
    if not isinstance(source_order,int) or source_order<2:
        raise ValueError("source_order must be integer >=2")

    def relative(predicted,reference):
        return float(
            np.linalg.norm(predicted-reference)/
            max(float(np.linalg.norm(reference)),1e-30)
        )

    metrics=[]
    drive=np.array([1.0+0.2j,-0.4+0.5j])
    for sample in selected:
        scene,frequency=scene_from_geometry_features(
            sample.features,
            max_return_waypoints=model.metadata["max_return_waypoints"],
        )
        source=build_bfzi_source(
            scene,frequency_hz=frequency,order=source_order,
        )
        basis=build_sea_basis(
            scene,modes_per_face=2,face_refinement=1,
        )
        if _scene_signature(source,basis)!=sample.scene_signature:
            raise ValueError(
                "label/source/basis signature mismatch for held-out "
                "scene; check the original --source-order"
            )
        # Completely separate inference calls for raw and corrected paths.
        uncorrected=model.predict(source,basis,power_consistent=False)
        corrected=model.predict(source,basis,power_consistent=True)

        probes=[
            np.asarray(scene.outer_center,dtype=float),
        ]
        for assembly in scene.assemblies:
            half=np.asarray(assembly.enclosure.half_extents)
            normal=int(np.argmin(half))
            tangents=[axis for axis in range(3) if axis!=normal]
            for sign in (-1,1):
                for u,v in ((0.,0.),(.50,.50)):
                    local=np.zeros(3)
                    local[normal]=sign*(
                        half[normal]+max(0.003,3*basis.wall_scale)
                    )
                    local[tangents]=np.array([u,v])*half[tangents]
                    probes.append(assembly.pose.position(local))
        points=np.array(probes,dtype=float)
        if not scene.in_seawater(points).all():
            raise ValueError("held-out probes must be in seawater")
        evaluated=basis.evaluate(points)
        true_j=np.einsum(
            "nmc,m->nc",evaluated,sample.raw_unit_modes @ drive,
            optimize=True,
        )
        true_q=np.sum(np.abs(true_j)**2,axis=1)/(
            2*scene.seawater.conductivity
        )
        true_z=sample.impedance_seawater
        true_power=0.5*float(np.real(np.vdot(
            drive,true_z.real @ drive,
        )))

        record={
            "scene_signature":sample.scene_signature[:12],
            "outside_training_range":corrected.out_of_training_range,
            "raw_mode_relative_error":relative(
                uncorrected.raw_unit_modes,sample.raw_unit_modes,
            ),
        }
        # Both paths use exactly the same independent positive cubature.
        # Report the uncorrected mismatch too; otherwise the tiny closed
        # mismatch can misleadingly look like an accuracy certificate.
        raw_spatial_power=float(
            corrected.closure_weights @ uncorrected.seawater_loss(
                corrected.closure_points,drive,
            )
        )
        raw_port_power=uncorrected.port_seawater_power(drive)
        record["raw_power_mismatch"]=abs(
            raw_spatial_power-raw_port_power
        )/max(abs(raw_port_power),1e-30)
        for name,predicted in (
            ("raw",uncorrected),("closed",corrected),
        ):
            p_z=predicted.impedance_seawater
            p_j=predicted.current_density(points,drive)
            p_q=np.sum(np.abs(p_j)**2,axis=1)/(
                2*scene.seawater.conductivity
            )
            record[f"{name}_z_relative_error"]=relative(p_z,true_z)
            record[f"{name}_j_relative_error"]=relative(p_j,true_j)
            record[f"{name}_probe_q_relative_error"]=relative(p_q,true_q)
            record[f"{name}_port_power_relative_error"]=(
                abs(predicted.port_seawater_power(drive)-true_power)/
                max(abs(true_power),1e-30)
            )
        record["response_correction_relative"]=(
            corrected.response_correction_relative
        )
        record["closed_power_mismatch"]=(
            abs(
                corrected.integrated_seawater_power(drive)
                -corrected.port_seawater_power(drive)
            )/max(abs(corrected.port_seawater_power(drive)),1e-30)
        )
        metrics.append(record)

    errors=[
        "raw_z_relative_error","closed_z_relative_error",
        "raw_j_relative_error","closed_j_relative_error",
        "raw_probe_q_relative_error","closed_probe_q_relative_error",
        "raw_port_power_relative_error","closed_port_power_relative_error",
        "raw_mode_relative_error","raw_power_mismatch",
        "response_correction_relative","closed_power_mismatch",
    ]
    summary={
        "reference_status":"provisional",
        "heldout_geometries":len(metrics),
        "outside_training_range_count":sum(
            int(m["outside_training_range"]) for m in metrics
        ),
        "mean":{key:float(np.mean([m[key] for m in metrics]))
                for key in errors},
        "max":{key:float(np.max([m[key] for m in metrics]))
               for key in errors},
        "per_geometry":metrics,
    }
    return summary
