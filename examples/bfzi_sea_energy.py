"""Quantify BFZI M4 port/volume Joule discrepancy from the original FV Hodge.

This script is an analytical diagnostic for the exact inherited BFZI solver.
It leaves the electromagnetic impedance and physical |J|^2/sigma heat intact.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter, WeightedFV


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/m4_energy_projection.json"))
    a=p.parse_args()
    sys.path.insert(0,str(a.bfzi_root.resolve()))
    import bfzi_v148_results as calc

    data=calc.build_circle_data("M4")
    model=BFZIV148Adapter(data,calc,sea_only=True)
    sig=np.full(len(model.vol),model.sigma0)
    proj=WeightedFV(model.proj,sig)
    C=list(data["C"])
    X,res=calc.core.solve_operator(
        data,proj,C,op_mode="KP",tol=2e-7,maxiter=180)
    PC=proj.project(C)
    PX=proj.project(X)
    G=[a-b for a,b in zip(C,PC)]
    GX=[a-b for a,b in zip(X,PX)]
    PC2=proj.project(PC)
    PX2=proj.project(PX)
    self_err=max(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))
                 for a,b in zip(PX,PX2))
    source_err=max(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))
                   for a,b in zip(PC,PC2))
    c=np.stack(C,axis=1)
    x=np.stack(X,axis=1)
    g=np.stack(G,axis=1)
    gx=np.stack(GX,axis=1)
    px=np.stack(PX,axis=1)
    w=model.omega*model.mu
    v=model.vol
    weight=(w*w)*sig*v
    m=(w*w)*sig
    z=calc.core.Z_rect(C,X,1j*w*sig*v,model.mu,model.omega)
    z-=calc.core.Z_rect(G,GX,1j*w*sig*v,model.mu,model.omega)
    dz=-1j*w*np.einsum("p,pij->ij",1j*w*sig,model.mass)
    z=calc.core.sym_complex(z+dz)
    base=(
        np.einsum("pic,pjc->pij",c,x,optimize=True)
        -np.einsum("pic,pjc->pij",g,gx,optimize=True))
    base=.5*(base+np.swapaxes(base,1,2))
    mass=(w*w)*np.einsum("p,pij->pij",sig,model.mass)
    rweak=weight[:,None,None]*base.real+mass
    rjoule=weight[:,None,None]*np.einsum(
        "pic,pjc->pij",px.conj(),px,optimize=True).real+mass
    Rport=z.real
    Rweak=rweak.sum(axis=0)
    Rjoule=rjoule.sum(axis=0)
    # Use an actual M4 driven two-port current, including mutual phase.
    air=np.array([[15.827666708487106,5.077003979629416],
                  [5.077003979629416,15.827666708487106]])*1e-6
    zd=z+1j*model.omega*air
    extra=np.diag([.5-1j*model.omega*air[0,0],
                   .5-1j*model.omega*air[1,1]])
    I=np.linalg.solve(zd+extra,np.array([150.,0.],complex))
    qweak=.5*np.real(np.einsum("i,pij,j->p",I.conj(),rweak,I))
    qjoule=.5*np.real(np.einsum("i,pij,j->p",I.conj(),rjoule,I))
    volume=float(qjoule.sum())
    port=float(.5*np.vdot(I,z@I).real)
    weak=float(qweak.sum())
    out={
        "sample":"M4","voltage_peak_V":150.,
        "original_BFZI_zport_ohm":Rport.tolist(),
        "original_FV_joule_matrix_ohm":Rjoule.tolist(),
        "weak_form_loss_matrix_ohm":Rweak.tolist(),
        "loss_matrix_relative_gap":float(np.linalg.norm(Rport-Rjoule)/np.linalg.norm(Rport)),
        "loss_matrix_weak_port_gap":float(np.linalg.norm(Rport-Rweak)/np.linalg.norm(Rport)),
        "projected_field_reprojection_gap":self_err,
        "projected_source_reprojection_gap":source_err,
        "Psea_from_ports_W":port,
        "Psea_from_J_squared_W":volume,
        "Psea_from_local_weak_form_W":weak,
        "port_vs_volume_relative_gap":abs(port-volume)/max(volume,1e-30),
        "weak_form_negative_cell_count":int(np.count_nonzero(qweak<0)),
        "weak_form_min_cell_W":float(qweak.min()),
        "original_gmres":res,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
