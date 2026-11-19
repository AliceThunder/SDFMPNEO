"""BFZI-free V7 dynamic impedance/current/temperature inference in the core domain.

Input JSON contains {"case":{...},"time_s":[...],"voltage_v":[...]}.
The frozen rank48 thermal model and unified cold V7 head share one rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.geom_cold_domain import check,load as load_domain
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def run(data,config,domain):
    case=data["case"]
    gate=check(case,domain)
    if not gate["accepted"]:
        return dict(status="out_of_domain",gate=gate),None
    t=np.asarray(data["time_s"],float)
    v=np.asarray(data["voltage_v"],float)
    ck=torch.load(config["thermal_model"],map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=int(ck["rank"]),
                   soe_quad=ck.get("soe_quad","trap"),
                   heat_dt=float(ck.get("heat_dt",1.875)),
                   heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    atlas=GeomInit.load(config["atlas"])
    cold=ColdNet.load(config["cold_model"])
    sf=SrcFix.load(config["source_model"])
    tan=restore_rank48(config["skin_model"],config["z_fix_model"])
    init=sf.apply(case,cold.apply(case,atlas.predict(case)))
    pred=rollout(case,atlas,net,ck,t,v,tan,init_override=init,
                 shape_mix=float(config["shape_mix"]))
    summary=dict(status="ok",gate=gate,
                 steps=len(t),duration_s=float(t[-1]),rank=int(ck["rank"]),
                 geometry_only=bool(pred["geometry_only"]),
                 cold_BFZI_solves=int(pred["cold_BFZI_solves"]),
                 heat_basis_builds=int(pred["heat_basis_builds"]),
                 heat_FV_solves=int(pred["heat_FV_solves"]),
                 max_material_K=float(np.max(pred["temp_group_K"])),
                 max_budget_W=float(np.max(abs(pred["material_budget_W"]))))
    return summary,pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v7_infer.npz"))
    p.add_argument("--config",type=Path,default=Path("runs/geomfrac_free_rank48_selection.json"))
    p.add_argument("--domain",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    a=p.parse_args()
    data=json.loads(a.input.read_text(encoding="utf-8-sig"))
    cfg=json.loads(a.config.read_text(encoding="utf-8-sig"))
    cfg=dict(cfg,thermal_model="runs/geomfrac_et_pose42_kell_r48_thermal_v1.pt",
             cold_model="runs/geomfrac_cold_v7.pt",
             skin_model="runs/geomfrac_rank48_skin_tan_v3.pt")
    torch.set_num_threads(2)
    summary,pred=run(data,cfg,load_domain(a.domain))
    if pred is not None:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.out,
            time_s=np.asarray(data["time_s"],float),
            voltage_v=np.asarray(data["voltage_v"],float),
            z6=pred["z6"],
            current_re=pred["current_A"].real,
            current_im=pred["current_A"].imag,
            temp_group_K=pred["temp_group_K"],
            Psea_W=pred["Psea_W"],
            Pcu_W=pred["Pcu_W"],
            material_budget_W=pred["material_budget_W"])
        summary["result"]=str(a.out)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
