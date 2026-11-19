"""Independent six-geometry rank48 thermal/impedance acceptance."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z
from sdfmpneo_bfzi.geom_skintan import restore_rank48


def get_init(case,atlas,ps,mu,src):
    x=atlas.predict(case)
    for model in (ps,mu,src):
        if model is not None:
            x=model.apply(case,x)
    return x


def load_net(path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if int(ck["rank"])!=48:
        raise ValueError(f"{path} is not a rank48 checkpoint")
    net=GeomFracET(ng=int(ck["ng"]),rank=48,
        soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    eck=dict(src_scale=ck["src_scale"],y_scale=ck["y_scale"],
             a_scale=ck["a_scale"],canonical=True,port_sea=True)
    return net,eck


def evaluate(rows,plan,atlas,ps,mu,src,net,ck,tan,shape_mix):
    by_case={}
    all_metrics=[]
    with torch.no_grad():
        for row in rows:
            name=Path(row["file"].replace("\\","/")).stem.split("_")[2]
            case=plan["cases"][name]
            init=get_init(case,atlas,ps,mu,src)
            pred=rollout(case,atlas,net,ck,row["time"],row["voltage"],
                         tan,init_override=init,shape_mix=shape_mix)
            m=metric(pred,row)
            all_metrics.append(m)
            by_case.setdefault(name,[]).append(m)
    def agg(vals):
        return dict(
            histories=len(vals),
            dynamic_z_pct=float(np.mean([x["dynamic_z_rel_L2"] for x in vals])*100),
            current_pct=float(np.mean([x["current_rel_L2"] for x in vals])*100),
            temp_rmse_K=float(np.mean([x["material_mean_rmse_K"] for x in vals])),
            temp_max_K=float(np.mean([x["material_mean_max_K"] for x in vals])),
            budget_W=float(max(x["energy_budget_W"] for x in vals)))
    return dict(overall=agg(all_metrics),
                per_geometry={k:agg(v) for k,v in sorted(by_case.items())})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_cold_dyn6_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_cold_dyn6_manifest.json"))
    p.add_argument("--config",type=Path,default=Path("runs/geomfrac_free_rank48_selection.json"))
    p.add_argument("--thermal-model",type=Path,default=Path("runs/geomfrac_et_pose42_kell_r48_thermal_v1.pt"))
    p.add_argument("--joint-model",type=Path,default=Path("runs/geomfrac_et_pose42_kell_r48_tangent_v1.pt"))
    p.add_argument("--skin-tangent",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v1.pt"))
    p.add_argument("--skin-base",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--v5-report",type=Path,default=Path("runs/geomfrac_cold_dyn6_eval.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_rank48_cold_dyn6_acceptance_v1.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text(encoding="utf-8-sig"))
    mf=json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    cfg=json.loads(a.config.read_text(encoding="utf-8-sig"))
    specs=mf.get("test",[])
    if len(specs)!=6:
        raise ValueError("the frozen independent acceptance must contain six geometries")
    rows=[r for spec in specs for r in prepare(spec,48)]
    names=sorted({Path(r["file"].replace("\\","/")).stem.split("_")[2] for r in rows})
    if len(names)!=6 or set(names)!=set(plan["test"]):
        raise ValueError("holdout geometry identities do not match the frozen plan")
    atlas=GeomInit.load(cfg["atlas"])
    if atlas.rank!=48:
        raise ValueError("rank48 geometry atlas required")
    ps=PortSelf.load(cfg["port_self_model"])
    mu=MutFix.load(cfg["mutual_model"])
    src=SrcFix.load(cfg["source_model"])
    base=load_z(a.skin_base)
    skin=restore_rank48(a.skin_tangent,a.skin_base)
    thermal,ck=load_net(a.thermal_model)
    joint,jck=load_net(a.joint_model)
    result=dict(status="complete",geometry_count=6,history_count=len(rows),
        selected_geometry=names,rank=48,atlas=cfg["atlas"],shape_mix=float(cfg["shape_mix"]),
        v5_reference=json.loads(a.v5_report.read_text(encoding="utf-8-sig"))["results"]["v5"],
        rank48_thermal_base_zfix=evaluate(rows,plan,atlas,ps,mu,src,thermal,ck,base,float(cfg["shape_mix"])),
        rank48_thermal_skin_tangent=evaluate(rows,plan,atlas,ps,mu,src,thermal,ck,skin,float(cfg["shape_mix"])),
        rank48_joint_skin_tangent=evaluate(rows,plan,atlas,ps,mu,src,joint,jck,skin,float(cfg["shape_mix"])),
        decision="independent comparison only; no model selection or hyperparameter tuning on these six geometries")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("R48_DYN6_ACCEPTANCE",json.dumps({k:v["overall"] for k,v in result.items()
        if isinstance(v,dict) and "overall" in v}),flush=True)


if __name__=="__main__":
    main()
