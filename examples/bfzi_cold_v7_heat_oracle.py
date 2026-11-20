"""Offline partial teacher-state substitution in frozen V7 tangent (six unseen geometries)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_free import circuit,geometry,rollout
from sdfmpneo_bfzi.geom_memory import geom_port
from sdfmpneo_bfzi.geom_heat import pack
from sdfmpneo_bfzi.geom_neural import GeomFracET

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cold',type=Path,default=Path('runs/geomfrac_cold_v7.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_cold_v7_heat_oracle.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    root=Path("runs")
    cfg=json.loads((root/"geomfrac_free_rank48_selection.json").read_text(encoding="utf-8-sig"))
    plan=json.loads((root/"geomfrac_cold_v7_dyn_plan.json").read_text(encoding="utf-8-sig"))
    mf=json.loads((root/"geomfrac_s5_cold_v7_core_manifest.json").read_text(encoding="utf-8-sig"))
    ck=torch.load(root/"geomfrac_et_pose42_kell_r48_thermal_v1.pt",map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=int(ck["rank"]),
        soe_quad=ck.get("soe_quad","trap"),heat_dt=float(ck.get("heat_dt",1.875)),heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load(a.cold)
    sf=SrcFix.load(cfg["source_model"])
    tan=restore_rank48(root/"geomfrac_rank48_skin_tan_v3.pt",cfg["z_fix_model"])
    ways=("pred","true_mean","true_skin","true_shape","true_all")
    err={k:[] for k in ways}
    cases={}
    for spec in mf["test"]:
        name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        case=plan["cases"][name]
        init=sf.apply(case,cold.apply(case,atlas.predict(case)))
        geo,feat=geometry(case,init)
        port=geom_port(geo)
        g=torch.as_tensor(feat[None],dtype=torch.float32)
        scale=np.sqrt(init["cap"][labels(48)])
        cases[name]=[]
        for row in prepare(spec,48):
            pred=rollout(case,atlas,net,ck,row["time"],row["voltage"],tan,
                init_override=init,shape_mix=float(cfg["shape_mix"]))
            ref=row["z6"];dr=ref-ref[0]
            den=max(float(np.linalg.norm(dr)),1e-12)
            ph=np.asarray(pred["heat_state"])
            th=np.asarray(row["a"])
            variants={"pred":ph,"true_mean":np.column_stack((th[:,:5],ph[:,5:])),
              "true_skin":np.column_stack((ph[:,:32],th[:,32:])),
              "true_shape":np.column_stack((ph[:,:5],th[:,5:])),
              "true_all":th}
            vres={}
            for method,hist in variants.items():
                zz=[]
                for k,tm in enumerate(row["time"]):
                    s=hist[k]*scale
                    z,*_=circuit(geo,init,port,s,tan,g,scale,
                       float(row["voltage"][k]),293.15,tm=float(tm),
                       heat_voltage=float(row["voltage"][k-1]) if k else 0.)
                    zz.append(pack(z))
                zz=np.asarray(zz)
                e=float(np.linalg.norm((zz-zz[0])-dr)/den)
                err[method].append(e)
                vres[method]=e
                if method=="pred":
                    mismatch=float(np.linalg.norm(zz-pred["z6"])/max(np.linalg.norm(pred["z6"]),1e-12))
                    if mismatch>1e-8:raise RuntimeError(f"circuit recalc differs {name} {mismatch}")
            cases[name].append(vres)
        print("THERM_ORACLE_COMPONENT",name,cases[name],flush=True)
    avg={k:float(np.mean(v)) for k,v in err.items()}
    print("THERM_ORACLE_SUMMARY",avg,flush=True)
    a.out.write_text(
        json.dumps(dict(methods=ways,summary=avg,cases=cases),indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
