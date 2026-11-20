"""Generate geometry-diverse fixed-hot-state BFZI labels from the V8 training split."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_param import originals,create
from examples.bfzi_physics_recheck import GRIDS,build
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_canon import make_rom,project
from sdfmpneo_bfzi.geom_free import geometry
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path,rows):
    np.savez_compressed(path,
        name=np.asarray([r["name"] for r in rows]),
        g=np.asarray([r["g"] for r in rows]),
        a=np.asarray([r["a"] for r in rows]),
        sea=np.asarray([r["sea"] for r in rows]),
        total=np.asarray([r["total"] for r in rows]),
        stats=np.asarray([r["stats"] for r in rows]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    p.add_argument("--per-geom",type=int,default=8)
    p.add_argument("--seed",type=int,default=9032)
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_hotstates_multi.npz"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v9_hotstates_multi.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=read(a.plan);mf=read(a.manifest)
    if len(plan["train"])!=48 or len(mf["train"])!=48:
        raise ValueError("V9 geometry-diverse pilot requires the frozen 48-geometry train split")
    byname={Path(s["anchor"].replace("\\","/")).stem.split("_")[2]:s
            for s in mf["train"]}
    if set(byname)!=set(plan["train"]):
        raise ValueError("V8 train plan and manifest geometry IDs differ")
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
    src=SrcFix.load(cfg["source_model"])
    bf,v150=originals(a.bfzi_root)
    rng=np.random.default_rng(a.seed)
    rows=[]
    if a.out.exists():
        with np.load(a.out) as d:
            rows=[dict(name=str(n),g=g,a=h,sea=s,total=z,stats=t)
                for n,g,h,s,z,t in zip(d["name"],d["g"],d["a"],
                                       d["sea"],d["total"],d["stats"])]
    report={"status":"running","per_geometry":a.per_geom,"seed":a.seed,
            "completed":{},"failed":{}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for name in plan["train"]:
        prior=sum(r["name"]==name for r in rows)
        if prior>=a.per_geom:
            report["completed"][name]=prior
            continue
        t0=time.perf_counter()
        try:
            spec=byname[name];case=plan["cases"][name]
            head=GeomHeat.load(spec["anchor"])
            rom=make_rom(head,48)
            parent="p03" if case["donor"]=="g1" else "p07"
            ext=GeomHeat.load(f"runs/bfzi_pose_{parent}_disjoint_anchor.npz").ext
            xz,yy=GRIDS["ultra"]
            data,_=create(bf,v150,"M4",**case["geometry"],sea_void="oriented",
                physical_box_only=True,grid_xz=xz,grid_y=yy,grid_max_nodes=4000)
            th=case["thermal"]
            opts=dict(alpha=float(case["alpha_per_K"]),source_order=5,
                skin=th.get("skin_m",.002),ins_k=th.get("ins_k",.25),
                ins_t=th.get("ins_t"),pkg_k=th.get("pkg_k",.25),
                pkg_sea_h=th.get("pkg_sea_h",20.),sea_max=373.15,
                cu_max=1357.77,pkg_max=None)
            et=build(data,bf,ext,opts)
            init=thermal_mix(atlas,src.apply(case,cold.apply(
                case,atlas.predict(case))),.25)
            g=geometry(case,init)[1]
            base=[]
            for file in spec["teacher"]:
                with np.load(str(file).replace("\\","/")) as d:
                    temp=np.asarray(d["temp_K"],float)
                    if temp.ndim!=2 or temp.shape[1]!=et.n:
                        raise ValueError(f"{name}: teacher temperature shape {temp.shape}, expected (*,{et.n})")
                    base.extend(temp[1:]-et.amb)
            base=np.asarray(base)
            if len(base)<a.per_geom:
                raise ValueError(f"{name}: too few non-ambient teacher states")
            skin_ids=np.arange(et.skin.start,et.skin.stop)
            skin_pkg=np.asarray(et.skin_pkg,int)
            if skin_pkg.shape!=(len(skin_ids),):
                raise ValueError(f"{name}: invalid skin package labels")
            tx=np.zeros(et.n);rx=np.zeros(et.n);wet=np.zeros(et.n)
            tx[skin_ids[skin_pkg==0]]=1.
            rx[skin_ids[skin_pkg==1]]=1.
            wet[:et.count]=1.
            cold_z,*_=et.em(np.zeros(et.n),np.array([1.,0.],complex))
            patterns=((0.,0.,0.),(10.,0.,0.),(0.,10.,0.),
                      (20.,0.,0.),(0.,20.,0.),(12.,-6.,0.),
                      (-6.,12.,0.),(8.,8.,5.))
            for j in range(prior,a.per_geom):
                ix=rng.integers(len(base));iy=rng.integers(len(base))
                w=rng.uniform(.2,.8)
                temp=w*base[ix]+(1.-w)*base[iy]
                dt,dr,dw=patterns[j%len(patterns)]
                temp=temp+dt*tx+dr*rx+dw*wet
                temp=np.clip(temp,0.,80.)
                z,*_=et.em(temp,np.array([1.,0.],complex))
                skin_tx=float(temp[skin_ids[skin_pkg==0]].mean())
                skin_rx=float(temp[skin_ids[skin_pkg==1]].mean())
                stats=np.array([float(temp[:et.count].mean()),
                    skin_tx,skin_rx,float(temp.max())])
                rows.append(dict(name=name,g=g,
                    a=project(rom,temp[np.asarray(head.state_order,int)],rom.phi),
                    sea=pack(z.sea_z-cold_z.sea_z),
                    total=pack(z.total_z-cold_z.total_z),stats=stats))
                if (j+1)%4==0 or j+1==a.per_geom:
                    save(a.out,rows)
                    report["completed"][name]=j+1
                    report["status"]="partial"
                    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
                    print("V9_MULTI",name,j+1,a.per_geom,flush=True)
            report["completed"][name]=a.per_geom
        except Exception as e:
            report["failed"][name]=f"{type(e).__name__}: {e}"
            print("V9_MULTI_FAILED",name,report["failed"][name],flush=True)
        report["seconds"]=round(time.perf_counter()-t0,2)
        a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    save(a.out,rows)
    report["status"]="complete" if not report["failed"] and len(rows)==48*a.per_geom else "partial"
    report["sample_count"]=len(rows)
    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V9_MULTI_DONE",len(rows),"geometries",len(report["completed"]),
          "failed",len(report["failed"]),flush=True)


if __name__=="__main__":
    main()
