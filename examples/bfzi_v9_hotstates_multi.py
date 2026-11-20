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
        idx=np.asarray([r["idx"] for r in rows],int),
        g=np.asarray([r["g"] for r in rows]),
        a=np.asarray([r["a"] for r in rows]),
        sea=np.asarray([r["sea"] for r in rows]),
        total=np.asarray([r["total"] for r in rows]),
        stats=np.asarray([r["stats"] for r in rows]),
        scale=np.asarray([r.get("scale",1.) for r in rows],float))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    p.add_argument("--per-geom",type=int,default=16)
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
            names=np.asarray(d["name"]).astype(str)
            if "idx" in d.files:
                indices=np.asarray(d["idx"],int)
            else:
                seen={}
                indices=[]
                for name in names:
                    indices.append(seen.get(name,0));seen[name]=seen.get(name,0)+1
                indices=np.asarray(indices,int)
            scales=(np.asarray(d["scale"],float) if "scale" in d.files
                    else np.ones(len(names)))
            rows=[dict(name=str(n),idx=int(i),g=g,a=h,sea=s,total=z,stats=t,scale=float(q))
                for n,i,g,h,s,z,t,q in zip(names,indices,d["g"],d["a"],
                                         d["sea"],d["total"],d["stats"],scales)]
    if a.report.exists():
        report=read(a.report)
    else:
        report={"completed":{},"failed":{}}
    report.update(status="running",per_geometry=a.per_geom,seed=a.seed)
    report.setdefault("completed",{});report.setdefault("failed",{})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for name in plan["train"]:
        done={r["idx"] for r in rows if r["name"]==name}
        missing=[j for j in range(a.per_geom) if j not in done]
        if not missing:
            report["completed"][name]=len(done)
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
            if report["failed"].get(name,{}).get("build"):
                report["failed"][name].pop("build",None)
                if not report["failed"][name]:
                    report["failed"].pop(name,None)
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
            low_pool=np.flatnonzero(np.max(base,axis=1)<=20.)
            if not len(low_pool):
                low_pool=np.arange(len(base))
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
                (20.,0.,0.),(0.,20.,0.),(12.,-6.,0.),(-6.,12.,0.),
                (8.,8.,5.),(40.,0.,20.),(0.,40.,20.),(0.,0.,30.),
                (40.,-10.,20.),(-10.,40.,20.),(20.,20.,25.),
                (60.,0.,30.),(0.,60.,30.))
            for j in missing:
                try:
                    dt,dr,dw=patterns[j%len(patterns)]
                    pool=low_pool if j>=8 else np.arange(len(base))
                    ok=False;temp=None;z=None;fac=1.
                    for attempt in range(8):
                        ix=rng.choice(pool);iy=rng.choice(pool)
                        w=rng.uniform(.2,.8)
                        raw=w*base[ix]+(1.-w)*base[iy]
                        delta=dt*tx+dr*rx+dw*wet
                        for fac in (1.,.8,.6,.4,.2,0.):
                            cand=np.clip(raw+fac*delta,0.,80.)
                            try:
                                et.check_domain(cand)
                                z,*_=et.em(cand,np.array([1.,0.],complex))
                                temp=cand;ok=True;break
                            except ValueError:
                                continue
                        if ok:break
                    if not ok:
                        raise ValueError("could not draw an in-domain temperature state after 8 attempts")
                    stats=np.array([float(temp[:et.count].mean()),
                        float(temp[skin_ids[skin_pkg==0]].mean()),
                        float(temp[skin_ids[skin_pkg==1]].mean()),
                        float(temp.max())])
                    rows.append(dict(name=name,idx=j,g=g,scale=fac,
                        a=project(rom,temp[np.asarray(head.state_order,int)],rom.phi),
                        sea=pack(z.sea_z-cold_z.sea_z),
                        total=pack(z.total_z-cold_z.total_z),stats=stats))
                    done.add(j)
                    report["failed"].get(name,{}).pop(str(j),None)
                    if not report["failed"].get(name):
                        report["failed"].pop(name,None)
                    if len(done)%4==0 or j==missing[-1]:
                        save(a.out,rows)
                    print("V9_MULTI",name,j+1,a.per_geom,flush=True)
                except Exception as e:
                    msg=f"{type(e).__name__}: {e}"
                    report["failed"].setdefault(name,{})[str(j)]=msg
                    print("V9_MULTI_SAMPLE_FAILED",name,j+1,msg,flush=True)
                    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
                    continue
            report["completed"][name]=len(done)
        except Exception as e:
            msg=f"{type(e).__name__}: {e}"
            report["failed"][name]={"build":msg}
            print("V9_MULTI_GEOM_FAILED",name,msg,flush=True)
        report["seconds"]=round(time.perf_counter()-t0,2)
        report["sample_count"]=len(rows)
        report["status"]="partial"
        a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    save(a.out,rows)
    report["sample_count"]=len(rows)
    report["status"]="complete" if len(rows)==48*a.per_geom and not report["failed"] else "partial"
    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V9_MULTI_DONE",len(rows),"geometries",len(report["completed"]),
          "failed",len(report["failed"]),flush=True)


if __name__=="__main__":
    main()
