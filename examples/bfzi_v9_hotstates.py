"""Generate BFZI V150 fixed-hot-state EM labels on existing training geometries."""
import argparse,json
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

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--per-geom",type=int,default=100)
    p.add_argument("--seed",type=int,default=9031)
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_hotstates.npz"))
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=read("runs/geomfrac_v8_tx_hot_plan.json")
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
    src=SrcFix.load(cfg["source_model"])
    bf,v150=originals(a.bfzi_root)
    rng=np.random.default_rng(a.seed)
    records=[]
    if a.out.exists():
        with np.load(a.out) as d:
            records=[dict(name=str(n),g=g,a=h,sea=s,total=z)
                for n,g,h,s,z in zip(d["name"],d["g"],d["a"],d["sea"],d["total"])]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for name in plan["train"]:
        try:
            c=plan["cases"][name]
            parent="p03" if c["donor"]=="g1" else "p07"
            ext=GeomHeat.load(f"runs/bfzi_pose_{parent}_disjoint_anchor.npz").ext
            xz,yy=GRIDS["ultra"]
            data,_=create(bf,v150,"M4",**c["geometry"],sea_void="oriented",
                          physical_box_only=True,grid_xz=xz,grid_y=yy,
                          grid_max_nodes=4000)
            th=c["thermal"]
            opts=dict(alpha=float(c["alpha_per_K"]),source_order=5,
                skin=th.get("skin_m",.002),ins_k=th.get("ins_k",.25),
                ins_t=th.get("ins_t"),pkg_k=th.get("pkg_k",.25),
                pkg_sea_h=th.get("pkg_sea_h",20.),sea_max=373.15,
                cu_max=1357.77,pkg_max=None)
            et=build(data,bf,ext,opts)
            head=GeomHeat.load(f"runs/bfzi_s5_{name}_ultra_v8txhot_anchor.npz")
            rom=make_rom(head,48)
            order=np.asarray(head.state_order,int)
            init=thermal_mix(atlas,src.apply(c,cold.apply(c,atlas.predict(c))),.25)
            g=geometry(c,init)[1]
            base=[]
            for j in (1,2):
                with np.load(f"runs/bfzi_s5_{name}_ultra_v8txhot_drive{j}.npz") as d:
                    base.extend(np.asarray(d["temp_K"])-et.amb)
            base=np.asarray(base)
            ids=np.arange(et.skin.start,et.skin.stop)
            masks=[]
            for side in (0,1):
                v=np.zeros(et.n)
                v[ids[np.asarray(et.skin_pkg)==side]]=1.
                masks.append(v)
            wet=np.zeros(et.n);wet[:et.count]=1.
            masks.append(wet)
            cold_z,*_=et.em(np.zeros(et.n),np.array([1.,0.],complex))
            done=sum(r["name"]==name for r in records)
            for k in range(done,a.per_geom):
                x=base[rng.integers(len(base))]
                y=base[rng.integers(len(base))]
                w=rng.uniform(.25,.75)
                temp=w*x+(1-w)*y
                temp=temp+rng.uniform(-12.,35.,2)@np.asarray(masks[:2])
                temp=temp+rng.uniform(-3.,8.)*wet
                temp=np.clip(temp,0.,80.)
                z,*_=et.em(temp,np.array([1.,0.],complex))
                records.append(dict(name=name,g=g,a=project(rom,temp[order],rom.phi),
                    sea=pack(z.sea_z-cold_z.sea_z),
                    total=pack(z.total_z-cold_z.total_z)))
                if (k+1)%10==0 or k+1==a.per_geom:
                    np.savez_compressed(a.out,
                        name=np.array([r["name"] for r in records]),
                        g=np.array([r["g"] for r in records]),
                        a=np.array([r["a"] for r in records]),
                        sea=np.array([r["sea"] for r in records]),
                        total=np.array([r["total"] for r in records]))
                    print("V9_HOT",name,k+1,a.per_geom,flush=True)
        except Exception as e:
            print("V9_HOT_FAILED",name,type(e).__name__,str(e),flush=True)
            continue
    print("V9_HOT_DONE",len(records),flush=True)

if __name__=="__main__":
    main()
