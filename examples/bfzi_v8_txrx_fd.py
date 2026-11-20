"""BFZI V150 fixed-thermal-state susceptibility data, training geometries only."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from examples.bfzi_param import originals,create
from examples.bfzi_physics_recheck import GRIDS,build
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_canon import make_rom,mode_scale,project
from sdfmpneo_bfzi.geom_free import geometry
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_srcfix import SrcFix

torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(r"D:\work\geomfrac-s5-BFZI")
pl=read("runs/geomfrac_v8_tx_hot_plan.json")
cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
atlas=GeomInit.load(cfg["atlas"])
cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
src=SrcFix.load(cfg["source_model"])
bf,v150=originals(root)
gx=[];ax=[];ys=[];tags=[];modes=[];rows=[]
for name in pl["train"]:
 start=time.perf_counter()
 c=pl["cases"][name]
 parent="p03" if c["donor"]=="g1" else "p07"
 ext=GeomHeat.load(f"runs/bfzi_pose_{parent}_disjoint_anchor.npz").ext
 xz,y=GRIDS["ultra"]
 data,_=create(bf,v150,"M4",**c["geometry"],sea_void="oriented",
   physical_box_only=True,grid_xz=xz,grid_y=y,grid_max_nodes=4000)
 mat=c["thermal"]
 opts=dict(alpha=float(c["alpha_per_K"]),source_order=5,
   skin=mat.get("skin_m",.002),ins_k=mat.get("ins_k",.25),
   ins_t=mat.get("ins_t"),pkg_k=mat.get("pkg_k",.25),
   pkg_sea_h=mat.get("pkg_sea_h",20.),
   sea_max=373.15,cu_max=1357.77,pkg_max=None)
 et=build(data,bf,ext,opts)
 head=GeomHeat.load(f"runs/bfzi_s5_{name}_ultra_v8txhot_anchor.npz")
 rom=make_rom(head,48)
 order=np.asarray(head.state_order,int)
 init=thermal_mix(atlas,src.apply(c,cold.apply(c,atlas.predict(c))),.25)
 g=geometry(c,init)[1]
 with np.load(f"runs/bfzi_s5_{name}_ultra_v8txhot_drive2.npz") as d:
  theta=d["temp_K"][-1]-et.amb
  ref=d["z6"][-1]
 def calc(t):
  z,*_=et.em(t,np.array([1.,0.],complex))
  return z
 cold_z=calc(np.zeros(et.n))
 hot_z=calc(theta)
 err=float(np.linalg.norm(pack(hot_z.total_z)-ref))
 if err>1e-7:raise RuntimeError(f"cold model mismatch {name} {err}")
 def add(t,z,label):
  gx.append(g.copy());ax.append(project(rom,t[order],rom.phi))
  ys.append(pack(z.sea_z-cold_z.sea_z))
  tags.append(name);modes.append(label)
 add(theta,hot_z,"warm")
 fields={}
 for side in (0,1):
  d=np.zeros(et.n)
  skinids=np.arange(et.skin.start,et.skin.stop)
  d[skinids[np.asarray(et.skin_pkg)==side]]=1.
  fields["skin_"+("tx" if side==0 else "rx")]=d
 fields["wet_all"]=np.r_[np.ones(et.count),np.zeros(et.n-et.count)]
 scale=mode_scale(rom,48)
 for m in (32,40):
  d=np.zeros(et.n);d[order]=rom.phi[:,m]*scale[m]
  fields["mode_"+str(m)]=d/max(float(np.max(abs(d))),1e-12)
 for label,delta in fields.items():
  amp=2.
  pp=calc(theta+amp*delta)
  mm=calc(theta-amp*delta)
  add(theta+amp*delta,pp,label+"_plus")
  add(theta-amp*delta,mm,label+"_minus")
  jac=pack((pp.sea_z-mm.sea_z)/(2*amp))
  rec=dict(name=name,mode=label,step_K=amp,
    dR11_ohm_K=float(jac[0]),dX11_ohm_K=float(jac[1]),
    dR22_ohm_K=float(jac[4]),dX22_ohm_K=float(jac[5]))
  rows.append(rec)
  print("PHYS_FD",json.dumps(rec),flush=True)
 print("PHYS_GEOM_DONE",name,round(time.perf_counter()-start,1),flush=True)
 # Save complete earlier geometry samples before the next expensive cold geometry.
 np.savez_compressed("runs/geomfrac_v8_txrx_fd_train.npz",
   g=np.asarray(gx),a=np.asarray(ax),sea=np.asarray(ys),
   geometry=np.asarray(tags),mode=np.asarray(modes))
 Path("runs/geomfrac_v8_txrx_fd_train.json").write_text(
  json.dumps(dict(status="partial",geometries=len(set(tags)),
    records=len(gx),physics="BFZI V150 S5 ultra fixed temperature EM",
    finite_difference=rows),indent=2)+"\n",encoding="utf-8")
Path("runs/geomfrac_v8_txrx_fd_train.json").write_text(
 json.dumps(dict(status="complete",geometries=len(set(tags)),
  records=len(gx),physics="BFZI V150 S5 ultra fixed temperature EM",
  finite_difference=rows),indent=2)+"\n",encoding="utf-8")
print("PHYS_FD_DONE",len(set(tags)),len(gx),len(rows),flush=True)
