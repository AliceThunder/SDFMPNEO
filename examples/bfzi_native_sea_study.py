"""Preview native original BFZI V150 near-package seawater FV refinement.

All mesh variants use the *original* BFZI volume-integrated sea builder.
Preview performs no Green/GMRES solves, no neural training, no true time data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.pkg_interface import face_links
from sdfmpneo_bfzi.geom_void import report as void_report


def values(head, wet):
    """Distance and face-conductance diagnostics on genuine BFZI wet cells."""
    rows=[]
    for p in range(2):
        mid=head._pkg_mid[p]
        half=head._pkg_half[p]
        rot=head._pkg_axes[p]
        loc=(wet-mid)@rot
        gap=np.linalg.norm(np.maximum(abs(loc)-half,0.),axis=1)
        inside=int(np.count_nonzero(np.all(abs(loc)<half-1e-12,axis=1)))
        links=face_links(wet,head.pkg_xyz[p],half,rot,
                         (4,2,4),0,.25,.6,20.,quad=2,near=4)
        ids=np.unique([k for k,_,_ in links])
        rows.append(dict(package=p+1,wet_inside_box=inside,
                         wet_min_gap_mm=float(gap.min()*1000),
                         wet_p10_gap_mm=float(np.percentile(gap,10)*1000),
                         linked_wet_gap_median_mm=float(
                             np.median(gap[ids])*1000),
                         linked_wet_gap_max_mm=float(
                             np.max(gap[ids])*1000),
                         interface_G_W_K=float(sum(g for _,_,g in links)),
                         links=len(links),linked_wet_cells=len(ids)))
    return rows


def build_args(meta):
    c=np.asarray(meta["coil_center_m"],float)
    shift=(c[1]-c[0])*1000-np.array([0.,40.,0.])
    return dict(
        alpha1=meta["alpha"][0],alpha2=meta["alpha"][1],
        core1_mm=meta["core_mm"][0],core2_mm=meta["core_mm"][1],
        shift_mm=shift,tilt_deg=meta["roll_deg"][1]-meta["roll_deg"][0],
        yaw_deg=meta["yaw_deg"][1]-meta["yaw_deg"][0],
        pitch_deg=meta["pitch_deg"][1]-meta["pitch_deg"][0],
        box1_mm=meta["box1_mm"],box2_mm=meta["box2_mm"])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfzi-root",type=Path,required=True)
    ap.add_argument("--runs",type=Path,default=Path("runs"))
    ap.add_argument("--out",type=Path,
                    default=Path("runs/bfzi_native_sea_preview.json"))
    a=ap.parse_args()
    bf,v150=originals(a.bfzi_root)
    grids=[
        ("legacy",None,None,0),
        ("face_coarse",.20,.15,0),
        ("face_medium",.14,.12,0),
        ("face_fine",.11,.09,0),
        ("layer_coarse",.22,.17,1),
    ]
    out=dict(method="native original BFZI V150 volume FV preview",
             grids={},geometries={},void_geometry={})
    for name in ("P03","P07"):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        head=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        cases={}
        for label,xz,y,layers in grids:
            d,info=create(bf,v150,"M4",**build_args(meta),
                          grid_xz=xz,grid_y=y,grid_layers=layers,
                          mesh_only=True)
            m=d["vol_meta"]
            diag=dict(grid_xz_m=xz,grid_y_m=y,layers=layers,
                      sea_cells=len(d["points"]),
                      original_cells=m["original_node_count"],
                      downsampled=m.get("downsampled",False),
                      rel_volume_error=m["relative_volume_error"],
                      near_package=values(head,d["points"]))
            cases[label]=diag
            if label=="legacy":
                out["void_geometry"][name]=void_report(head,m)
            print(name,label,"cells",diag["sea_cells"],"/",
                  diag["original_cells"],"downsampled",
                  diag["downsampled"],"median_gap_mm",
                  [round(v["linked_wet_gap_median_mm"],2)
                   for v in diag["near_package"]],flush=True)
        out["geometries"][name]=cases
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
