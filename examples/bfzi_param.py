"""Build continuous coil geometry using only original BFZI V150 primitives.

The coil curve, FV sea, near-field Green, projection and terminal leads are
all built by BFZI itself. Optional parameters change the native CoilSpec.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np


def originals(root):
    sys.path.insert(0,str(Path(root).resolve()))
    import bfzi_v148_results as bf
    import run_v150 as v150
    return bf,v150


def create(bf,original,sid="M4",*,alpha=None,core_mm=None,
           shift_mm=(0.,0.,0.),tilt_deg=0.,yaw_deg=0.,
           pitch_deg=0.,box_mm=None):
    """Rebuild the genuine BFZI field operators for modified native coils."""
    sid=sid.upper()
    if sid=="M4":
        sample=bf.common_circle_sample("M4",center_distance_mm=40.)
        d=.040
        coils=[bf.circle_coil((0.,-d/2,0.)),
               bf.circle_coil((0., d/2,0.))]
        cfg=bf.circle_cfg()
        pitch=.005
        for p in (1,2):
            sample.update({
                f"coil{p}_coil_outer_mm":180.,
                f"coil{p}_turns":10,
                f"coil{p}_outer_package_mm":204.,
                f"coil{p}_package_x_half_mm":102.,
                f"coil{p}_package_z_pos_half_mm":102.,
                f"coil{p}_package_z_neg_half_mm":102.,
                f"coil{p}_aout_mm":88.85,
                f"coil{p}_r0_mm":88.85})
    elif sid in ("M1","M2","M3"):
        base_id=int(bf.base.SAMPLES[sid]["base_id"])
        sample,coils,cfg=bf.core.build_coils_and_cfg(
            base_id,bf.base.common_overrides())
        sample=dict(sample)
        coils=bf.base.rotate_coils_x_to_z(coils)
        pitch=.006
    else:
        raise ValueError("two-coil original BFZI samples are M1/M2/M3/M4")
    old=[dataclasses.replace(c) for c in coils]
    for p,c in enumerate(old):
        core=(float(c.rounded_square_core_half_m)
              if core_mm is None else float(core_mm)*.001)
        a=(float(c.rounded_square_alpha)
           if alpha is None else float(alpha))
        outer=float(c.effective_outer_radius_m)
        r0=outer-core
        inner_r=r0-a*c.turns*pitch
        inner_chi=core-(1-a)*c.turns*pitch
        if not (.01<=a<=1. or a==1.) or core<0 or r0<=0:
            raise ValueError("invalid BFZI rounded-square alpha/core")
        if inner_r<=0 or inner_chi< -1e-10:
            raise ValueError("rounded-square core/corner exhausted by turns")
        shift=np.asarray(shift_mm,float)*.001 if p==1 else np.zeros(3)
        if shift.shape!=(3,):
            raise ValueError("three relative translation coordinates required")
        center=np.asarray(c.center,float)+shift
        angle=(float(c.roll_deg)+float(tilt_deg) if p==1
               else float(c.roll_deg))
        family=("degenerated_circular_roundedsquare"
                if abs(core)<1e-12 and abs(a-1.)<1e-12
                else "standard_rounded_square")
        coils[p]=dataclasses.replace(
            c,center=tuple(center),rounded_square_alpha=a,
            rounded_square_core_half_m=core,
            inner_radius_m=inner_r,
            rounded_corner_radius_m=r0,
            shape_family=family,roll_deg=angle,
            yaw_deg=float(c.yaw_deg)+(float(yaw_deg) if p==1 else 0.),
            pitch_deg=float(c.pitch_deg)+(float(pitch_deg) if p==1 else 0.))
    if box_mm is not None:
        box=float(box_mm)
        if box<=0 or box<2.*max(
            float(c.effective_outer_radius_m+c.wire_radius_m)
            for c in coils)*1000:
            raise ValueError("package must contain BFZI winding")
        cfg=dict(cfg)
        cfg["BSR_PACKAGE_PLANE_HALF_M"]=box*.0005
        for p in (1,2):
            sample[f"coil{p}_outer_package_mm"]=box
            sample[f"coil{p}_package_x_half_mm"]=box/2.
            sample[f"coil{p}_package_z_pos_half_mm"]=box/2.
            sample[f"coil{p}_package_z_neg_half_mm"]=box/2.
    # Each operator is called directly from original BFZI; no alternate
    # geometry generator, approximation to the source, or FEM mesh.
    t0=time.perf_counter()
    fs=bf.fs
    families=[bf._strict_family(c,cfg) for c in coils]
    mu=float(fs.PhysicalConstants().mu)
    omega=float(original.OMEGA) if hasattr(original,"OMEGA") else (
        2.*np.pi*float(sample["frequency_hz"]))
    sigma=float(bf.SIGMA_SPM)
    points,vol,volume_meta,records=(
        fs._build_v33_rectangular_sea_volume_cells_integrated(coils,cfg))
    G0,gmeta=bf._build_G0_volume_matrix_near_integrated(
        points,vol,records or [],cfg=cfg,near_layers=1)
    C=[bf.base.source_to_volume_records_centerline_family_zlead(
        records or [],fam,cfg)[0] for fam in families]
    data=dict(points=points,volumes=vol,records=records,
              vw=(1j*omega*mu*sigma)*np.asarray(vol,float),
              C=C,G0=np.asarray(G0,complex),
              mu=mu,omega=omega,sample=sample,coils=coils,
              families=families,cfg=cfg,cfg_overrides=cfg,
              vol_meta=volume_meta,
              gmeta=dict(gmeta,geometry_family="BFZI_native_continuous",
                         prep_sec=time.perf_counter()-t0))
    details=dict(sample=sid,
                 alpha=[float(c.rounded_square_alpha) for c in coils],
                 core_mm=[float(c.rounded_square_core_half_m)*1000
                          for c in coils],
                 coil_center_m=[list(c.center) for c in coils],
                 roll_deg=[float(c.roll_deg) for c in coils],
                 yaw_deg=[float(c.yaw_deg) for c in coils],
                 pitch_deg=[float(c.pitch_deg) for c in coils],
                 box_mm=box_mm,
                 sea_cells=len(points),
                 bfzi_prep_sec=time.perf_counter()-t0)
    return data,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--sample",choices=("M1","M2","M3","M4"),default="M4")
    p.add_argument("--alpha",type=float,default=None)
    p.add_argument("--core-mm",type=float,default=None)
    p.add_argument("--shift-mm",nargs=3,type=float,
                   default=[0.,0.,0.])
    p.add_argument("--tilt-deg",type=float,default=0.,
                   help="Receiver native roll-angle adjustment in degrees")
    p.add_argument("--yaw-deg",type=float,default=0.)
    p.add_argument("--pitch-deg",type=float,default=0.)
    p.add_argument("--box-mm",type=float,default=None)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_param_anchor.npz"))
    p.add_argument("--probe",action="store_true",
                   help="Full physical direct 30s thermal steps for comparison")
    p.add_argument("--voltage",type=float,default=75.)
    p.add_argument("--horizon",type=float,default=120.)
    p.add_argument("--step",type=float,default=30.)
    p.add_argument("--heat",type=float,default=None)
    p.add_argument("--cool",type=float,default=None)
    a=p.parse_args()
    bf,original=originals(a.bfzi_root)
    data,info=create(
        bf,original,a.sample,alpha=a.alpha,core_mm=a.core_mm,
        shift_mm=a.shift_mm,tilt_deg=a.tilt_deg,
        yaw_deg=a.yaw_deg,pitch_deg=a.pitch_deg,box_mm=a.box_mm)
    from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
    from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    t0=time.perf_counter()
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak")
    ex=np.diag([.5-1j*em.omega*em.air[0,0],
                .5-1j*em.omega*em.air[1,1]])
    et=BFZIV148Pkg3D(em,external=ex,pkg_shape=(4,2,4),pkg_k=.25)
    anchor=GeomHeat.from_original(em,et,ex)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    anchor.save(a.out)
    info.update(geometry=anchor.geom.tolist(),
                state_count=len(anchor.cap),
                original_cold_Z6=anchor.z0.tolist(),
                anchor_build_sec=time.perf_counter()-t0)
    if a.probe:
        if a.step<=0 or a.horizon<=0:
            p.error("positive horizon and time step required")
        t=np.arange(0.,a.horizon+a.step/2,a.step)
        v=np.full(len(t),float(a.voltage))
        if a.heat is not None or a.cool is not None:
            if a.heat is None or a.cool is None:
                p.error("--heat and --cool must be specified together")
            v=np.where((t<=a.heat)|(t>a.heat+a.cool),
                       a.voltage,0.)
        def drive(now):
            j=int(np.argmin(abs(t-now)))
            return np.array([v[j],0.],complex)
        seq=et.trajectory(t,drive)
        state=np.asarray([s.temp for s in seq])
        z=np.asarray([[
            s.impedance[0,0].real,s.impedance[0,0].imag,
            s.impedance[0,1].real,s.impedance[0,1].imag,
            s.impedance[1,1].real,s.impedance[1,1].imag] for s in seq])
        info.update(probe_points=len(t),
                    max_temperature_K=float(state.max()+et.amb),
                    final_Z6=z[-1].tolist())
        np.savez_compressed(a.out.with_name(a.out.stem+"_probe.npz"),
                            time_s=t,voltage_v=v,
                            temp_K=state+et.amb,z6=z)
    a.out.with_suffix(".json").write_text(
        json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2),flush=True)


if __name__=="__main__":
    main()
