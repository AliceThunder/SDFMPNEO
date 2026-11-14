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
           alpha1=None,alpha2=None,core1_mm=None,core2_mm=None,
           shift_mm=(0.,0.,0.),tilt_deg=0.,yaw_deg=0.,
           pitch_deg=0.,box_mm=None,box1_mm=None,box2_mm=None,
           grid_xz=None,grid_y=None,grid_layers=0,grid_factor=.5,
           grid_max_nodes=1400,mesh_only=False,physical_box_only=False,
           sea_void="aabb"):
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
    alphas=(alpha1,alpha2)
    cores=(core1_mm,core2_mm)
    for p,c in enumerate(old):
        core_input=cores[p] if cores[p] is not None else core_mm
        alpha_input=alphas[p] if alphas[p] is not None else alpha
        core=(float(c.rounded_square_core_half_m)
              if core_input is None else float(core_input)*.001)
        a=(float(c.rounded_square_alpha)
           if alpha_input is None else float(alpha_input))
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
    chosen=(box1_mm if box1_mm is not None else box_mm,
            box2_mm if box2_mm is not None else box_mm)
    if any(v is not None for v in chosen):
        cfg=dict(cfg)
        sizes=[]
        for p,v in enumerate(chosen):
            if v is None:
                v=float(sample.get(f"coil{p+1}_outer_package_mm",
                                   sample.get("outer_package_mm",204.)))
            box=float(v)
            if box<2.*(float(coils[p].effective_outer_radius_m)+
                       float(coils[p].wire_radius_m))*1000.:
                raise ValueError("each package must contain its BFZI winding")
            sizes.append(box)
            prefix=f"coil{p+1}_"
            sample[prefix+"outer_package_mm"]=box
            sample[prefix+"package_x_half_mm"]=box/2.
            sample[prefix+"package_z_pos_half_mm"]=box/2.
            sample[prefix+"package_z_neg_half_mm"]=box/2.
        # Original BFZI sea volume has a shared envelope setting;
        # each thermal package below uses its independent sample half-size.
        cfg["BSR_PACKAGE_PLANE_HALF_M"]=max(sizes)*.0005
    if sea_void not in ("aabb","oriented"):
        raise ValueError("original BFZI sea void must be aabb or oriented")
    if sea_void=="oriented":
        physical_box_only=True
    if physical_box_only:
        # Physically isolated sealed package control: no dry riser and no
        # additional world-axis wall extensions. The original BFZI OBB AABB
        # is then exact when both coils are axis aligned.
        cfg=dict(cfg,BSR_PACKAGE_TOP_RISER_ENABLE=False,
                 BSR_PACKAGE_WALL_X_EXTRA_M=0.,
                 BSR_PACKAGE_WALL_NEG_Z_EXTRA_M=0.)
    if (grid_xz is None)!=(grid_y is None):
        raise ValueError("both original BFZI physical grid sizes are required")
    if grid_xz is not None:
        if (min(grid_xz,grid_y,grid_factor)<=0 or grid_layers<0
                or grid_max_nodes<1):
            raise ValueError("positive original BFZI sea grid settings required")
        cfg=dict(cfg,ENV_RESOLVENT_GRID_MODE="physical_cell_size",
                 ENV_RESOLVENT_CELL_SIZE_XZ_M=float(grid_xz),
                 ENV_RESOLVENT_CELL_SIZE_Y_M=float(grid_y),
                 ENV_RESOLVENT_BOUNDARY_LAYER_ENABLE=True,
                 ENV_RESOLVENT_BOUNDARY_LAYER_COUNT=int(grid_layers),
                 ENV_RESOLVENT_BOUNDARY_LAYER_FACTOR=float(grid_factor),
                 ENV_RESOLVENT_MAX_NODES=int(grid_max_nodes))
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
    if sea_void=="oriented":
        from sdfmpneo_bfzi.sea_obb import build as obb_build
        points,vol,volume_meta,records=obb_build(
            fs,records,volume_meta,coils,sample,cfg,
            max_nodes=grid_max_nodes)
        # Original rectangular subbox self quadrature is incompatible
        # with a rotated polyhedral cut; use original volume-sphere mode.
        cfg=dict(cfg,ENV_G_SELF_MODE="equal_volume_sphere")
    if mesh_only:
        return dict(points=points,volumes=vol,records=records,
                    cfg=cfg,vol_meta=volume_meta),dict(
            sea_cells=len(points),sea_grid=volume_meta.get("grid",{}),
            original_sea_cells=volume_meta.get("original_node_count"),
            grid_downsampled=volume_meta.get("downsampled",False))
    if volume_meta.get("downsampled",False):
        raise ValueError("BFZI native sea grid exceeds node budget; "
                         "increase grid cell size to preserve exact FV volumes")
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
    normals=[]
    for fam in families:
        pos=np.asarray(fam.active.pos,float)
        q=pos-pos.mean(axis=0)
        _,eig=np.linalg.eigh(q.T@q)
        n=eig[:,0]
        if n[1]<0:
            n=-n
        normals.append(n)
    angle=float(np.degrees(np.arccos(np.clip(
        abs(np.dot(normals[0],normals[1])),-1.,1.))))
    details=dict(sample=sid,
                 alpha=[float(c.rounded_square_alpha) for c in coils],
                 core_mm=[float(c.rounded_square_core_half_m)*1000
                          for c in coils],
                 coil_center_m=[list(c.center) for c in coils],
                 roll_deg=[float(c.roll_deg) for c in coils],
                 yaw_deg=[float(c.yaw_deg) for c in coils],
                 pitch_deg=[float(c.pitch_deg) for c in coils],
                 winding_normal_xyz=[n.tolist() for n in normals],
                 physical_relative_tilt_deg=angle,
                 box_mm=box_mm,
                 box1_mm=float(sample.get("coil1_outer_package_mm",
                         sample.get("outer_package_mm",204.))),
                 box2_mm=float(sample.get("coil2_outer_package_mm",
                         sample.get("outer_package_mm",204.))),
                 sea_cells=len(points),
                 sea_void_model=sea_void,
                 sea_volume_relative_error=volume_meta["relative_volume_error"],
                 bfzi_prep_sec=time.perf_counter()-t0)
    return data,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--sample",choices=("M1","M2","M3","M4"),default="M4")
    p.add_argument("--alpha",type=float,default=None)
    p.add_argument("--core-mm",type=float,default=None)
    p.add_argument("--alpha1",type=float,default=None)
    p.add_argument("--alpha2",type=float,default=None)
    p.add_argument("--core1-mm",type=float,default=None)
    p.add_argument("--core2-mm",type=float,default=None)
    p.add_argument("--shift-mm",nargs=3,type=float,
                   default=[0.,0.,0.])
    p.add_argument("--tilt-deg",type=float,default=0.,
                   help="Receiver native roll-angle adjustment in degrees")
    p.add_argument("--yaw-deg",type=float,default=0.)
    p.add_argument("--pitch-deg",type=float,default=0.)
    p.add_argument("--box-mm",type=float,default=None)
    p.add_argument("--box1-mm",type=float,default=None)
    p.add_argument("--box2-mm",type=float,default=None)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_param_anchor.npz"))
    p.add_argument("--probe",action="store_true",
                   help="Full physical direct 30s thermal steps for comparison")
    p.add_argument("--voltage",type=float,default=75.)
    p.add_argument("--teacher-out",type=Path,default=None,
                   help="Original BFZI 600s programs for new pose-conditional neural training")
    p.add_argument("--indices",type=int,nargs="+",default=[0,1,4,8,9,20,29])
    p.add_argument("--teacher-step",type=int,default=30,
                   help="Original BFZI direct dynamic teacher grid in seconds")
    p.add_argument("--case-name",type=str,default="P01")
    p.add_argument("--sea-grid-xz",type=float,default=None,
                   help="Original BFZI sea FV target x/z cell size, m")
    p.add_argument("--sea-grid-y",type=float,default=None,
                   help="Original BFZI sea FV target y cell size, m")
    p.add_argument("--sea-grid-layers",type=int,default=0)
    p.add_argument("--sea-grid-factor",type=float,default=.5)
    p.add_argument("--sea-grid-max-nodes",type=int,default=1400)
    p.add_argument("--sea-source-radius",type=float,default=.06,
                   help="Fixed physical source moment radius, m")
    p.add_argument("--sea-source-mask",choices=("legacy","oriented","physical","all"),
                   default="legacy",help="Original global or coil-local weak Joule source support")
    p.add_argument("--sea-fv-faces",choices=("approx","exact"),
                   default="approx",help="Original volume-fraction or exact OBB wetted FV shared faces")
    p.add_argument("--sea-void",choices=("aabb","oriented"),
                   default="aabb",help="Native BFZI AABB or exact oriented sealed-package sea cut")
    p.add_argument("--physical-box-only",action="store_true",
                   help="Disable BFZI riser/extra walls for an isolated sealed-box control")
    p.add_argument("--sea-grid-mesh-only",action="store_true",
                   help="Inspect native BFZI seawater FV grid without EM")
    p.add_argument("--pkg-sea-model",choices=("nearest","face-resistance","face-adjacent","face-normal"),
                   default="nearest",help="Legacy total-G or oriented surface-resistance FV interface")
    p.add_argument("--pkg-sea-h",type=float,default=20.,
                   help="Oriented interface surface coefficient W/(m2 K)")
    p.add_argument("--pkg-sea-quad",type=int,default=2,
                   help="Quadrature points per package surface edge")
    p.add_argument("--pkg-sea-near",type=int,default=4,
                   help="Wet FV cells per package surface quadrature point")
    p.add_argument("--sea-tangent-rank",type=int,default=0,
                   help="Precompute original BFZI sea-Z thermal Jacobian in physical 3D ROM modes")
    p.add_argument("--sea-tangent-fd",choices=("central","forward"),
                   default="forward",
                   help="Central: 2 BFZI solves per mode; forward: 1 using cached cold Z")
    p.add_argument("--sea-tangent-amplitude",type=float,default=.2,
                   help="Maximum absolute sea temperature perturbation in K")
    p.add_argument("--sea-tangent-cold-subspace",action="store_true",
                   help="Use k cold-heat-response directions (k <= 8) instead of k original ROM modes")
    p.add_argument("--horizon",type=float,default=120.)
    p.add_argument("--step",type=float,default=30.)
    p.add_argument("--heat",type=float,default=None)
    p.add_argument("--cool",type=float,default=None)
    a=p.parse_args()
    bf,original=originals(a.bfzi_root)
    data,info=create(
        bf,original,a.sample,alpha=a.alpha,core_mm=a.core_mm,
        alpha1=a.alpha1,alpha2=a.alpha2,
        core1_mm=a.core1_mm,core2_mm=a.core2_mm,
        shift_mm=a.shift_mm,tilt_deg=a.tilt_deg,
        yaw_deg=a.yaw_deg,pitch_deg=a.pitch_deg,
        box_mm=a.box_mm,box1_mm=a.box1_mm,box2_mm=a.box2_mm,
        grid_xz=a.sea_grid_xz,grid_y=a.sea_grid_y,
        grid_layers=a.sea_grid_layers,grid_factor=a.sea_grid_factor,
        grid_max_nodes=a.sea_grid_max_nodes,
        mesh_only=a.sea_grid_mesh_only,
        physical_box_only=a.physical_box_only,
        sea_void=a.sea_void)
    if a.sea_grid_mesh_only:
        print(json.dumps(info,indent=2))
        return
    from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
    from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    t0=time.perf_counter()
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak",fv_faces=a.sea_fv_faces,
                       source_mask=a.sea_source_mask,
                       source_radius=a.sea_source_radius)
    ex=np.diag([.5-1j*em.omega*em.air[0,0],
                .5-1j*em.omega*em.air[1,1]])
    et=BFZIV148Pkg3D(
        em,external=ex,pkg_shape=(4,2,4),pkg_k=.25,
        pkg_sea_model=a.pkg_sea_model,pkg_sea_h=a.pkg_sea_h,
        pkg_sea_quad=a.pkg_sea_quad,pkg_sea_near=a.pkg_sea_near)
    anchor=GeomHeat.from_original(em,et,ex)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    anchor.save(a.out)
    if a.sea_tangent_rank:
        from sdfmpneo_bfzi.geom_tangent import SeaTangent
        from sdfmpneo_bfzi.geom_rom import GeomHeatROM
        temp_rom=GeomHeatROM.build(
            anchor,rank=max(32,a.sea_tangent_rank+5))
        t_tan=time.perf_counter()
        if a.sea_tangent_cold_subspace:
            if a.sea_tangent_fd!="forward" or a.sea_tangent_rank>8:
                p.error("cold heat-response subspace requires forward FD and rank <= 8")
            from sdfmpneo_bfzi.geom_tangent import basis
            from sdfmpneo_bfzi.geom_tangent_subspace import build as cold_build
            modes=basis(anchor.vol,temp_rom.phi[:anchor.ns],rank=8)
            tangent,stats=cold_build(
                em,anchor,temp_rom,modes,a.sea_tangent_rank,
                amplitude=a.sea_tangent_amplitude)
            info["sea_tangent_cold_response"]=stats
        else:
            tangent=SeaTangent.build(em,anchor,temp_rom.phi,
                                      rank=a.sea_tangent_rank,
                                      amplitude=a.sea_tangent_amplitude,
                                      method=a.sea_tangent_fd)
        tangent_path=a.out.with_name(
            a.out.stem+"_sea_tangent.npz")
        tangent.save(tangent_path)
        info["sea_tangent_file"]=str(tangent_path)
        info["sea_tangent_rank"]=len(tangent.jac)
        info["sea_tangent_fd"]=tangent.method
        info["sea_tangent_amplitude_K"]=tangent.amplitude
        info["sea_tangent_directional_solves"]=tangent.nsolve
        info["sea_tangent_build_sec"]=time.perf_counter()-t_tan
    info.update(geometry=anchor.geom.tolist(),
                sea_void_model=a.sea_void,
                sea_fv_faces=a.sea_fv_faces,
                sea_source_mask=a.sea_source_mask,
                sea_source_radius_m=a.sea_source_radius,
                pkg_sea_model=a.pkg_sea_model,
                pkg_sea_h_W_m2K=(a.pkg_sea_h if a.pkg_sea_model=="face-resistance" else None),
                pkg_sea_quad=(a.pkg_sea_quad if a.pkg_sea_model=="face-resistance" else None),
                pkg_sea_near=(a.pkg_sea_near if a.pkg_sea_model=="face-resistance" else None),
                pkg_sea_total_G_W_K=[
                    float(sum(edge[2] for edge in links))
                    for links in et.pkg_links],
                state_count=len(anchor.cap),
                original_cold_Z6=anchor.z0.tolist(),
                pose_extra18=anchor.pose.tolist(),
                oriented_pkg_axes=anchor._pkg_axes.tolist(),
                pkg_half_sizes_m=anchor._pkg_half.tolist(),
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
    if a.teacher_out is not None:
        from examples.bfzi_frac_data import programs,drive,pack_z
        allcases=programs()
        if not a.indices or any(i<0 or i>=len(allcases) for i in a.indices):
            p.error("teacher program indices must be 0–29")
        if a.teacher_step<=0 or 600%a.teacher_step:
            p.error("--teacher-step must divide 600 s")
        times=np.arange(0.,601.,a.teacher_step,dtype=float)
        temps=[];zs=[];vs=[];curr=[];split=[];cases=[]
        teacher_start=time.perf_counter()
        for cid in a.indices:
            volt,win,tag=allcases[cid]
            func=lambda now: drive(volt,win,now)
            trajectory=et.trajectory(times,func)
            temps.append(np.stack([s.temp+et.amb for s in trajectory]))
            zs.append(np.stack([pack_z(s.impedance) for s in trajectory]))
            vs.append([func(now)[0].real for now in times])
            curr.append(np.stack([s.currents for s in trajectory]))
            split.append(tag=="train")
            cases.append(dict(id=cid,sample=a.case_name,
                              voltage_v=volt,windows_s=win,split=tag))
        a.teacher_out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            a.teacher_out,time_s=times,voltage_v=np.asarray(vs),
            temp_K=np.asarray(temps),z6=np.asarray(zs),
            currents_A=np.asarray(curr),
            train=np.asarray(split,bool),nsea=et.count,
            npkg=et.pkg_n,ambient_K=et.amb,external_z=ex,
            sea_xyz_m=em.points,sea_vol_m3=em.vol,
            pkg_xyz_m=np.asarray(et.pkg_xyz),
            pkg_vol_m3=np.asarray(et.pkg_vol),
            sample=np.asarray(a.case_name),
            geometry=anchor.geom,pose=anchor.pose,
            cases_json=json.dumps(cases))
        info["teacher_out"]=str(a.teacher_out)
        info["teacher_cases"]=len(a.indices)
        info["teacher_time_step_s"]=int(a.teacher_step)
        info["original_BFZI_dynamic_seconds"]=time.perf_counter()-teacher_start
    a.out.with_suffix(".json").write_text(
        json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2),flush=True)


if __name__=="__main__":
    main()
