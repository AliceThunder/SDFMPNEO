"""End-to-end original-BFZI cold field / shared-ROM causal spatial response."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .frac_memory import zmat

def source_at(em, pts):
    """Original BFZI V150 coil and z-lead source at arbitrary physical XYZ."""
    q=np.asarray(pts,float)
    if q.ndim!=2 or q.shape[1]!=3:
        raise ValueError("XYZ query must have shape (n,3)")
    cfg=em.base["cfg"]
    calc=em.calc
    src=[]
    for fam in em.base["families"]:
        geom,_=calc.zlead_line_geom_for_coil(fam.coil,cfg)
        c=calc.fs.air_projection_to_points_centerline_family_quadratic(
            q,fam,point_chunk=512)
        if len(geom.pos):
            c+=calc.base._air_projection_to_points(
                q,np.asarray(geom.pos,float),np.asarray(geom.dl,float),
                chunk_points=512,chunk_line=1024)
        src.append(c)
    return np.stack(src,axis=1)

def material_xyz(em):
    """Original BFZI coil-attached 4x2x4 package and eight skin-layer XYZ."""
    from .pose import axes
    dims=(4,2,4)
    sample=em.base["sample"]
    cfg=em.base["cfg"]
    frac=1.4**np.arange(8)
    dx=.002*frac/frac.sum()
    centers=[];skin=[];copper=[]
    for p,fam in enumerate(em.base["families"]):
        prefix=f"coil{p+1}_"
        half=np.array([
            float(sample.get(prefix+"package_x_half_mm",102.))*.001,
            float(cfg.get("BSR_PACKAGE_NORMAL_HALF_M",.0055)),
            max(float(sample.get(prefix+"package_z_pos_half_mm",102.)),
                float(sample.get(prefix+"package_z_neg_half_mm",102.)))*.001])
        mid=np.asarray(fam.coil.center,float)
        R=axes(fam.coil)
        step=2*half/np.array(dims)
        cells=[]
        for t in np.ndindex(*dims):
            cell=mid+R@(-half+(np.asarray(t)+.5)*step)
            cells.append(cell)
            for axis in range(3):
                for side,sign in ((0,-1),(dims[axis]-1,1)):
                    if t[axis]!=side:
                        continue
                    face=cell+sign*.5*step[axis]*R[:,axis]
                    normal=sign*R[:,axis]
                    x=0.
                    for w in dx:
                        skin.append(face+(x+.5*w)*normal)
                        x+=w
        centers.append(cells)
        copper.append(mid)
    return (np.asarray(centers),np.asarray(copper),np.asarray(skin))


def complete(em,cold,model,pred,voltage,*,xyz=None):
    """FV q is exact-to-port by normalization; no dynamic BFZI EM calls.

    The near-source point query uses original BFZI C(q) and the cold native
    FV J, with conductivity scaling and source-covariance recovery.
    """
    ns=len(em.vol)
    times=np.asarray(pred["time_s"],float)
    volt=np.asarray(voltage,float)
    temp=np.asarray(pred["temp_K"],float)
    cur=np.asarray(pred["current_A"],complex)
    if temp.shape!=(len(times),ns+64+2+1024):
        raise ValueError("target geometry thermal coordinates mismatch")
    if volt.shape!=times.shape or cur.shape!=(len(times),2):
        raise ValueError("BFZI circuit history shape mismatch")
    sigma=em.sigma0*(1+em.alpha*(temp[:,:ns]-em.ambient))
    if np.any(sigma<=0):
        raise ValueError("nonpositive BFZI seawater conductivity")
    factor=sigma/em.sigma0
    # Native BFZI original FV J, with local temperature conductivity gain.
    J=np.einsum("npc,tp->tnc",cold.unit_j,cur,optimize=True)
    J*=factor[:,:,None]
    raw=.5*np.real(np.einsum(
        "ti,nij,tj->tn",np.conj(cur),cold.weak_q,cur,optimize=True))
    raw=np.maximum(raw*factor,0.)
    norm=raw@em.vol
    psea=np.asarray(pred["Psea_W"],float)
    q=np.zeros_like(raw)
    live=norm>1e-20
    q[live]=raw[live]*(psea[live]/norm[live])[:,None]
    qpow=q@em.vol
    gap=float(np.max(abs(qpow-psea)))
    if np.any((~live)&(psea>1e-9)):
        raise ValueError("positive BFZI seawater power without spatial q")
    # Circuit power balance: voltage, external loads, copper and water.
    ex=model.ext
    z=np.asarray(pred["z6"],float)
    Z=zmat(z)
    pin=.5*np.real(volt*np.conj(cur[:,0]))
    pext=.5*np.real(np.einsum(
        "ti,ij,tj->t",np.conj(cur),ex,cur,optimize=True))
    pc=np.asarray(pred["Pcu_W"],float).sum(axis=1)
    circuit_gap=pin-pext-psea-pc
    pout=.5*max(float(ex[1,1].real),0.)*abs(cur[:,1])**2
    eta=np.divide(pout,pin,out=np.zeros_like(pout),where=pin>1e-15)
    pkg_xyz,cu_xyz,skin_xyz=material_xyz(em)
    if len(skin_xyz)!=1024:
        raise ValueError("BFZI physical source-skin topology mismatch")
    field=dict(temp_K=temp,z6=z,current_A=cur,eta=eta,
               time_s=times,voltage_v=volt,sea_xyz=em.points,
               pkg_xyz=pkg_xyz,copper_xyz=cu_xyz,skin_xyz=skin_xyz,
               sea_vol=em.vol,Jsea_A_m2=J,qsea_W_m3=q,
               Psea_W=psea,Pcu_W=pred["Pcu_W"],
               Pinput_W=pin,Pexternal_W=pext,
               Pload_W=pout,
               port_power_gap_W=circuit_gap,
               q_power_gap_W=qpow-psea,
               thermal_budget_W=pred["total_budget_W"])
    if xyz is not None and len(xyz):
        pts=np.asarray(xyz,float).reshape(-1,3)
        # Native volume-averaged J from the cold projector, plus original
        # source C(q) variation in the containing FV temperature.
        tree=cKDTree(em.points)
        cell=tree.query(pts)[1]
        native=source_at(em,pts)
        base=np.stack(em.base["C"],axis=1)[cell]
        # The correction has zero FV mean in the native quadrature limit.
        add=np.einsum("qpc,tp->tqc",native-base,cur,optimize=True)
        Jq=J[:,cell]-1j*(em.omega*em.mu)*sigma[:,cell,None]*add
        qq=.5*np.sum(abs(Jq)**2,axis=2)/sigma[:,cell]
        field.update(query_xyz=pts,query_J_A_m2=Jq,query_q_W_m3=qq,
                     query_T_K=temp[:,cell])
    return field

def metrics(field):
    qgap=float(np.max(abs(field["q_power_gap_W"])))
    pgap=float(np.max(abs(field["port_power_gap_W"])))
    watts=np.asarray(field["Psea_W"])+np.sum(field["Pcu_W"],axis=1)
    active=watts>1e-8
    pmax=max(qgap,pgap)
    tr=np.abs(field["thermal_budget_W"])
    return dict(
        q_power_max_abs_W=qgap,port_power_max_abs_W=pgap,
        thermal_budget_max_abs_W=float(tr.max()),
        max_rel_port_power=float(np.max(
            abs(field["port_power_gap_W"][active])/watts[active]))
            if np.any(active) else 0.,
        max_rel_q_power=float(np.max(
            abs(field["q_power_gap_W"][active])/watts[active]))
            if np.any(active) else 0.,
        max_rel_thermal_power=float(np.max(
            tr[active]/watts[active])) if np.any(active) else 0.,
        max_abs_shutdown_budget_W=float(np.max(tr[~active]))
            if np.any(~active) else 0.)
