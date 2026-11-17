"""Evaluate BFZI-free cold-start transients against original S5 physics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit,mix_init
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_rom import groups
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def metric(pred,row):
    truth=row['z6']
    actual=row['temp_K']
    cap=row['rom'].cap
    tm=np.column_stack([(actual[:,sl]*cap[sl]).sum(axis=1)/cap[sl].sum()
                        for sl in groups(row['rom'])])
    z=pred['z6']
    dz=z-z[0]
    dr=truth-truth[0]
    with np.load(row['file'].split('#')[0]) as raw:
        if 'currents_A' in raw:
            itrue=np.asarray(raw['currents_A'])
        else:
            from sdfmpneo_bfzi.frac_memory import zmat
            itrue=np.array([np.linalg.solve(zmat(zz)+row['rom'].ext,
                        np.array([vv,0.],complex))
                        for zz,vv in zip(truth,row['voltage'])])
        psea=np.asarray(raw['Psea_W']) if 'Psea_W' in raw else None
        pcu=np.asarray(raw['Pcu_W']) if 'Pcu_W' in raw else None
    if itrue.shape!=pred['current_A'].shape:
        raise ValueError('physical matched-current shape mismatch')
    names=('sea','pkg_tx','pkg_rx','cu_tx','cu_rx')
    mdiff=pred['temp_group_K']-tm
    material={name+'_mean_rmse_K':float(np.sqrt(np.mean(mdiff[:,j]**2)))
              for j,name in enumerate(names)}
    material.update({name+'_mean_max_K':float(np.max(abs(mdiff[:,j])))
              for j,name in enumerate(names)})
    zden=max(float(np.linalg.norm(dr)),1e-12)
    parts=('R11','X11','R12','X12','R22','X22')
    impedance={name+'_dynamic_error_L2_over_total_delta':
               float(np.linalg.norm((dz-dr)[:,j])/zden)
               for j,name in enumerate(parts)}
    powers={}
    if psea is not None:
        powers['sea_power_rel_L2']=float(
            np.linalg.norm(pred['Psea_W']-psea)/
            max(np.linalg.norm(psea),1e-12))
    if pcu is not None:
        powers['copper_power_rel_L2']=float(
            np.linalg.norm(pred['Pcu_W']-pcu)/
            max(np.linalg.norm(pcu),1e-12))
    return dict(**material,**impedance,**powers,
        file=row['file'],
        cold_z_rel_L2=float(np.linalg.norm(z[0]-truth[0])/
                            max(np.linalg.norm(truth[0]),1e-15)),
        dynamic_z_rel_L2=float(np.linalg.norm(dz-dr)/
                               max(np.linalg.norm(dr),1e-12)),
        full_z_rel_L2=float(np.linalg.norm(z-truth)/
                            max(np.linalg.norm(truth),1e-12)),
        current_rel_L2=float(np.linalg.norm(pred['current_A']-itrue)/
                             max(np.linalg.norm(itrue),1e-12)),
        current_max_abs_A=float(np.max(abs(pred['current_A']-itrue))),
        material_mean_max_K=float(np.max(abs(pred['temp_group_K']-tm))),
        material_mean_rmse_K=float(np.sqrt(np.mean((pred['temp_group_K']-tm)**2))),
        sea_mean_max_K=float(np.max(abs(pred['temp_group_K'][:,0]-tm[:,0]))),
        pkg_mean_max_K=float(np.max(abs(pred['temp_group_K'][:,1:3]-tm[:,1:3]))),
        copper_max_K=float(np.max(abs(pred['temp_group_K'][:,3:]-tm[:,3:]))),
        energy_budget_W=float(np.max(abs(pred['material_budget_W']))))


def evaluate(atlas,net,ck,tan,plan,manifest,split,extra=None,weight=0.,field=None,dyn=None,skin=None,port_self=None,mutual=None,source=None,shape_mix=0.):
    rows=[]
    for spec in manifest[split]:
        name=Path(spec['anchor']).stem.split('_')[2]
        case=plan['cases'][name]
        tprep=time.perf_counter()
        init=mix_init(case,atlas,extra,weight)
        if port_self is not None:
            init=port_self.apply(case,init)
        if mutual is not None:
            init=mutual.apply(case,init)
        if source is not None:
            init=source.apply(case,init)
        prep_s=time.perf_counter()-tprep
        for hi,hist in enumerate(prepare(spec,32)):
            t0=time.perf_counter()
            pred=rollout(case,atlas,net,ck,hist['time'],hist['voltage'],tan,
                         init_override=init,extra=dyn,shape_mix=shape_mix)
            dt=time.perf_counter()-t0
            result=metric(pred,hist)
            t2=time.perf_counter()
            rep=rollout(case,atlas,net,ck,hist['time'],hist['voltage'],tan,
                        init_override=init,extra=dyn,shape_mix=shape_mix)
            result['repeat_s']=time.perf_counter()-t2
            if not np.allclose(rep['z6'],pred['z6'],atol=1e-10,rtol=1e-10):
                raise RuntimeError(f'{name}: repeated rollout differs')
            result['geometry_prepare_s']=prep_s if hi==0 else 0.
            result['first_new_geometry_s']=prep_s+dt if hi==0 else None
            if field is not None:
                tf=time.perf_counter()
                rom=hist['rom']
                xyz=np.concatenate((rom.xyz,*rom.pkg_xyz),axis=0)
                regs=np.r_[np.zeros(rom.nwet,int),
                    np.repeat([5,6],rom.nskin//2),
                    np.repeat(1,rom.npkg),np.repeat(2,rom.npkg)]
                sea_skin=None
                if skin is not None:
                    from sdfmpneo_bfzi.geom_free import geometry
                    geom=np.broadcast_to(geometry(case,init)[1],
                                         (len(hist['time']),skin.ng)).copy()
                    with torch.no_grad():
                        sea_skin=skin(
                            torch.as_tensor(geom,dtype=torch.float32),
                            torch.as_tensor(pred['heat_state'],dtype=torch.float32),
                            torch.as_tensor(hist['time'][:,None],dtype=torch.float32),
                            torch.as_tensor(np.r_[0.,hist['voltage'][:-1]][:,None],
                                            dtype=torch.float32)).numpy()
                field_t=field.query(case,xyz,regs,pred['heat_state'],
                        pred['ambient'],weights=rom.cap[:len(xyz)],skin=sea_skin)
                actual=hist['temp_K'][:,:len(xyz)]
                result['temperature_max_K']=float(np.max(abs(field_t-actual)))
                diff=field_t-actual
                result['temperature_rmse_K']=float(
                    np.sqrt(np.mean(diff**2)))
                for label,mask in (
                    ('sea_wet',regs==0),
                    ('sea_skin',np.isin(regs,[5,6])),
                    ('package_tx',regs==1),
                    ('package_rx',regs==2)):
                    if np.any(mask):
                        result[label+'_rmse_K']=float(
                            np.sqrt(np.mean(diff[:,mask]**2)))
                        result[label+'_max_K']=float(np.max(abs(diff[:,mask])))
                result['field_decode_s']=time.perf_counter()-tf
                result['field_query_points']=int(len(xyz))
                if hi==0:
                    result['first_new_geometry_full_s']=(
                        prep_s+dt+result['field_decode_s'])
            result['seconds']=dt
            result['geometry']=name
            rows.append(result)
        print('FREE_EVAL',name,[(r['file'].split('_')[-1],
            round(r['dynamic_z_rel_L2']*100,2),
            round(r['material_mean_max_K'],3))
            for r in rows if r['geometry']==name],flush=True)
    return dict(
        geometry_count=len(manifest[split]),history_count=len(rows),
        rows=rows,
        summary={k+'_mean':float(np.mean([r[k] for r in rows if r.get(k) is not None]))
          for k in rows[0] if k not in ('file','geometry')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',choices=('train','val','test'),default='val')
    p.add_argument('--atlas',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--cold-atlas',type=Path,default=None)
    p.add_argument('--cold-weight',type=float,default=0.)
    p.add_argument('--field-atlas',type=Path,default=None)
    p.add_argument('--dyn-model',type=Path,default=None)
    p.add_argument('--skin-model',type=Path,default=None)
    p.add_argument('--port-self-model',type=Path,default=None)
    p.add_argument('--mutual-model',type=Path,default=None)
    p.add_argument('--source-model',type=Path,default=None)
    p.add_argument('--shape-mix',type=float,default=0.)
    p.add_argument('--model',type=Path,default=Path('runs/geomfrac_et_v4_8g_quality.pt'))
    p.add_argument('--z-fix-model',type=Path,default=Path('runs/geomfrac_et_v9_ztune.pt'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    atlas=GeomInit.load(args.atlas)
    extra=GeomInit.load(args.cold_atlas) if args.cold_atlas else None
    tan=load_z(args.z_fix_model)
    field=None
    if args.field_atlas:
        from sdfmpneo_bfzi.geom_field import FieldAtlas
        field=FieldAtlas.load(args.field_atlas)
    dyn=None
    if args.dyn_model:
        from sdfmpneo_bfzi.geom_dyn import restore
        dyn=restore(args.dyn_model)
    skin=None
    if args.skin_model:
        from sdfmpneo_bfzi.geom_skin import restore as load_skin
        skin=load_skin(args.skin_model)
    port_self=None
    if args.port_self_model:
        from sdfmpneo_bfzi.geom_portself import PortSelf
        port_self=PortSelf.load(args.port_self_model)
    mutual=None
    if args.mutual_model:
        from sdfmpneo_bfzi.geom_mutual import MutFix
        mutual=MutFix.load(args.mutual_model)
    source=None
    if args.source_model:
        from sdfmpneo_bfzi.geom_srcfix import SrcFix
        source=SrcFix.load(args.source_model)
    pl=json.loads(args.plan.read_text(encoding='utf-8'))
    mf=json.loads(args.manifest.read_text(encoding='utf-8'))
    res=evaluate(atlas,net,ck,tan,pl,mf,args.split,
                 extra,args.cold_weight,field,dyn,skin,port_self,mutual,source,
                 args.shape_mix)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(res,indent=2),encoding='utf-8')
    print('FREE_EVAL_MEANS',[(k,round(v*100 if 'z_rel' in k else v,4))
       for k,v in res['summary'].items()],flush=True)


if __name__=='__main__':
    main()
