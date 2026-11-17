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
    return dict(
        file=row['file'],
        cold_z_rel_L2=float(np.linalg.norm(z[0]-truth[0])/
                            max(np.linalg.norm(truth[0]),1e-15)),
        dynamic_z_rel_L2=float(np.linalg.norm(dz-dr)/
                               max(np.linalg.norm(dr),1e-12)),
        full_z_rel_L2=float(np.linalg.norm(z-truth)/
                            max(np.linalg.norm(truth),1e-12)),
        material_mean_max_K=float(np.max(abs(pred['temp_group_K']-tm))),
        material_mean_rmse_K=float(np.sqrt(np.mean((pred['temp_group_K']-tm)**2))),
        sea_mean_max_K=float(np.max(abs(pred['temp_group_K'][:,0]-tm[:,0]))),
        pkg_mean_max_K=float(np.max(abs(pred['temp_group_K'][:,1:3]-tm[:,1:3]))),
        copper_max_K=float(np.max(abs(pred['temp_group_K'][:,3:]-tm[:,3:]))),
        energy_budget_W=float(np.max(abs(pred['material_budget_W']))))


def evaluate(atlas,net,ck,tan,plan,manifest,split,extra=None,weight=0.,field=None):
    rows=[]
    for spec in manifest[split]:
        name=Path(spec['anchor']).stem.split('_')[2]
        case=plan['cases'][name]
        init=mix_init(case,atlas,extra,weight)
        for hist in prepare(spec,32):
            t0=time.perf_counter()
            pred=rollout(case,atlas,net,ck,hist['time'],hist['voltage'],tan,
                         init_override=init)
            dt=time.perf_counter()-t0
            result=metric(pred,hist)
            if field is not None:
                rom=hist['rom']
                xyz=np.concatenate((rom.xyz,*rom.pkg_xyz),axis=0)
                regs=np.r_[np.zeros(rom.nwet,int),
                    np.repeat([5,6],rom.nskin//2),
                    np.repeat(1,rom.npkg),np.repeat(2,rom.npkg)]
                field_t=field.query(case,xyz,regs,pred['heat_state'],
                                    pred['ambient'],weights=rom.cap[:len(xyz)])
                actual=hist['temp_K'][:,:len(xyz)]
                result['temperature_max_K']=float(np.max(abs(field_t-actual)))
                result['temperature_rmse_K']=float(
                    np.sqrt(np.mean((field_t-actual)**2)))
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
        summary={k+'_mean':float(np.mean([r[k] for r in rows]))
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
    pl=json.loads(args.plan.read_text(encoding='utf-8'))
    mf=json.loads(args.manifest.read_text(encoding='utf-8'))
    res=evaluate(atlas,net,ck,tan,pl,mf,args.split,
                 extra,args.cold_weight,field)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(res,indent=2),encoding='utf-8')
    print('FREE_EVAL_MEANS',[(k,round(v*100 if 'z_rel' in k else v,4))
       for k,v in res['summary'].items()],flush=True)


if __name__=='__main__':
    main()
