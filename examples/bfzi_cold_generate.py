"""Resumable BFZI V150 cold-only reference production.

Each geometry is solved in an isolated process. Failures are logged and the
remaining candidate geometries continue; completed results survive restarts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from examples.bfzi_cold_v7_replacements import apply as apply_replacements


def _save(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def _pack(z):
    a=np.asarray(z,complex)
    return np.array([a[0,0].real,a[0,0].imag,
                     a[0,1].real,a[0,1].imag,
                     a[1,1].real,a[1,1].imag])


def one(bfzi_root,geom,path,grid='ultra',source_order=5):
    t_all=time.perf_counter()
    t=time.perf_counter()
    from examples.bfzi_param import originals,create
    from examples.bfzi_physics_recheck import GRIDS
    from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
    from examples.bfzi_cold_lead_guard import gap
    import_sec=time.perf_counter()-t

    t=time.perf_counter()
    bf,v150=originals(bfzi_root)
    orig_sec=time.perf_counter()-t
    t=time.perf_counter()
    spacing=gap(bf,geom)
    if not all(p['clear'] for p in spacing):
        raise ValueError('native BFZI terminal leads overlap: '
                         +str([p['gap_mm'] for p in spacing]))
    if grid not in GRIDS:
        raise ValueError(f'unknown BFZI grid {grid}')
    xz,y=GRIDS[grid]
    gap_sec=time.perf_counter()-t

    t=time.perf_counter()
    data,info=create(bf,v150,'M4',**geom,sea_void='oriented',
                     physical_box_only=True,grid_xz=xz,grid_y=y,
                     grid_max_nodes=4000)
    create_sec=time.perf_counter()-t
    t=time.perf_counter()
    em=BFZIV148Adapter(data,bf,heat_mode='port-weak',
                      fv_faces='exact',source_mask='subcell',
                      source_order=source_order)
    adapter_sec=time.perf_counter()-t
    amb=293.15
    t=time.perf_counter()
    ans=em.solve(np.full(len(em.vol),amb),np.full(2,amb))
    solve_sec=time.perf_counter()-t
    air=np.asarray(em.air,float)
    z0=_pack(ans.total_z)
    sea=_pack(ans.sea_z)
    cu=np.asarray(ans.copper_z,complex)
    if (not np.isfinite(z0).all() or not np.isfinite(sea).all()
            or not np.isfinite(air).all()
            or np.linalg.norm(z0)<=0):
        raise RuntimeError('nonfinite cold port response')
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # NPZ is committed only after the entire cold solve succeeds.
    t=time.perf_counter()
    tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,z0=z0,sea=sea,air=air,
                        copper=np.stack((cu.real,cu.imag),axis=-1),
                        omega=float(em.omega),sigma=float(em.sigma0),
                        cells=len(em.vol),ambient=amb)
    tmp.replace(path)
    save_sec=time.perf_counter()-t
    profile=dict(import_sec=import_sec,originals_sec=orig_sec,
                 lead_check_sec=gap_sec,create_sec=create_sec,
                 create_report_sec=float(info.get('bfzi_prep_sec',0.)),
                 create_detail=dict(info.get('prep_profile',{})),
                 adapter_sec=adapter_sec,adapter_detail=dict(em.profile),
                 solve_sec=solve_sec,save_sec=save_sec,
                 total_sec=time.perf_counter()-t_all,
                 cells=len(em.vol),green_shape=list(data['G0'].shape),
                 gmres=ans.residual)
    print('COLD_DONE',path.name,'cells',len(em.vol),
          'seconds',round(info['bfzi_prep_sec'],2),flush=True)
    print('COLD_PROFILE',json.dumps(profile,separators=(',',':')),flush=True)


def collect(plan,store,report_path,out):
    """Convert completed small files into a portable geometry-labelled dataset."""
    rows={'train':[],'val':[],'test':[]}
    for split in rows:
        for key in plan[split]:
            file=store/f'{key}.npz'
            if not file.exists():continue
            try:
                with np.load(file) as d:
                    z=np.asarray(d['z0'],float)
                    sea=np.asarray(d['sea'],float)
                    air=np.asarray(d['air'],float)
                    cu=np.asarray(d['copper'],float)
                    if z.shape!=(6,) or sea.shape!=(6,) or air.shape!=(2,2):
                        raise ValueError('cold dimensions invalid')
                    if not (np.isfinite(z).all() and np.isfinite(sea).all()
                            and np.isfinite(air).all()):
                        raise ValueError('cold data nonfinite')
                    rows[split].append(dict(name=key,
                        geometry=plan['cases'][key],z0=z.tolist(),
                        sea=sea.tolist(),air=air.tolist(),
                        copper=cu.tolist(),omega=float(d['omega']),
                        sigma=float(d['sigma'])))
            except (ValueError,OSError,KeyError) as e:
                print('COLD_BAD',key,str(e),flush=True)
    data=dict(description='BFZI V150 cold-only two-port reference',
              plan=str(report_path),source='original BFZI V150 cold solver',
              **rows)
    _save(out,data)
    print('COLD_COLLECT',*(f'{s}={len(rows[s])}' for s in rows),flush=True)
    return data


def generate(plan,bfzi_root,out,limit=None,timeout=600,grid='ultra',
             source_order=5,retry=False,offset=0):
    out=Path(out)
    store=out/'samples'
    store.mkdir(parents=True,exist_ok=True)
    report_path=out/'progress.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {
        'status':'running','attempts':{},'failures':{}}
    keys=plan['train']+plan['val']+plan['test']
    active=set(keys)
    report['attempts']=report.get('attempts',{})
    report['failures']={k:v for k,v in report.get('failures',{}).items()
                        if k in active and not (store/f'{k}.npz').exists()}
    todo=keys[offset:] if limit is None else keys[offset:offset+limit]
    done=failed=skipped=0
    for index,key in enumerate(todo,1):
        file=store/f'{key}.npz'
        if file.exists():
            done+=1
            continue
        if key in report['failures'] and not retry:
            skipped+=1
            continue
        cmd=[sys.executable,'-m','examples.bfzi_cold_generate',
             '--one','--bfzi-root',str(bfzi_root),
             '--case-json',json.dumps(plan['cases'][key],separators=(',',':')),
             '--output',str(file),'--grid',grid,
             '--source-order',str(source_order)]
        start=time.monotonic()
        msg=''
        try:
            p=subprocess.run(cmd,check=False,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,text=True,timeout=timeout)
            if p.returncode!=0 or not file.exists():
                raise RuntimeError((p.stdout or 'worker did not save data')[-1500:])
            profile=None
            for line in (p.stdout or '').splitlines():
                if line.startswith('COLD_PROFILE '):
                    try:
                        profile=json.loads(line[len('COLD_PROFILE '):])
                    except (ValueError,TypeError):
                        pass
            report['attempts'][key]=dict(status='ok',
                seconds=round(time.monotonic()-start,2),profile=profile)
            report['failures'].pop(key,None)
            done+=1
            print('COLD',index,len(todo),key,'OK',
                  report['attempts'][key]['seconds'],
                  'profile',profile,flush=True)
        except (subprocess.TimeoutExpired,OSError,RuntimeError) as e:
            msg=str(e)[-1200:]
            report['failures'][key]=dict(reason=msg,
                seconds=round(time.monotonic()-start,2))
            failed+=1
            print('COLD',index,len(todo),key,'FAILED',msg[:160],flush=True)
        report['status']='running'
        report['completed']=done
        report['failed_count']=len(report['failures'])
        _save(report_path,report)
    data=collect(plan,store,report_path,out/'data.json')
    total=sum(len(data[s]) for s in ('train','val','test'))
    report.update(status='complete' if total==len(keys) else 'partial',
                  completed=total,expected=len(keys),
                  failed_count=len(report['failures']))
    _save(report_path,report)
    print('COLD_SUMMARY',total,len(keys),'failed',failed,
          'prior_failed',skipped,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold500'))
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--grid',choices=('ultra','superfine'),default='ultra')
    p.add_argument('--source-order',type=int,default=5)
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--one',action='store_true')
    p.add_argument('--case-json')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.one:
        if not a.case_json or not a.output: p.error('--one requires case and output')
        one(a.bfzi_root,json.loads(a.case_json),a.output,a.grid,a.source_order)
    else:
        plan=apply_replacements(json.loads(a.plan.read_text(encoding='utf-8-sig')))
        generate(plan,a.bfzi_root,a.out,a.limit,a.timeout,
                 a.grid,a.source_order,a.retry_failed,a.offset)


if __name__=='__main__':
    main()
