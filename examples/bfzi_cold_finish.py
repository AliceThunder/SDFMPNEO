"""Finish a cold500 run: replenish invalid geometries, train and evaluate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from examples.bfzi_cold_generate import generate
from examples.bfzi_cold_topup import topup
from examples.bfzi_cold_train import train as cold_fit
from examples.bfzi_cold_res_train import fit as res_fit
from examples.bfzi_param import originals
from examples.bfzi_cold_lead_guard import gap,scan,apply


GOAL=dict(train=400,val=50,test=50)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _save(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def finish(plan_path,store,bfzi_root,max_rounds=4):
    pp=store/'progress.json'
    if not pp.exists():
        raise RuntimeError('cold500 generation has not started')
    if _read(pp).get('status')=='running':
        print('COLD_FINISH_PENDING','producer still running',flush=True)
        return None
    bf,_=originals(bfzi_root)
    plan=_read(plan_path)
    guard=scan(plan,bf)
    rejected=apply(plan,guard)
    if any(rejected.values()):
        _save(plan_path,plan)
        print('COLD_LEAD_EXCLUDED',{k:len(v) for k,v in rejected.items()},
              flush=True)
        # Raw sample NPZ files are retained; only eligible split IDs change.
        generate(plan,bfzi_root,store,timeout=600)
    accept=lambda g:all(p['clear'] for p in gap(bf,g))
    for j in range(max_rounds):
        plan=_read(plan_path)
        data=_read(store/'data.json')
        counts={k:len(data[k]) for k in GOAL}
        print('COLD_COUNTS',counts,flush=True)
        if all(counts[k]>=GOAL[k] for k in GOAL):
            break
        plan,done,added=topup(plan,store,62027+j,accept=accept)
        if not added:
            raise RuntimeError('insufficient valid cases with no top-up candidates')
        _save(plan_path,plan)
        print('COLD_TOPUP',j+1,
              {k:len(v) for k,v in added.items()},flush=True)
        generate(plan,bfzi_root,store,timeout=600)
    data=_read(store/'data.json')
    counts={k:len(data[k]) for k in GOAL}
    if any(counts[k]<GOAL[k] for k in GOAL):
        raise RuntimeError(f'cold dataset still incomplete: {counts}')
    old=_read(Path('runs/geomfrac_cold_ports_v1.json'))
    cfg=_read(Path('runs/geomfrac_free_pose42_kell_v1.json'))
    nres=cold_fit(data,old,epochs=400,
                  out='runs/geomfrac_coldnet_v1.pt')
    rres=res_fit(data,old,cfg,out='runs/geomfrac_coldres_v1.pt',
                 epochs=400)
    _save('runs/geomfrac_coldnet_v1.json',nres)
    _save('runs/geomfrac_coldres_v1.json',rres)
    report=dict(status='cold_independent_evaluation_complete',
                valid_cases=counts,
                plain=dict(olddev=nres['val'],newtest=nres['test']),
                residual=dict(olddev=rres['residual_old_dev'],
                              newtest=rres['residual_new_test']),
                baseline_new_test=rres['baseline_new_test'],
                note='Frozen dynamic V5 closed-loop acceptance is a separate run')
    for label,path in [('plain','runs/geomfrac_coldnet_v1.pt'),
                       ('residual','runs/geomfrac_coldres_v1.pt')]:
        cmd=[sys.executable,'-m','examples.bfzi_cold_closed_eval',
             '--cold-model',path,
             '--out',f'runs/geomfrac_cold_{label}_closed_v1.json']
        subprocess.run(cmd,check=True)
        check=_read(f'runs/geomfrac_cold_{label}_closed_v1.json')
        report[label+'_closed']=check
    _save('runs/geomfrac_cold500_final.json',report)
    print('COLD_FINISH_DONE',json.dumps(counts),
          'plain_test',nres['test']['cold_Z_pct'],
          'residual_test',rres['residual_new_test']['cold_Z_pct'],
          flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--store',type=Path,
                   default=Path('runs/geomfrac_cold500'))
    p.add_argument('--watch',action='store_true',
                   help='wait for active cold producer before finishing')
    a=p.parse_args()
    if a.watch:
        print('COLD_FINISH_WATCH',a.store,flush=True)
        start=time.monotonic()
        while (a.store/'progress.json').exists() and (
                _read(a.store/'progress.json').get('status')=='running'):
            if time.monotonic()-start>2.5*3600:
                raise TimeoutError('producer did not finish within watch window')
            time.sleep(20)
    finish(a.plan,a.store,a.bfzi_root)


if __name__=='__main__':
    main()
