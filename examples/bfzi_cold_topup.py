"""Append replacements after a cold-only BFZI batch has finished."""
import argparse
import json
from pathlib import Path

from examples.bfzi_cold_plan import sample


def topup(plan,store,seed=62027):
    goals=dict(train=400,val=50,test=50)
    done={part:sum((store/'samples'/f'{k}.npz').exists()
                   for k in plan[part]) for part in goals}
    need={part:max(0,goals[part]-done[part]) for part in goals}
    if not any(need.values()):
        return plan,done,{}
    # Reserve some extra candidate geometries for physically invalid cases.
    counts={part:int(n*1.2)+2 if n else 0 for part,n in need.items()}
    n=sum(counts.values())
    occupied=[c for c in plan['cases'].values()]
    new=sample(seed,n,existing=occupied)
    keys=list(new['cases'])
    number=max(int(k[1:]) for k in plan['cases'])+1
    extra={}
    idx=0
    for split,count in counts.items():
        extra[split]=[]
        for _ in range(count):
            original=keys[idx]
            key=f'C{number:04d}'
            number+=1
            plan['cases'][key]=new['cases'][original]
            plan[split].append(key)
            extra[split].append(key)
            idx+=1
    plan.setdefault('extensions',[]).append(dict(seed=seed,needed=need,
                                                  added=counts))
    return plan,done,extra


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--store',type=Path,
                   default=Path('runs/geomfrac_cold500'))
    p.add_argument('--seed',type=int,default=62027)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    progress=a.store/'progress.json'
    if not progress.exists():
        p.error('run initial cold generation before top-up')
    status=json.loads(progress.read_text(encoding='utf-8'))
    if status.get('status')=='running':
        p.error('cold generator is still running; top-up after batch completion')
    old=len(plan['cases'])
    new,done,added=topup(plan,a.store,a.seed)
    if added:
        a.plan.write_text(json.dumps(new,indent=2)+'\n',encoding='utf-8')
    print('COLD_TOPUP','completed',done,'new_candidates',
          {s:len(v) for s,v in added.items()},'total',len(new['cases']),
          'previous',old,flush=True)


if __name__=='__main__':
    main()
