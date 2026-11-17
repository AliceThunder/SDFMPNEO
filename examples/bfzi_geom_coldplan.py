"""Freeze new geometry-disjoint cold-only training cases."""
import json
from pathlib import Path
from examples.bfzi_s5_v4_plan import design
from examples.bfzi_s5_buildset import check_plan


def main():
    fresh=design(seed=2457)
    names=[f'C{k:02d}' for k in range(1,25)]
    cases={name:fresh['cases'][f'N{i+2:02d}'] for i,name in enumerate(names,1)}
    old=[]
    for fn in ('geomfrac_s5_v4_plan.json','geomfrac_s5_v5_plan.json',
               'geomfrac_s5_v6_plan.json','geomfrac_s5_v7_plan.json',
               'geomfrac_s5_v9_plan.json'):
        old.extend(json.loads((Path('runs')/fn).read_text())['cases'].values())
    seen={json.dumps(c['geometry'],sort_keys=True) for c in old}
    assert all(json.dumps(c['geometry'],sort_keys=True) not in seen
               for c in cases.values())
    plan=dict(description='Extra 24 cold-only BFZI training geometries, no dynamic truth',
              seed=2457,train=names,val=[],test=[],cases=cases)
    check_plan(plan)
    out=Path('runs/geomfrac_cold_v2_plan.json')
    out.write_text(json.dumps(plan,indent=2))
    print('FROZEN_COLD_ONLY',len(names),out)


if __name__=='__main__':
    main()
