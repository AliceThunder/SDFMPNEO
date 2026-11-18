"""Quick physical-valid cold training on a snapshot of the running producer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.bfzi_param import originals
from examples.bfzi_cold_lead_guard import gap
from examples.bfzi_cold_train import train as fit_plain
from examples.bfzi_cold_res_train import fit as fit_res


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('runs/geomfrac_cold500/data_partial.json'))
    p.add_argument('--old',type=Path,default=Path('runs/geomfrac_cold_ports_v1.json'))
    p.add_argument('--config',type=Path,default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_coldclean_pilot.json'))
    p.add_argument('--epochs',type=int,default=180)
    a=p.parse_args()
    new=json.loads(a.data.read_text(encoding='utf-8-sig'))
    old=json.loads(a.old.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    bf,_=originals(a.bfzi_root)
    orig={part:len(old[part]) for part in ('train','dev')}
    for part in ('train','dev'):
        old[part]=[r for r in old[part] if all(q['clear'] for q in gap(bf,r['geometry']))]
    valid={part:len(old[part]) for part in ('train','dev')}
    print('CLEAN_COUNTS',json.dumps({part:len(new[part]) for part in ('train','val','test')}),
          'old',orig,'old_valid',valid,flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    plain=fit_plain(new,old,epochs=a.epochs,
                    out='runs/geomfrac_coldclean_plain.pt')
    residual=fit_res(new,old,cfg,epochs=a.epochs,
                     out='runs/geomfrac_coldclean_res.pt')
    res=dict(status='interim_snapshot',new_counts={part:len(new[part])
              for part in ('train','val','test')},
              legacy_orig=orig,legacy_valid=valid,
              plain=dict(best_epoch=plain['best_epoch'],val=plain['val'],test=plain['test']),
              residual=dict(best_epoch=residual['best_epoch'],
                    old_dev=residual['residual_old_dev'],
                    new_val=residual['residual_new_dev'],
                    new_test=residual['residual_new_test'],
                    base_new_test=residual['baseline_new_test']))
    a.out.write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print('CLEAN_RESULT',json.dumps({k:res[k] for k in ('new_counts','legacy_valid')}),
          'plain_test',plain['test']['cold_Z_pct'],
          'residual_test',residual['residual_new_test']['cold_Z_pct'],
          'base_test',residual['baseline_new_test']['cold_Z_pct'],flush=True)


if __name__=='__main__':
    main()
