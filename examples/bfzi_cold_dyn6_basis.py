"""Measure canonical 32/48/64-mode thermal field projection on new S5 holdouts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from examples.bfzi_geom_basis_check import metrics
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.s5_bridge import teacher_temp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,
      default=Path('runs/geomfrac_s5_cold_dyn6_manifest.json'))
    p.add_argument('--out',type=Path,
      default=Path('runs/geomfrac_cold_dyn6_basis.json'))
    p.add_argument('--ranks',nargs='+',type=int,default=[32,48,64])
    a=p.parse_args()
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    out=dict(status='complete',ranks=a.ranks,cases={})
    for spec in mf.get('test',[]):
        anchor=Path(spec['anchor'])
        head=GeomHeat.load(anchor)
        name=anchor.stem.split('_')[2]
        out['cases'][name]={}
        for file in spec['teacher']:
            with np.load(file) as z:
                temp=teacher_temp(z['temp_K'],head)
            label=Path(file).stem.split('_')[-1]
            out['cases'][name][label]={}
            for rank in a.ranks:
                try:
                    m=metrics(head,temp,rank)
                    out['cases'][name][label][str(rank)]={
                        'global_max_K':m['global_max_K'],
                        'global_rmse_K':m['global_rmse_K'],
                        'sea_skin_max_K':m['regions'].get('sea_skin',{}).get('max_abs_K'),
                        'sea_skin_rmse_K':m['regions'].get('sea_skin',{}).get('rmse_K'),
                        'skin_tx_max_K':m['regions'].get('skin_tx',{}).get('max_abs_K'),
                        'package_tx_max_K':m['regions']['package_tx']['max_abs_K'],
                        'package_rx_max_K':m['regions']['package_rx']['max_abs_K']}
                except Exception as e:
                    out['cases'][name][label][str(rank)]={'error':f'{type(e).__name__}: {e}'}
        print('BASIS_CASE',name,out['cases'][name],flush=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
