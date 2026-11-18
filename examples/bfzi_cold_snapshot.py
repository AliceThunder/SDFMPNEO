"""Read complete cold reference files while the producer continues running."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.bfzi_cold_generate import collect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--store',type=Path,
                   default=Path('runs/geomfrac_cold500'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold500/data_partial.json'))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    collect(plan,a.store/'samples',a.store/'progress.json',a.out)


if __name__=='__main__':
    main()
