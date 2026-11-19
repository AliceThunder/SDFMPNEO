"""Complete frozen six-geometry physical dynamic validation after a pilot solve."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def alive(pid):
    cmd=['tasklist','/FI',f'PID eq {pid}','/NH']
    res=subprocess.run(cmd,capture_output=True,text=True,check=False)
    return any(len(parts)>=2 and parts[1]==str(pid)
               for line in res.stdout.splitlines()
               if (parts:=line.split()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--pilot-pid',type=int,default=0)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold_dyn6_plan.json'))
    p.add_argument('--tag',default='cold_dyn6')
    p.add_argument('--wait-min',type=float,default=120.)
    a=p.parse_args()
    if a.pilot_pid:
        print('DYN_WAIT_PILOT',a.pilot_pid,flush=True)
        until=time.monotonic()+a.wait_min*60
        while alive(a.pilot_pid) and time.monotonic()<until:
            time.sleep(15)
        if alive(a.pilot_pid):
            raise TimeoutError('pilot process still running')
    print('DYN_BUILD_ALL',flush=True)
    cmd=[sys.executable,'-m','examples.bfzi_s5_buildset',
         '--bfzi-root',str(a.bfzi_root),'--plan',str(a.plan),
         '--tag',a.tag,'--grid','ultra','--duration','300']
    subprocess.run(cmd,check=True)
    print('DYN_EVAL',flush=True)
    subprocess.run([sys.executable,'-m','examples.bfzi_cold_dyn6_eval'],
                   check=True)
    print('DYN_COMPLETE',flush=True)


if __name__=='__main__':
    main()
