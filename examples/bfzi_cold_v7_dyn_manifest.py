"""Filter the accumulated S5 manifest to the frozen V7 core dynamic holdout."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def name(spec):
    return Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_cold_v7_dyn_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v7core_manifest.json"),
                   help="newly generated four-geometry BFZI dynamic manifest")
    p.add_argument("--base-manifest",type=Path,default=Path("runs/geomfrac_s5_cold_dyn6_manifest.json"),
                   help="existing dynamic manifest containing C0416/C0725")
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_s5_cold_v7_core_manifest.json"))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8-sig"))
    src=json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    base=json.loads(a.base_manifest.read_text(encoding="utf-8-sig"))
    wanted=set(plan["test"])
    reuse=set(plan.get("reused_existing",[]))
    fresh=wanted-reuse
    rows=[s for s in base.get("test",[]) if name(s) in reuse]
    rows.extend(s for s in src.get("test",[]) if name(s) in fresh)
    found={name(s) for s in rows}
    if found!=wanted:raise RuntimeError(f"manifest missing {sorted(wanted-found)}")
    out=dict(train=[],val=[],test=rows)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("V7_DYN_MANIFEST",len(rows),sorted(found),flush=True)
if __name__=="__main__":main()
