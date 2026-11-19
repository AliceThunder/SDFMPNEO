"""Merge V7 cold references with physically valid legacy cold samples."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from examples.bfzi_cold_v7_plan import category,key,read
from examples.bfzi_cold_v7_replacements import apply as apply_replacements

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_cold_v7_plan.json"))
    p.add_argument("--new",type=Path,default=Path("runs/geomfrac_cold_v7/data.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_cold_v7/data_merged.json"))
    a=p.parse_args()
    plan=apply_replacements(read(a.plan));new=read(a.new)
    parts={s:[] for s in ("train","val","test")}
    for src in plan["existing_sources"].values():
        d=read(src)
        for s in parts:
            for row in d.get(s,[]):
                item=dict(row);item["stratum"]=category(item["geometry"])
                parts[s].append(item)
    old=read(plan["legacy_source"])
    selected={"train":set(plan["legacy_included"]["train"]),
              "val":set(plan["legacy_included"]["val"])}
    for s,old_part in (("train","train"),("val","dev")):
        for row in old.get(old_part,[]):
            if row["name"] in selected[s]:
                item=dict(row);item["stratum"]=category(item["geometry"])
                parts[s].append(item)
    for s in parts:
        for row in new.get(s,[]):
            item=dict(row);item["stratum"]=plan["strata"][row["name"]]
            parts[s].append(item)
    seen={}
    for s,rows in parts.items():
        names=set()
        for row in rows:
            k=key(row["geometry"])
            if k in seen:raise ValueError(f"geometry overlap {seen[k]} / {s}: {row['name']}")
            if row["name"] in names:raise ValueError(f"duplicate row name in {s}: {row['name']}")
            seen[k]=s;names.add(row["name"])
    from collections import Counter
    counts={s:dict(total=len(rows),strata=dict(Counter(r["stratum"] for r in rows)))
            for s,rows in parts.items()}
    if counts["train"]["total"]!=1500:
        raise ValueError(f"V7 requires exactly 1500 unique train geometries: {counts['train']}")
    if counts["train"]["strata"]!={"core":1050,"boundary":300,"extreme":150}:
        raise ValueError(f"V7 training strata mismatch: {counts['train']['strata']}")
    out=dict(description="V7 merged BFZI V150 cold-only data",
             version=7,plan=str(a.plan),source="BFZI V150",
             train=parts["train"],val=parts["val"],test=parts["test"])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    report=dict(status="merged",counts=counts,
                geometry_unique=len(seen),
                inference_domain=plan["domain"]["inference"])
    a.out.with_name("merge_report.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V7_MERGED",json.dumps(report),flush=True)

if __name__=="__main__":main()
