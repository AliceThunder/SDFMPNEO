"""Apply small, explicit replacement records to the frozen V7 plan."""
from __future__ import annotations
import json
from pathlib import Path

def apply(plan,path=Path("runs/geomfrac_cold_v7_replacements.json")):
    if not path.exists():
        return plan
    patch=json.loads(path.read_text(encoding="utf-8-sig"))
    for item in patch.get("items",[]):
        old=item["replaced"]
        new=item["replacement"]
        found=[]
        for split in ("train","val","test"):
            if old in plan[split]:
                found.append(split)
        if len(found)!=1:
            raise ValueError(f"V7 replacement source must occur in exactly one split: {old}")
        split=found[0]
        if old not in plan["strata"] or plan["strata"][old]!=new["stratum"]:
            raise ValueError(f"V7 replacement stratum mismatch: {old}")
        name=new["name"]
        if name in plan["cases"] or name in plan["strata"]:
            raise ValueError(f"V7 replacement ID already exists: {name}")
        plan[split][plan[split].index(old)]=name
        plan["cases"].pop(old)
        plan["strata"].pop(old)
        plan["cases"][name]=new["geometry"]
        plan["strata"][name]=new["stratum"]
    return plan
