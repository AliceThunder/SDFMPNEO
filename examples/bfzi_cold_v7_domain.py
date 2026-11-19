"""Calibrate and report the V7 core-domain inference gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sdfmpneo_bfzi.geom_cold_domain import fit,save,load,check,in_core

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("runs/geomfrac_cold_v7/data_merged.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    a=p.parse_args()
    d=json.loads(a.data.read_text(encoding="utf-8-sig"))
    model=fit(d["train"],d["val"])
    save(model,d["train"],a.out)
    model=load(a.out)
    report={}
    for split in ("val","test"):
        rows=[r for r in d[split] if in_core(r["geometry"])]
        results=[check(r["geometry"],model) for r in rows]
        report[split]=dict(core=len(rows),
            accepted=sum(x["accepted"] for x in results),
            rate=(sum(x["accepted"] for x in results)/len(results) if results else None),
            nn_p50=(sorted(x["distance"] for x in results)[len(results)//2]
                    if results else None))
    report.update(status="calibrated",domain_model=str(a.out),
                  threshold=model["max_nn_distance"],
                  train_core=model["train_core"],val_core=model["val_core"])
    a.out.with_name("domain_report.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V7_DOMAIN",json.dumps(report),flush=True)
if __name__=="__main__":main()
