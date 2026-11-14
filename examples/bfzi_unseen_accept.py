"""Aggregate the frozen U01-U12 BFZI V150 four-run direct-physics acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS=[f"U{i:02d}" for i in range(1,13)]
CFG=[("ultra",2),("ultra",1),("superfine",2),("superfine",1)]
FILES=[
    "bfzi_physical_unseen12_probe.json",
    "bfzi_physical_U09_corrected_probe.json",
    "bfzi_physical_unseen12_tail.json",
    "bfzi_unseen_U01_U04_h1.json",
    "bfzi_unseen_U05_U12_h1.json",
    "bfzi_unseen_U01_U06_sf_h2.json",
    "bfzi_unseen_U07_U12_sf_h2.json",
    "bfzi_unseen_U01_U06_sf_h1.json",
    "bfzi_unseen_U07_U12_sf_h1.json",
]


def cmp(a,b):
    if a is None or b is None:
        return None
    ta,tb=a["time_s"],b["time_s"]
    if len(ta)!=len(tb) or not np.allclose(ta,tb,rtol=0,atol=1e-8):
        return dict(error="nonmatching output times")
    if len(ta)!=41 or abs(float(ta[-1])-600)>1e-8:
        return dict(error="incomplete time history")
    x=a["z6"]-a["z6"][0]
    y=b["z6"]-b["z6"][0]
    return dict(dz=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),
                cu=float(np.max(abs(a["Tcu_K"]-b["Tcu_K"]))),
                pkg=float(np.max(abs(a["Tpkg_K"]-b["Tpkg_K"]))))


def check(v,kind):
    if not v or "error" in v:
        return False
    lim=(.01,.5,.5) if kind=="time" else (.02,1.,1.5)
    return all(v[k]<=t for k,t in zip(("dz","cu","pkg"),lim))


def fmt(v,kind):
    if v is None:
        return "待运行"
    if "error" in v:
        return "失败"
    return f'{v["dz"]*100:.3f}% / {v["cu"]:.3f} / {v["pkg"]:.3f}'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",type=Path,default=Path("runs"))
    ap.add_argument("--run-tag",default="",help="select independent new direct-physics run")
    ap.add_argument("--out",type=Path,default=Path("runs/bfzi_physical_unseen12_final.json"))
    ap.add_argument("--report",type=Path,default=Path(
        "docs/research/bfzi_uwpt_electrothermal/evidence/physical_unseen12_final.md"))
    a=ap.parse_args()
    meta={}
    names=(sorted(p.name for p in a.runs.glob(
               f"bfzi_unseen_*_{a.run_tag}_*.json")) if a.run_tag else
           sorted(set(FILES+
               [p.name for p in a.runs.glob("bfzi_unseen_U*.json")]+
               [p.name for p in a.runs.glob("bfzi_unseen_resume_*.json")])))
    for name in names:
        p=a.runs/name
        if not p.exists():
            continue
        d=json.loads(p.read_text(encoding="utf-8"))
        for key,case in d.get("cases",{}).items():
            for tag,rec in case.get("runs",{}).items():
                h=d["settings"]["inner_step_s"]
                if h in (1,2) and tag in ("ultra_dt15","superfine_dt15"):
                    g=tag.removesuffix("_dt15")
                    meta[(key,g,int(h))]=rec
    if not a.run_tag:
        base=a.runs/"bfzi_physical_unseen12_final.json"
        if base.exists():
            frozen=json.loads(base.read_text(encoding="utf-8"))
            for key,c in frozen["cases"].items():
                for g,h in CFG:
                    rec=c["runs"].get(f"{g}_h{h}")
                    if rec is not None:
                        meta.setdefault((key,g,h),rec)
    spec=json.loads((a.runs/"bfzi_physical_unseen12_corrected.json").read_text(encoding="utf-8"))
    cases={}
    for key in KEYS:
        raw={}
        detail={}
        for grid,h in CFG:
            pat=(f"bfzi_phys_{key}_{grid}_dt15_h{h}*"
                 + (f"_{a.run_tag}.npz" if a.run_tag else ".npz"))
            files=list(a.runs.glob(pat))
            if not a.run_tag and base.exists():
                name=frozen["cases"][key]["runs"][f"{grid}_h{h}"].get("file")
                p=a.runs/Path(name).name if name else None
                files=[p] if p is not None and p.exists() else []
            rec=meta.get((key,grid,h))
            k=f"{grid}_h{h}"
            if len(files)==1 and rec is not None:
                with np.load(files[0]) as z:
                    raw[k]={s:z[s] for s in ("time_s","z6","Tcu_K","Tpkg_K","temp_K")}
                detail[k]=dict(**rec,file=str(files[0]).replace("\\","/"))
            else:
                raw[k]=None
                detail[k]=dict(status="pending",files_found=len(files))
        tests={
            "time_ultra":cmp(raw["ultra_h2"],raw["ultra_h1"]),
            "time_superfine":cmp(raw["superfine_h2"],raw["superfine_h1"]),
            "space_h2":cmp(raw["ultra_h2"],raw["superfine_h2"]),
            "space_h1":cmp(raw["ultra_h1"],raw["superfine_h1"]),
        }
        phys=True
        for k,r in detail.items():
            if r.get("status")!="candidate" or r.get("completed_s")!=600:
                phys=False
            if (r.get("status")=="candidate" and (
                r["max_water_K"]>373.15 or
                r["max_copper_K"]>=1357.77 or
                r["peak_thermal_residual_W"]>1e-4 or
                r["peak_global_budget_W"]>1e-4 or
                r["max_rel_port_power"]>1e-10 or
                r["max_rel_q_power"]>1e-10 or
                abs(r["skin_water_capacity_error_J_K"])>1e-5)):
                phys=False
            if raw[k] is not None and np.min(raw[k]["temp_K"])<271.15:
                phys=False
        checks={k:check(v,"time" if k.startswith("time") else "space")
                for k,v in tests.items()}
        complete=all(v is not None for v in raw.values())
        state=("pending" if not complete else
               "pass" if phys and all(checks.values()) else "fail")
        failed=[k for k in tests if tests[k] is not None and not checks[k]]
        if complete and not phys:
            failed.insert(0,"physical_validity_or_conservation")
        dyn=None
        ref=next((k for k in ("superfine_h1","superfine_h2","ultra_h1","ultra_h2")
                  if raw[k] is not None),None)
        if ref is not None:
            z=raw[ref]["z6"]
            t=raw[ref]["time_s"]
            tc=raw[ref]["Tcu_K"].max(axis=1)
            tp=raw[ref]["Tpkg_K"].max(axis=1)
            dyn=dict(reference=ref,
                     max_delta_R_ohm=float(np.max(abs(z[:,0::2]-z[0,0::2]))),
                     max_delta_X_ohm=float(np.max(abs(z[:,1::2]-z[0,1::2]))),
                     peak_copper_s=float(t[np.argmax(tc)]),
                     peak_package_s=float(t[np.argmax(tp)]))
        cases[key]=dict(state=state,geometry=spec["cases"][key]["geometry"],
                        thermal=spec["cases"][key]["thermal"],runs=detail,
                        comparisons=tests,checks=checks,physical_ok=phys,
                        failed_checks=failed,dynamics=dyn)
    counts={name:sum(c["state"]==name for c in cases.values())
            for name in ("pass","fail","pending")}
    vals={k:[c["comparisons"][k] for c in cases.values()
             if c["comparisons"][k] is not None and
                "error" not in c["comparisons"][k]]
          for k in ("time_ultra","time_superfine","space_h2","space_h1")}
    stats={k:{m:dict(max=float(max(x[m] for x in v)),
                     mean=float(np.mean([x[m] for x in v])),
                     p95=float(np.percentile([x[m] for x in v],95)))
              for m in ("dz","cu","pkg")} if v else {}
           for k,v in vals.items()}
    rows=[r for c in cases.values() for r in c["runs"].values()
          if r.get("status")=="candidate"]
    fields=("max_rel_port_power","max_rel_q_power","peak_thermal_residual_W",
            "peak_global_budget_W","skin_water_capacity_error_J_K")
    conservation={k:max(abs(r[k]) for r in rows) if rows else None
                  for k in fields}
    out=dict(method="BFZI V150 four-combination unseen independent physics",
             run_tag=a.run_tag,
             grid={"ultra":"2668-class wet FV","superfine":"3729-class wet FV"},
             true_heat_step_s={"h2":1.875,"h1":1.0},
             output_dt_s=15,elapsed_s=600,counts=counts,
             accepted=counts["pass"]==12,original_U09="invalid BFZI geometry; preserved",
             corrected_U09="alpha2=0.86, core2_mm=20",
             stats=stats,conservation=conservation,cases=cases)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[
        "# BFZI V150 U01–U12 独立物理直接法四组合验收",
        "",
        f'当前结果：通过 {counts["pass"]}/12，失败 {counts["fail"]}/12，待完成 {counts["pending"]}/12；总验收 {"PASS" if out["accepted"] else "OPEN"}。',
        "",
        "对每个工况进行 ultra/superfine 两档原生湿 FV × 1.875/1 s 真实热步，统一 15 s 阻抗采样，完整 600 s。",
        "",
        "## 全部样本",
        "",
        "| 样本 | 状态 | 时间 ultra | 时间 superfine | 空间 1.875 s | 空间 1 s | 铜/封装/海水最高温 (K) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key,c in cases.items():
        vs=[r for r in c["runs"].values() if r.get("status")=="candidate"]
        peak=("/".join(f"{max(r[x] for r in vs):.2f}" for x in
              ("max_copper_K","max_package_K","max_water_K")) if vs else "待运行")
        lines.append("| "+ " | ".join([
            key,c["state"],*(fmt(c["comparisons"][k],"time") for k in
                                 ("time_ultra","time_superfine")),
            *(fmt(c["comparisons"][k],"space") for k in
              ("space_h2","space_h1")),peak])+" |")
    lines+=["","表中四类误差分别为动态复阻抗相对 L2、铜最高逐时刻绝对差 K、封装最高逐时刻绝对差 K。",
            "时间门槛：1%、0.5 K、0.5 K。空间门槛：2%、1 K、1.5 K。",
            "",
            "## 已有四组合误差统计",""]
    for k,rows in stats.items():
        lines.append(f'### {k}（{len(vals[k])} 组可比较）')
        for m,title,unit in (("dz","动态复阻抗","%"),
                             ("cu","铜温度","K"),("pkg","封装温度","K")):
            if not rows:
                continue
            s=rows[m]
            a1=100 if m=="dz" else 1
            lines.append(f'{title}：最大 {s["max"]*a1:.4f}{unit}，均值 {s["mean"]*a1:.4f}{unit}，P95 {s["p95"]*a1:.4f}{unit}。')
        lines.append("")
    lines+=["## 物理有效域、功率与热能守恒",
            "",
            "每一组合均需满足海水 271.15–373.15 K、铜低于 1357.77 K；端口电功率相对误差和海水积分焦耳损耗相对误差均不超过 1e-10，热节点及全局热预算不超过 1e-4 W，海水热容量守恒。",
            "",
            "## 几何修订与可复现性","",
            "原冻结 U09 的 alpha2=0.9851324743684382、core2_mm=25 未通过原 BFZI 几何有效性检查，保留原始失败记录。修订 U09 仅改变 alpha2=0.86、core2_mm=20；其余样本与功率、热参数均保持冻结。",
            "",
            "在仓库根目录运行（BFZI 依赖目录为相邻 BFZI）：",
            "",
            f"    python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys {' '.join(KEYS)} --dt 15 --grids ultra superfine --inner-step 2{f' --run-tag {a.run_tag}' if a.run_tag else ''} --out runs/bfzi_unseen_all_{a.run_tag + '_' if a.run_tag else ''}h2.json",
            f"    python -m examples.bfzi_physics_recheck --bfzi-root ../BFZI --runs runs --spec-file runs/bfzi_physical_unseen12_corrected.json --keys {' '.join(KEYS)} --dt 15 --grids ultra superfine --inner-step 1{f' --run-tag {a.run_tag}' if a.run_tag else ''} --out runs/bfzi_unseen_all_{a.run_tag + '_' if a.run_tag else ''}h1.json",
            f"    python -m examples.bfzi_unseen_accept{f' --run-tag {a.run_tag}' if a.run_tag else ''} --out {a.out} --report {a.report}",
            "",
            f"源数据保存在 runs/bfzi_phys_U*{f'_{a.run_tag}' if a.run_tag else ''}.npz；所有样本的单组合功率、温度及热残差在 {a.out} 中。",
            ""]
    lines+=["## 守恒指标（所有已完成组合的最大绝对值）",""]
    for k,v in conservation.items():
        lines.append(f"{k}: {v:.6e}" if v is not None else f"{k}: 待运行")
    lines+=["","## 动态阻抗与温度峰值时刻","",
            "| 样本 | 最大动态 R (Ω) | 最大动态 X (Ω) | 铜峰时刻 (s) | 封装峰时刻 (s) |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for key,c in cases.items():
        d=c["dynamics"]
        if d:
            lines.append(f'| {key} | {d["max_delta_R_ohm"]:.6g} | {d["max_delta_X_ohm"]:.6g} | {d["peak_copper_s"]:.0f} | {d["peak_package_s"]:.0f} |')
        else:
            lines.append(f"| {key} | 待运行 | 待运行 | 待运行 | 待运行 |")
    lines+=["","## 失败项",""]
    lines.extend(f"{k}: {', '.join(c['failed_checks'])}" for k,c in
                 cases.items() if c["state"]=="fail")
    lines.append("")
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text("\n".join(lines),encoding="utf-8")
    print("U01-U12",counts,"accepted",out["accepted"])
    for key,c in cases.items():
        print(key,c["state"]," ".join(
            f'{k}:{v["dz"]*100:.3f}%' for k,v in c["comparisons"].items()
            if v and "dz" in v))


if __name__=="__main__":
    main()
