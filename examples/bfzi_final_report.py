"""Report frozen independent BFZI acceptance."""
import json
from pathlib import Path
p=Path("runs")
r=json.loads((p/"bfzi_final_accept.json").read_text())["cases"]
rows=[]
for k in sorted(r):
    x=r[k];s=x["score"];e=x["energy"];t=x["timing"]
    rows.append((k,x["spec"]["family"],x["spec"]["voltage"],
        100*s["delta_Z_rel_L2"],s["copper_max_abs_K"],
        s["pkg_max_abs_K"],t["online_speedup"],
        e["max_rel_thermal_power"],x["passed"]))
n=len(rows);passed=sum(x[-1] for x in rows)
stats=dict(total=n,passed=passed,failed=n-passed,pass_rate=passed/max(n,1),
    worst_delta_Z_percent=max(x[3] for x in rows),
    worst_copper_K=max(x[4] for x in rows),
    worst_pkg_K=max(x[5] for x in rows),
    min_online_speedup=min(x[6] for x in rows),
    worst_thermal_budget=max(x[7] for x in rows))
(p/"bfzi_final_summary.json").write_text(json.dumps(stats,indent=2))
lines=["# BFZI V150 最终全新样本独立验收","",
"冻结 G1/G2 供体的 32 阶 ROM 和 SOE 分数阶神经残差；",
"新几何仅计算原 BFZI V150 legacy 冷场。原 BFZI V150 subcell",
"+ source-skin 完整非线性时程生成独立参考，未用新样本重新训练。",
"24 组预定新工况：圆形/连续圆角/圆角方形/混合线圈；",
"三轴姿态、封装、相对位移、20–110 V、海水温度电导系数和多次启停。",
"600 s，30 s 采样。部分新几何原 2 mm skin 壳超过湿 FV 容量，",
"参考物理法在这些工况采用 0.8 mm 八层 skin 壳，属额外热物理外推。",
"","## 验收标准","",
"ΔZ 相对 L2 ≤3%；铜/封装最大温差 ≤2 K；在线加速 ≥20 倍；",
"端口功率、海水 q 积分、热收支相对残差 ≤1e-8。","",
"## 结果","",
f"已完成 {n}/24；通过 {passed}；失败 {n-passed}；通过率 {passed/max(n,1):.1%}。",
f"最大 ΔZ 误差 {stats['worst_delta_Z_percent']:.3f}%；最大铜温差 {stats['worst_copper_K']:.3f} K；",
f"最大封装温差 {stats['worst_pkg_K']:.3f} K；最小在线加速 {stats['min_online_speedup']:.1f} 倍。",
"","| 样本 | 几何 | V | ΔZ % | 铜 K | 封装 K | 加速 | 热残差 | 判定 |",
"|---|---|---:|---:|---:|---:|---:|---:|---|"]
for k,g,v,z,c,p,s,b,ok in rows:
    lines.append(f"| {k} | {g} | {v:.0f} | {z:.3f} | {c:.3f} | {p:.3f} | {s:.1f} | {b:.2e} | {'通过' if ok else '失败'} |")
lines+=["","## 最终判定","",
    ("24/24 全部通过。" if n==24 and passed==24 else
     "最终整体验收未通过。失败工况作为冻结的独立泛化基准；"
     "优先解决跨几何和高功率热动力学误差，不修改本轮真值。"),
    "","原始 JSON：runs/bfzi_final_accept.json；",
    "每组完整物理和代理时程：runs/bfzi_final_Fij.npz。"]
Path("docs/research/bfzi_uwpt_electrothermal/evidence/final_unseen_acceptance.md").write_text(
    "\n".join(lines),encoding="utf-8")
print(json.dumps(stats,indent=2))
