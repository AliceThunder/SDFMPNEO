"""Compare held-out BFZI M4 dynamic ROM to the full original resolver."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--direct",type=Path,required=True)
    p.add_argument("--rom",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    full=json.loads(a.direct.read_text(encoding="utf-8"))
    red=json.loads(a.rom.read_text(encoding="utf-8"))
    def z(trace):
        x=np.asarray([r["Z_ohm"] for r in trace],float)
        return x[...,0]+1j*x[...,1]
    f=full["trace"]
    r=red["trace"]
    if len(f)!=len(r) or any(x["time_s"]!=y["time_s"] for x,y in zip(f,r)):
        raise ValueError("direct and ROM time grids differ")
    zf,zr=z(f),z(r)
    tf=np.asarray([x["Tsea_max_K"] for x in f])
    tr=np.asarray([x["Tsea_max_K"] for x in r])
    cf=np.asarray([x["Tcu_K"] for x in f])
    cr=np.asarray([x["Tcu_K"] for x in r])
    ef=np.asarray([x["efficiency"] for x in f])
    er=np.asarray([x["efficiency"] for x in r])
    pkg_fields=("Qcu_to_pkg_W","Qpkg_to_sea_W","Epkg_J")
    has_pkg=all(all(k in x for k in pkg_fields) for x in f+r)
    has_energy=all("material_balance_relative_gap" in x for x in f[1:]+r[1:])
    with np.load(a.direct.with_suffix(".npz")) as d, np.load(a.rom.with_suffix(".npz")) as q:
        jf=np.asarray(d["Jsea_unit_A_m2_per_A"])
        jr=np.asarray(q["Jsea_unit_A_m2_per_A"])
        heat_f=np.asarray(d["qsea_W_m3"])
        heat_r=np.asarray(q["qsea_W_m3"])
        tsea_f=np.asarray(d["Tsea_K"])
        tsea_r=np.asarray(q["Tsea_K"])
        tpkg_f=np.asarray(d["Tpkg_K"])
        tpkg_r=np.asarray(q["Tpkg_K"])
        pkg_field=None
        if "Tpkg_cells_K" in d and "Tpkg_cells_K" in q:
            pkf=np.asarray(d["Tpkg_cells_K"])
            pkr=np.asarray(q["Tpkg_cells_K"])
            if pkf.shape!=pkr.shape:
                raise ValueError("3-D package grid shapes differ")
            pkg_field=float(np.max(abs(pkf-pkr)))
        rms_j=float(np.linalg.norm(jf-jr)/max(np.linalg.norm(jf),1e-30))
        rms_q=float(np.linalg.norm(heat_f-heat_r)/max(np.linalg.norm(heat_f),1e-30))
    t_full=float(full["trajectory_runtime_sec"])
    t_rom=float(red["trajectory_runtime_sec"])
    data={
        "direct":str(a.direct),"rom":str(a.rom),
        "rom_basis":red.get("rom"),
        "heat_rom_basis":red.get("heat_rom"),
        "steps":len(f),
        "complex_Z_relative_L2":float(np.linalg.norm(zf-zr)/np.linalg.norm(zf)),
        "complex_Z_max_abs_ohm":float(np.max(abs(zf-zr))),
        "sea_J_unit_relative_L2":rms_j,
        "sea_q_relative_L2":rms_q,
        "Tsea_max_absolute_K":float(np.max(abs(tf-tr))),
        "Tsea_spatial_max_absolute_K":float(np.max(abs(tsea_f-tsea_r))),
        "Tpkg_max_absolute_K":float(np.max(abs(tpkg_f-tpkg_r))),
        "Tcu_max_absolute_K":float(np.max(abs(cf-cr))),
        "eta_max_absolute":float(np.max(abs(ef-er))),
        "direct_trajectory_sec":t_full,
        "rom_trajectory_sec":t_rom,
        "speedup":t_full/max(t_rom,1e-30),
        "rom_thermal_balance_max":max(x["thermal_balance_relative_gap"] or 0 for x in r)
    }
    if pkg_field is not None:
        data["Tpkg_spatial_max_absolute_K"]=pkg_field
    if has_pkg:
        for name,key in (
            ("Qcu_to_pkg_max_absolute_W","Qcu_to_pkg_W"),
            ("Qpkg_to_sea_max_absolute_W","Qpkg_to_sea_W"),
            ("Epkg_max_absolute_J","Epkg_J")
        ):
            vf=np.asarray([x[key] for x in f])
            vr=np.asarray([x[key] for x in r])
            data[name]=float(np.max(abs(vf-vr)))
    if has_energy:
        data["rom_material_balance_max"]=max(
            max(x["material_balance_relative_gap"].values()) for x in r[1:])
        data["direct_material_balance_max"]=max(
            max(x["material_balance_relative_gap"].values()) for x in f[1:])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    main()
