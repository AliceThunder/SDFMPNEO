"""V2.1 transient electrothermal solver using the original BFZI v148 backend.

Usage:
  python run.py --mode bfzi-et-v148 --bfzi-root /path/to/BFZI --sample M4

The BFZI checkout is read, never modified. Original BFZI v148 geometry,
Green/resolvent, source mass and air inductance are the electromagnetic base.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_et import BFZIV148ET


def pair(x):
    z=np.asarray(x)
    return np.stack((z.real,z.imag),axis=-1).tolist()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True,
                   help="Path to existing AliceThunder/BFZI checkout")
    p.add_argument("--sample",choices=("M1","M2","M3","M4"),default="M4")
    p.add_argument("--voltage",type=float,default=3.)
    p.add_argument("--step",type=float,default=60.)
    p.add_argument("--heat",type=float,default=240.)
    p.add_argument("--cool",type=float,default=180.)
    p.add_argument("--restart",type=float,default=180.)
    p.add_argument("--sea-alpha",type=float,default=.02)
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_v148_dynamic.json"))
    a=p.parse_args(argv)
    if (not a.bfzi_root.is_dir() or
            not (a.bfzi_root/"bfzi_v148_results.py").exists()):
        p.error("bfzi-root must be an existing BFZI repository checkout")
    if min(a.voltage,a.step,a.heat,a.cool,a.restart) <= 0:
        p.error("drive voltage and times must be positive")
    sys.path.insert(0,str(a.bfzi_root.resolve()))
    import bfzi_v148_results as calc

    data=(calc.build_circle_data("M4") if a.sample=="M4"
          else calc.base.build_dual_data(a.sample))
    backend=BFZIV148Adapter(data,calc,sea_alpha=a.sea_alpha)
    extra=np.diag([.5-1j*backend.omega*backend.air[i,i] for i in range(2)])
    et=BFZIV148ET(backend,external=extra)
    duration=a.heat+a.cool+a.restart
    times=np.unique(np.r_[np.arange(0,duration,a.step),
                           a.heat,a.heat+a.cool,duration])
    def drive(t):
        active=(t<=a.heat or t>a.heat+a.cool)
        return np.array([a.voltage if active else 0.,0.],complex)

    trace=et.trajectory(times,drive)
    # Exactly the same original BFZI v148 isothermal sea impedance is the
    # regression baseline: geometry, Green, port orientation and local mass.
    z_ref,_,_,_,_ = calc.formal_with_local_mass(data)
    warm_rel = float(np.linalg.norm(trace[0].sea_z-z_ref)
                     / max(np.linalg.norm(z_ref),1e-30))
    # Independent BFZI master/outputs_v148 archive, M4 two-port sea reaction.
    # This is a source-code regression target, not an experimental reference.
    archive_rel = None
    if a.sample == "M4":
        z11 = .12781381449951726-0.0038545269438708036j
        z12 = .10805885891894453-0.003731629612853443j
        archive = np.array([[z11,z12],[z12,z11]])
        archive_rel = float(np.linalg.norm(z_ref-archive)
                            / np.linalg.norm(archive))
    rows=[]
    for s in trace:
        rows.append({
            "time_s":s.time,
            "drive":"on" if np.any(abs(s.currents)>0) else "off",
            "Z_ohm":pair(s.impedance),
            "Zsea_ohm":pair(s.sea_z),
            "Leff_H":(s.impedance.imag/backend.omega).tolist(),
            "I_A":pair(s.currents),
            "Psea_volume_W":s.sea_power,
            "Psea_port_W":float(.5*np.real(
                np.vdot(s.currents,s.sea_z@s.currents))),
            "Psea_port_volume_relative_gap":float(
                abs(.5*np.real(np.vdot(s.currents,s.sea_z@s.currents))
                    -s.sea_power)/max(abs(s.sea_power),1e-30)),
            "Pcu_W":s.copper_power.tolist(),
            "heat_input_W":s.heat_power,
            "stored_heat_J":s.energy,
            "Tsea_max_K":float(et.amb+s.temp[:et.count].max()),
            "Tpkg_K":(et.amb+s.temp[et.pkg]).tolist(),
            "Tcu_K":(et.amb+s.temp[et.cu]).tolist(),
        })
    fields=a.out.with_suffix(".npz")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        fields,
        time_s=times,
        sea_xyz_m=backend.points,
        sea_vol_m3=backend.vol,
        Tsea_K=np.stack([et.amb+s.temp[:et.count] for s in trace]),
        qsea_W_m3=np.stack([s.sea_q for s in trace]),
        Tpkg_K=np.stack([et.amb+s.temp[et.pkg] for s in trace]),
        Tcu_K=np.stack([et.amb+s.temp[et.cu] for s in trace]))
    report={
        "status":"Original BFZI v148 warm electrothermal prototype; thermal and EM numerical accuracy OPEN",
        "sample":a.sample,
        "em_backend":"independent AliceThunder/BFZI v148 Green/resolvent and local source correction",
        "sea_temperature_model":"cellwise sigma(T), sigma-weighted FV phi, cellwise local mass",
        "thermal_model":"original BFZI FV face connectivity, 3-D sea cells + 2 package + 2 copper nodes",
        "frequency_hz":backend.omega/(2*np.pi),
        "v148_isothermal_Zsea_relative_error":warm_rel,
        "v148_M4_archived_Zsea_relative_error":archive_rel,
        "sea_alpha_per_K":a.sea_alpha,
        "ambient_K":et.amb,
        "fields_npz":str(fields),
        "trace":rows
    }
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "out":str(a.out),
        "fields":str(fields),
        "seawater_cells":et.count,
        "steps":len(trace),
        "Z11_first_ohm":rows[0]["Z_ohm"][0][0],
        "Z11_last_ohm":rows[-1]["Z_ohm"][0][0],
        "v148_isothermal_Zsea_relative_error":warm_rel,
        "v148_M4_archive_relative_error":archive_rel,
        "max_sea_power_relative_gap":max(r["Psea_port_volume_relative_gap"] for r in rows),
        "sea_P_first_W":rows[0]["Psea_volume_W"],
        "max_temperature_K":max(r["Tsea_max_K"] for r in rows),
    },indent=2))


if __name__=="__main__":
    main()
