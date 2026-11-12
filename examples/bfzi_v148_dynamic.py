"""V2.1 transient electrothermal solver using the original BFZI v148 backend.

Usage:
  python run.py --mode bfzi-et-v148 --bfzi-root /path/to/BFZI --sample M4

The BFZI checkout is read, never modified. Original BFZI v148 geometry,
Green/resolvent, source mass and air inductance are the electromagnetic base.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_et import BFZIV148ET, circuit_power


def pair(x):
    z=np.asarray(x)
    return np.stack((z.real,z.imag),axis=-1).tolist()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True,
                   help="Path to existing AliceThunder/BFZI checkout")
    p.add_argument("--sample",choices=("M1","M2","M3","M4"),default="M4")
    p.add_argument("--voltage",type=float,default=3.)
    p.add_argument("--source-r",type=float,default=.5,help="transmitter source series resistance, ohm")
    p.add_argument("--load-r",type=float,default=.5,help="receiver real load, ohm")
    p.add_argument("--step",type=float,default=60.)
    p.add_argument("--heat",type=float,default=240.)
    p.add_argument("--cool",type=float,default=180.)
    p.add_argument("--restart",type=float,default=180.)
    p.add_argument("--sea-alpha",type=float,default=.02)
    p.add_argument("--check-only",action="store_true",
                   help="Verify original BFZI isothermal complex Z without thermal time stepping")
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_v148_dynamic.json"))
    a=p.parse_args(argv)
    if (not a.bfzi_root.is_dir() or
            not (a.bfzi_root/"bfzi_v148_results.py").exists()):
        p.error("bfzi-root must be an existing BFZI repository checkout")
    if (a.source_r <= 0 or a.load_r <= 0):
        p.error("source and load resistances must be positive")
    if min(a.voltage,a.step,a.heat,a.cool,a.restart) <= 0:
        p.error("drive voltage and times must be positive")
    rev=subprocess.run(
        ["git","-C",str(a.bfzi_root.resolve()),"rev-parse","HEAD"],
        capture_output=True,text=True,check=False)
    bfzi_sha=rev.stdout.strip() if rev.returncode==0 else None
    front="v150" if (a.bfzi_root/"run_v150.py").is_file() else "v148/v149"
    sys.path.insert(0,str(a.bfzi_root.resolve()))
    import bfzi_v148_results as calc

    data=(calc.build_circle_data("M4") if a.sample=="M4"
          else calc.base.build_dual_data(a.sample))
    backend=BFZIV148Adapter(data,calc,sea_alpha=a.sea_alpha,
                           sea_only=a.check_only)
    extra=np.diag([a.source_r-1j*backend.omega*backend.air[0,0],
                   a.load_r-1j*backend.omega*backend.air[1,1]])
    # Check the original v148 isothermal electromagnetic computation
    # before investing in any expensive repeated thermal time steps.
    cold=backend.solve(np.full(len(backend.vol),backend.ambient),
                       np.full(2,backend.ambient))
    z_ref,_,_,_,_=calc.base.formal_with_local_mass(data)
    warm_rel=float(np.linalg.norm(cold.sea_z-z_ref)
                   / max(np.linalg.norm(z_ref),1e-30))
    archive_rel=None
    if a.sample=="M4":
        import csv
        saved=a.bfzi_root/"outputs_v148"/"bfzi_v148_all_samples_matrix.csv"
        if saved.is_file():
            z_arch=np.zeros((2,2),complex)
            found=set()
            with saved.open(encoding="utf-8-sig",newline="") as inp:
                for row in csv.DictReader(inp):
                    if row["sample_id"]=="M4":
                        i,j=int(row["row"])-1,int(row["col"])-1
                        if i not in (0,1) or j not in (0,1):
                            raise ValueError("invalid M4 archive matrix indices")
                        z_arch[i,j]=(float(row["R_ohm"])
                            +1j*backend.omega*float(row["L_sea_reaction_uH"])*1e-6)
                        found.add((i,j))
            if len(found)==4:
                archive_rel=float(np.linalg.norm(z_ref-z_arch)
                                  / max(np.linalg.norm(z_arch),1e-30))
    original_pass=bool(warm_rel<=1e-5)
    archive_pass=bool(a.sample!="M4" or
                      (archive_rel is not None and archive_rel<=1e-5))
    baseline={
        "sample":a.sample,"frequency_hz":backend.omega/(2*np.pi),
        "bfzi_frontend":front,"bfzi_commit":bfzi_sha,
        "bfzi_em_solver":"bfzi_v148_results / bfzi_v142_results",
        "baseline_tolerance":1e-5,
        "v148_isothermal_Zsea_relative_error":warm_rel,
        "v148_M4_archived_Zsea_relative_error":archive_rel,
        "warm_Zsea_ohm":pair(cold.sea_z),
        "original_Zsea_ohm":pair(z_ref),
        "v148_original_solver_pass":original_pass,
        "v148_M4_archive_pass":archive_pass,
        "isothermal_baseline_pass":original_pass and archive_pass,
    }
    if a.check_only:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(baseline,indent=2),encoding="utf-8")
        print(json.dumps(baseline,indent=2))
        if not baseline["isothermal_baseline_pass"]:
            raise RuntimeError("BFZI v148 isothermal solver or M4 archive impedance mismatch")
        return
    if not baseline["isothermal_baseline_pass"]:
        raise RuntimeError("BFZI v148 isothermal solver or M4 archive impedance mismatch; use --check-only")
    et=BFZIV148ET(backend,external=extra)
    duration=a.heat+a.cool+a.restart
    times=np.unique(np.r_[np.arange(0,duration,a.step),
                           a.heat,a.heat+a.cool,duration])
    def drive(t):
        active=(t<=a.heat or t>a.heat+a.cool)
        return np.array([a.voltage if active else 0.,0.],complex)

    trace=et.trajectory(times,drive)
    rows=[]
    for k,s in enumerate(trace):
        v=drive(s.time)
        if k:
            dt=s.time-trace[k-1].time
            # Implicit FV heat step: storage + ambient dissipation = Joule heat.
            p_out=float(np.sum(et.H@s.temp))
            defect=(s.energy-trace[k-1].energy)+dt*(p_out-s.heat_power)
            scale=max(abs(s.energy-trace[k-1].energy),
                      dt*abs(p_out),dt*abs(s.heat_power),1e-30)
            heat_gap=float(abs(defect)/scale)
        else:
            p_out=0.
            heat_gap=None
        p_in,p_load,p_src,eta = circuit_power(
            v,s.currents,a.source_r,a.load_r)
        p_sea=float(.5*np.real(np.vdot(s.currents,s.sea_z@s.currents)))
        p_gap=p_in-p_load-p_src-p_sea-float(np.sum(s.copper_power))
        rows.append({
            "time_s":s.time,
            "drive":"on" if np.any(abs(s.currents)>0) else "off",
            "Z_ohm":pair(s.impedance),
            "Zsea_ohm":pair(s.sea_z),
            "Leff_H":(s.impedance.imag/backend.omega).tolist(),
            "I_A":pair(s.currents),
            "Psea_volume_W":s.sea_power,
            "Psea_port_W":p_sea,
            "circuit_power_relative_gap":float(
                abs(p_gap)/max(abs(p_in),1e-30)),
            "Psea_port_volume_relative_gap":float(
                abs(.5*np.real(np.vdot(s.currents,s.sea_z@s.currents))
                    -s.sea_power)/max(abs(s.sea_power),1e-30)),
            "Pcu_W":s.copper_power.tolist(),
            "Pinput_W":p_in,
            "Pload_W":p_load,
            "Psource_loss_W":p_src,
            "efficiency":eta,
            "heat_input_W":s.heat_power,
            "stored_heat_J":s.energy,
            "ambient_cooling_W":p_out,
            "thermal_balance_relative_gap":heat_gap,
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
        "bfzi_frontend":front,"bfzi_commit":bfzi_sha,
        "em_backend":"independent AliceThunder/BFZI v148 Green/resolvent and local source correction",
        "sea_temperature_model":"cellwise sigma(T), sigma-weighted FV phi, cellwise local mass",
        "thermal_model":"original BFZI FV face connectivity, 3-D sea cells + 2 package + 2 copper nodes",
        "frequency_hz":backend.omega/(2*np.pi),
        "isothermal_regression":baseline,
        "sea_alpha_per_K":a.sea_alpha,
        "source_resistance_ohm":a.source_r,
        "load_resistance_ohm":a.load_r,
        "external_Z_ohm":pair(extra),
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
        "max_circuit_power_relative_gap":max(r["circuit_power_relative_gap"] for r in rows),
        "max_thermal_balance_relative_gap":max(
            r["thermal_balance_relative_gap"] for r in rows[1:]),
        "efficiency_first":rows[0]["efficiency"],
        "efficiency_last":rows[-1]["efficiency"],
        "sea_P_first_W":rows[0]["Psea_volume_W"],
        "max_temperature_K":max(r["Tsea_max_K"] for r in rows),
    },indent=2))


if __name__=="__main__":
    main()
