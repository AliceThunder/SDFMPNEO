"""V2.1: evolving two-port BFZI impedance, spatial seawater and copper heat.

Run: python run.py --mode bfzi-et-dynamic --out runs/bfzi_et_dynamic.json
Uses the existing BFZI Green-Galerkin R/L/K response as the electromagnetic
prototype. Original independent BFZI v148 kernel integration remains next.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.source import build_bfzi_source
from sdfmpneo_bfzi.sea_reference import SeaMQSConfig, solve_seawater_reference
from sdfmpneo_bfzi.dynamic_et import BFZIDynamicET


def pairs(z):
    a = np.asarray(z)
    return np.stack((a.real,a.imag),axis=-1).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frequency",type=float,default=200000.)
    ap.add_argument("--voltage",type=float,default=12.)
    ap.add_argument("--step",type=float,default=60.,help="thermal step in seconds")
    ap.add_argument("--heat",type=float,default=240.,help="first powered duration, s")
    ap.add_argument("--cool",type=float,default=180.,help="powered-off duration, s")
    ap.add_argument("--restart",type=float,default=180.,help="restart duration, s")
    ap.add_argument("--sea-step",type=float,default=.16,help="3-D seawater heat-cell width, m")
    ap.add_argument("--sea-alpha",type=float,default=.02,help="local seawater conductivity coefficient, 1/K")
    ap.add_argument("--radial",type=int,default=5)
    ap.add_argument("--polar",type=int,default=7)
    ap.add_argument("--azimuthal",type=int,default=10)
    ap.add_argument("--out",type=Path,default=Path("runs/bfzi_et_dynamic.json"))
    args = ap.parse_args(argv)
    if min(args.step,args.heat,args.cool,args.restart,args.frequency) <= 0:
        ap.error("frequency and all stage durations/step must be positive")
    scene = build_demo_scene()
    sources = build_bfzi_source(scene,args.frequency,order=4)
    ref = solve_seawater_reference(sources,config=SeaMQSConfig(
        radial_order=args.radial,polar_order=args.polar,
        azimuthal_order=args.azimuthal,field_softening=.025))
    # One transmitter voltage source, one passive loaded receiver; fixed
    # series compensation cancels the reference air self-inductance.
    ext = np.diag([.5-1j*sources.omega*sources.inductance_air[k,k]
                   for k in range(2)])
    et = BFZIDynamicET(ref,sea_step=args.sea_step,
                       sea_alpha=args.sea_alpha,external=ext)
    total = args.heat+args.cool+args.restart
    t = np.r_[np.arange(0,total,args.step),total]
    def voltage(now):
        on = now <= args.heat or now > args.heat+args.cool
        return np.asarray((args.voltage if on else 0.,0.),complex)

    trace = et.trajectory(t,voltage)
    report = {
        "status":"V2.1 deterministic time-coupled thermal prototype; physical reference OPEN",
        "em_backend":"SDFMPNEO BFZI Green-Galerkin R/L/K with spatial sigma(T)",
        "independent_bfzi_v148_backend":"not yet attached",
        "heat_backend":"3D seawater thermal cells + two package and two copper heat states",
        "frequency_hz":args.frequency,
        "sea_alpha_per_K":args.sea_alpha,
        "ambient_K":et.amb,
        "voltage_on_V_peak":args.voltage,
        "heat_stage_s":args.heat,
        "cool_stage_s":args.cool,
        "restart_stage_s":args.restart,
        "sea_cell_centers_m":et.centers.tolist(),
        "sea_cell_volumes_m3":et.vol.tolist(),
        "external_Z_ohm":pairs(ext),
        "trace":[{
            "time_s":float(s.time),
            "stage":"on" if np.any(abs(s.currents)>0) else "off",
            "Z_ohm":pairs(s.impedance),
            "Zsea_ohm":pairs(s.sea_z),
            "L_eff_H":(s.impedance.imag/sources.omega).tolist(),
            "I_A":pairs(s.currents),
            "sea_P_W":s.sea_power,
            "cu_P_W":s.copper_power.tolist(),
            "thermal_input_W":s.heat_power,
            "stored_heat_J":s.energy,
            "Tsea_cell_K":(s.temp[:et.count]+et.amb).tolist(),
            "qsea_cell_W_m3":s.sea_q.tolist(),
            "Tpkg_K":(s.temp[et.pkg]+et.amb).tolist(),
            "Tcu_K":(s.temp[et.cu]+et.amb).tolist(),
            "max_temperature_K":float(et.amb+s.temp.max()),
        } for s in trace],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    brief=[{"time_s":r["time_s"],"stage":r["stage"],
            "max_T_K":r["max_temperature_K"],
            "Z11_ohm":r["Z_ohm"][0][0],
            "Z12_ohm":r["Z_ohm"][0][1],
            "sea_P_W":r["sea_P_W"]} for r in report["trace"]]
    print(json.dumps({"file":str(args.out),"thermal_nodes":et.n,
                      "steps":brief},indent=2))


if __name__=="__main__":
    main()
