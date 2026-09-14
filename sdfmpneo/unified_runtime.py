"""Runtime for the geometry-to-tensor + geometry-aware thermal ROM path."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import uuid

import numpy as np

from .unified_geometry import UnifiedUWPTGeometry, sample_geometry
from .unified_model import UnifiedNeuralElectroThermalModel
from .unified_open_boundary import OpenBoundaryBackground
from .unified_tensor_surrogate import (
    TensorDataset,
    encode_geometry,
    generate_tensor_dataset,
    solve_port_truth_tensors,
)
from .unified_tensor_training import train_matrix_tensor_surrogate
from .unified_thermal import GeometryAwareThermalLibrary, build_geometry_aware_thermal_library

_CACHE_FORMAT = 12


def jsonable(value):
    if is_dataclass(value): return jsonable(asdict(value))
    if isinstance(value,np.ndarray):
        if np.iscomplexobj(value): return {"real":value.real.tolist(),"imag":value.imag.tolist()}
        return value.tolist()
    if isinstance(value,np.generic):
        if np.iscomplexobj(value): return {"real":float(np.real(value)),"imag":float(np.imag(value))}
        return value.item()
    if isinstance(value,complex): return {"real":float(value.real),"imag":float(value.imag)}
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    return value


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _progress(message,percent,monitor=None):
    print(f"{message}……{float(percent):.0f}%",flush=True)
    if monitor is not None:
        with monitor._lock: monitor.data.update(progress_percent=float(percent),progress_message=str(message))


def _signature(settings):
    keys=("BACKGROUND","DEFAULT_GEOMETRY","GEOMETRY_SAMPLING","PHYSICS","MATERIALS","REGIONS","TRAINING")
    payload={k:settings[k] for k in keys}
    payload["TRAINING"]={k:v for k,v in payload["TRAINING"].items() if k not in {"network","optimizer","device"}}
    payload["cache_format"]=_CACHE_FORMAT
    text=json.dumps(jsonable(payload),sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def build_background(settings,*,bounds=None):
    regions=settings["REGIONS"]; cfg=dict(settings["BACKGROUND"])
    if bounds is not None: cfg["bounds"]=np.asarray(bounds,float).tolist()
    return OpenBoundaryBackground.from_config(
        cfg,frequency_hz=settings["PHYSICS"]["frequency_hz"],materials=settings["MATERIALS"],
        coil_materials=regions["coil_materials"],package_materials=regions["package_materials"],
        seawater_material=regions["seawater_material"],ambient_temperature=settings["PHYSICS"]["ambient_temperature"],
    )


def _sample_geometries(settings,n,rng,background):
    out=[]; attempts=0
    while len(out)<int(n):
        attempts+=1
        if attempts>100*max(1,int(n)):
            raise ValueError("geometry sampling produced too many invalid physical geometries; adjust sampling ranges or BACKGROUND bounds")
        candidate=sample_geometry(settings["DEFAULT_GEOMETRY"],settings.get("GEOMETRY_SAMPLING"),rng)
        try:
            background.validate_geometry(UnifiedUWPTGeometry.from_mapping(candidate)); encode_geometry(candidate)
        except ValueError: continue
        out.append(candidate)
    return out


def _cache_paths(directory):
    return directory/"unified.cache.json",directory/"unified.geometry_thermal.npz",directory/"unified.tensor_dataset.npz"


def _require_effective_thermal_basis(report):
    get=report.get if isinstance(report,dict) else lambda name,default=None:getattr(report,name,default)
    trajectory=get("maximum_validation_trajectory_relative_error",None)
    if trajectory is None:
        raise RuntimeError("geometry-aware thermal cache lacks the required held-out trajectory audit; regenerate training cache")
    if bool(get("converged",False)): return
    energy=get("maximum_validation_relative_energy_error",None)
    if energy is None or float(energy)==0.0: energy=get("maximum_anchor_relative_energy_error",float("nan"))
    target=get("target_relative_error",float("nan")); rank=int(get("basis_dimension",0)); reason=str(get("stop_reason","unknown"))
    worst=(
        get("worst_validation_trajectory",{})
        or get("worst_validation_anchor",{})
        or get("worst_training_anchor",{})
        or {}
    )
    raise RuntimeError(
        f"geometry-aware thermal ROM 未达到训练要求：rank={rank}, energy error={float(energy):.3e}, "
        f"trajectory error={float(trajectory):.3e}, target={float(target):.3e}, stop={reason}, "
        f"worst={json.dumps(jsonable(worst),ensure_ascii=False,sort_keys=True)}。"
    )


def _relative_error(value,reference):
    a=np.asarray(value); b=np.asarray(reference)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))


def _open_boundary_convergence(settings,background,geometries,monitor=None):
    cfg=dict(settings["BACKGROUND"].get("open_boundary_check",{})); tolerance=float(cfg.get("relative_tolerance",5e-2))
    padding=np.asarray(cfg.get("padding",0.12),float)
    if padding.ndim==0: padding=np.full(3,float(padding))
    if padding.shape!=(3,) or np.any(padding<=0.0): raise ValueError("open_boundary_check.padding must be a positive scalar or length-3 vector")
    if not 0.0<tolerance<1.0: raise ValueError("open_boundary_check.relative_tolerance must lie in (0, 1)")
    bounds=np.asarray(settings["BACKGROUND"]["bounds"],float); reference_bounds=bounds.copy(); reference_bounds[:,0]-=padding; reference_bounds[:,1]+=padding
    reference=build_background(settings,bounds=reference_bounds); rows=[]
    for index,geometry in enumerate(geometries):
        if monitor is not None: monitor.checkpoint()
        z_base,_,_,audit_base=solve_port_truth_tensors(background,geometry); z_ref,_,_,audit_ref=solve_port_truth_tensors(reference,geometry)
        total=_relative_error(z_base,z_ref); real=_relative_error(z_base.real,z_ref.real); imag=_relative_error(z_base.imag,z_ref.imag)
        rows.append({"index":index,"relative_z_error":total,"relative_resistive_error":real,"relative_reactive_error":imag,
                     "base_power_balance_error":audit_base["open_boundary_power_balance_relative_error"],
                     "reference_power_balance_error":audit_ref["open_boundary_power_balance_relative_error"]})
        print(f"开放边界域扩展检查……{index+1}/{len(geometries)}  Z={total:.3e} ReZ={real:.3e} ImZ={imag:.3e}",flush=True)
    worst_total=max(row["relative_z_error"] for row in rows); worst_real=max(row["relative_resistive_error"] for row in rows); worst_imag=max(row["relative_reactive_error"] for row in rows)
    return {"sample_count":len(rows),"padding":padding.tolist(),"relative_tolerance":tolerance,
            "maximum_relative_z_error":worst_total,"maximum_relative_resistive_error":worst_real,
            "maximum_relative_reactive_error":worst_imag,"converged":bool(max(worst_total,worst_real,worst_imag)<=tolerance),"samples":rows}


def _physics_gate(dataset,open_boundary_convergence):
    audit=dict(dataset.audit)
    checks={
        "linear_solve_ok":audit["maximum_linear_relative_residual"]<=1e-8,
        "reciprocity_ok":audit["maximum_reciprocity_relative_error"]<=1e-8,
        "volume_passivity_ok":audit["minimum_d_vol_eigenvalue"]>=-1e-9,
        "physical_outward_passivity_ok":audit["minimum_physical_outward_eigenvalue"]>=-1e-9,
        "implied_outward_passivity_ok":audit["minimum_implied_outward_eigenvalue"]>=-1e-9,
        "independent_poynting_balance_ok":audit["maximum_open_boundary_power_balance_relative_error"]<=1e-7,
        "modal_loewner_ok":audit["maximum_relative_loewner_violation"]<=1e-8,
        "outward_power_form_available":bool(audit["independent_outward_power_available"]>=0.5),
        "open_boundary_domain_converged":bool(open_boundary_convergence["converged"]),
    }
    certified=all(bool(v) for v in checks.values())
    return {**{k:bool(v) for k,v in checks.items()},
            "internal_truth_gate_passed":bool(all(checks[k] for k in checks if k!="open_boundary_domain_converged")),
            "reaction_impedance_convention":"negative_source_reaction","open_boundary_verified":bool(certified),
            "independent_outward_power_verified":bool(checks["outward_power_form_available"] and checks["independent_poynting_balance_ok"]),
            "boundary_model":"silver_muller_impedance","open_boundary_convergence":open_boundary_convergence,
            "certified":bool(certified),"status":"certified" if certified else "physics_gate_failed","audit":audit}


def train(settings,model_path,settings_dir,monitor=None):
    from .training.monitor import TrainingStopped
    settings_dir.mkdir(parents=True,exist_ok=True); sig=_signature(settings); meta_path,thermal_path,data_path=_cache_paths(settings_dir)
    checkpoint=Path(settings["FILES"]["training_checkpoint"]); checkpoint=checkpoint if checkpoint.is_absolute() else Path(settings["ROOT"])/checkpoint
    try:
        _progress("构建开放边界固定背景物理空间",0,monitor); bg=build_background(settings); _progress("构建开放边界固定背景物理空间",6,monitor)
        print(f"背景空间：{bg.n_cells} cells，{bg.n_edges} Maxwell edge DOFs；Maxwell 仅用于离线 open-boundary truth。",flush=True)
        valid_cache=False; cache_meta={}
        if meta_path.is_file() and thermal_path.is_file() and data_path.is_file():
            try:
                cache_meta=json.loads(meta_path.read_text(encoding="utf-8")); valid_cache=cache_meta.get("signature")==sig and int(cache_meta.get("cache_format",-1))==_CACHE_FORMAT
            except (OSError,ValueError,TypeError): valid_cache=False

        if valid_cache:
            _progress("复用 geometry-aware thermal library 与 tensor truth 数据",35,monitor)
            thermal_report=cache_meta.get("thermal_basis_report",{}); _require_effective_thermal_basis(thermal_report)
            library=GeometryAwareThermalLibrary.load(thermal_path); bg.set_thermal_library(library); dataset=TensorDataset.load(data_path)
            if dataset.thermal_rank!=bg.thermal_rank: raise RuntimeError("cached tensor dataset thermal rank 与 geometry-aware library 不一致")
        else:
            checkpoint.unlink(missing_ok=True); rng=np.random.default_rng(int(settings["TRAINING"].get("seed",17)))
            n_basis=int(settings["TRAINING"].get("basis_samples",8)); n_basis_val=int(settings["TRAINING"].get("basis_validation_samples",6))
            basis_geometries=_sample_geometries(settings,n_basis,rng,bg); validation_geometries=_sample_geometries(settings,n_basis_val,rng,bg)
            _progress("构建 geometry-aware canonical thermal ROM",8,monitor)
            library,thermal_obj=build_geometry_aware_thermal_library(
                bg,settings["DEFAULT_GEOMETRY"],basis_geometries,validation_geometries=validation_geometries,
                target_relative_error=float(settings["TRAINING"].get("thermal_basis_energy_tolerance",5e-2)),
                time_scales=settings["TRAINING"].get("thermal_time_scales",[0.1,1.0,10.0]),
                trajectory_times=settings["TRAINING"].get("thermal_trajectory_times"),
                maximum_rank=settings["TRAINING"].get("thermal_basis_max_rank"),
                conditioning_limit=float(settings["TRAINING"].get("thermal_basis_conditioning_limit",1e10)),monitor=monitor,
            )
            _require_effective_thermal_basis(thermal_obj); bg.set_thermal_library(library); library.save(thermal_path); thermal_report=asdict(thermal_obj)
            _progress("构建 geometry-aware canonical thermal ROM",28,monitor)
            n_tensor=int(settings["TRAINING"].get("n_tensor_samples",96)); tensor_geometries=_sample_geometries(settings,n_tensor,rng,bg)
            dataset=generate_tensor_dataset(bg,tensor_geometries,seed=int(settings["TRAINING"].get("seed",17)),monitor=monitor); dataset.save(data_path)
            write_json(meta_path,{"cache_format":_CACHE_FORMAT,"signature":sig,"thermal_basis_report":thermal_report,
                                  "thermal_representation":"geometry_aware_bg_tx_rx_canonical_modes","em_representation":"geometry_to_port_and_joule_tensors","em_boundary":"silver_muller_impedance"})
            _progress("生成几何 tensor truth 数据",50,monitor)

        gate_cfg=dict(settings["BACKGROUND"].get("open_boundary_check",{})); gate_count=max(1,int(gate_cfg.get("samples",3)))
        gate_rng=np.random.default_rng(int(settings["TRAINING"].get("seed",17))+104729); gate_geometries=_sample_geometries(settings,gate_count,gate_rng,bg)
        _progress("验证开放边界与独立 Poynting 功率",51,monitor); boundary_convergence=_open_boundary_convergence(settings,bg,gate_geometries,monitor=monitor); gate=_physics_gate(dataset,boundary_convergence)
        if not gate["certified"]: raise RuntimeError("Physics Gate failed; refusing surrogate training: "+json.dumps(jsonable(gate),sort_keys=True))
        print("Physics Gate：linear residual / reciprocity / independent Poynting / open-boundary convergence / passivity / modal Loewner 全部通过。",flush=True)

        _progress("训练 geometry→tensor POD-MLP",55,monitor)
        surrogate,report=train_matrix_tensor_surrogate(
            dataset,network_settings=settings["TRAINING"].get("network"),training_settings=settings["TRAINING"].get("optimizer"),
            device=settings["TRAINING"].get("device","cuda"),monitor=monitor,checkpoint_path=checkpoint,
        )
        model=UnifiedNeuralElectroThermalModel(bg,surrogate,default_geometry=settings["DEFAULT_GEOMETRY"],
            current_offset=settings["PORTS"].get("current_offset"),current_matrix=settings["PORTS"].get("current_matrix"),production_domain=settings.get("GEOMETRY_SAMPLING"))
        _progress("保存统一 geometry-aware tensor-ROM 模型",98,monitor)
        model.save(model_path,metadata={"thermal_basis_report":thermal_report,"physics_gate":gate,"training_report":asdict(report)})
        checkpoint.unlink(missing_ok=True)
        write_json(settings_dir/"training.report.json",{"model":str(model_path),"background_cells":bg.n_cells,"offline_maxwell_dofs":bg.n_edges,
            "online_em_representation":"Z_field + D_vol + geometry-dependent modal H_j","thermal_basis_rank":bg.thermal_rank,
            "thermal_representation":"deterministic Phi(g) = BG + transported local blocks","thermal_basis":thermal_report,"physics_gate":gate,"training":report})
        _progress("训练完成",100,monitor)
        print(f"训练完成：geometry-aware thermal rank={bg.thermal_rank}，tensor POD rank={report.pod_rank}，在线 Maxwell solve=0，best epoch={report.best_epoch}，validation matrix loss={report.best_validation_loss:.6g}，test relative tensor error={report.test_relative_tensor_error:.6g}。",flush=True)
        print(f"模型已保存：{model_path}",flush=True)
        if monitor is not None: monitor.finish("completed",model=str(model_path))
        return 0
    except TrainingStopped:
        if monitor is not None: monitor.finish("stopped",checkpoint=str(checkpoint) if checkpoint.is_file() else None)
        print(f"训练已停止；tensor MLP 检查点：{checkpoint}" if checkpoint.is_file() else "训练已停止。",flush=True); return 130


def _device(requested):
    value=str(requested)
    if value.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available(): return "cpu"
        except ImportError: return "cpu"
    return value


def _initial_state(model,prediction,geometry):
    value=prediction.get("initial_temperature_rise",0.0)
    if value is None or (isinstance(value,str) and value.lower()=="ambient"): return np.zeros(model.thermal_rank)
    array=np.asarray(value,float)
    if array.ndim==0 and float(array)==0.0: return np.zeros(model.thermal_rank)
    return model.project_initial_temperature(array,geometry)


def predict(settings,model_path,output_path,settings_dir):
    if not model_path.is_file(): raise FileNotFoundError(f"模型不存在：{model_path}")
    device=_device(settings["TRAINING"].get("device","cuda")); model=UnifiedNeuralElectroThermalModel.load(model_path,device=device)
    p=settings["PREDICTION"]; geometry=p.get("geometry") or settings["DEFAULT_GEOMETRY"]; initial=_initial_state(model,p,geometry)
    operating=p.get("drive",p.get("operating",[1.0]+[0.0]*(model.current_dimension-1))); results=[]; tensors=model.tensors(geometry)
    print(f"加载 geometry-aware tensor electrothermal ROM：{model_path}  device={device}  thermal rank={model.thermal_rank}  POD={model.surrogate.pod_rank}  tensor projection correction={tensors.projection_correction:.3e}",flush=True)
    for requested in p["times"]:
        if isinstance(requested,str) and requested.lower()=="inf":
            result=model.steady_state(initial_guess=initial,geometry=geometry,operating=operating,tolerance=float(p.get("steady_tolerance",1e-10)),max_iterations=int(p.get("steady_max_iterations",40)))
            print(f"t=inf，Tmax={result.maximum_temperature:.6g} K，thermal residual={result.residual_norm:.3e}，stable={result.stable}，spectral_abscissa={result.spectral_abscissa:.3e}，Pvol={result.volume_power:.6g} W，Pwire={result.wire_power:.6g} W",flush=True)
            results.append({"time":"inf","steady_state":result})
        else:
            t=float(requested); result=model.predict(t,initial_state=initial,geometry=geometry,operating=operating,max_step=float(p.get("max_step",100)),method=p.get("method","etd2_adaptive"),rtol=float(p.get("rtol",1e-5)),atol=float(p.get("atol",1e-8)),initial_step=p.get("initial_step"))
            print(f"t={t:g}s，Tmax={result.maximum_temperature:.6g} K，steps={result.steps}，Pvol={result.volume_power:.6g} W，Pwire={result.wire_power:.6g} W",flush=True)
            results.append({"time":t,"prediction":result})
    write_json(output_path,{"model":str(model_path),"geometry":geometry,"operating":operating,"results":results}); write_json(settings_dir/"predict.settings.json",{"model":str(model_path),"prediction":p})
    print(f"推理结果已保存：{output_path}",flush=True); return 0


def _worker_from_file(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8")); wrapper=payload["settings"]
    return execute_training(wrapper["parameters"],Path(wrapper["model_path"]),Path(wrapper["settings_dir"]),Path(payload["session_dir"]))


def launch(settings,argv=None):
    parser=argparse.ArgumentParser(description="统一 geometry→tensor + geometry-aware thermal ROM（在线无 Maxwell/FGMRES）")
    parser.add_argument("--mode",choices=("train","predict"),default=settings.get("MODE","train")); parser.add_argument("--model")
    group=parser.add_mutually_exclusive_group(); group.add_argument("--gui",action="store_true"); group.add_argument("--headless",action="store_true")
    parser.add_argument("--worker-config",help=argparse.SUPPRESS); args=parser.parse_args(argv)
    if args.worker_config: return _worker_from_file(args.worker_config)
    root=Path(settings["ROOT"]); model_path=Path(args.model or settings["FILES"]["model"]); model_path=model_path if model_path.is_absolute() else root/model_path
    settings_dir=Path(settings["FILES"]["settings_dir"]); settings_dir=settings_dir if settings_dir.is_absolute() else root/settings_dir; settings_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=="predict":
        output=Path(settings["FILES"]["predictions"]); output=output if output.is_absolute() else root/output
        return predict(settings,model_path,output,settings_dir)
    if not args.headless and (args.gui or settings["MONITOR"].get("enabled",True)):
        from .training.qt_monitor import launch_window
        log_root=Path(settings["MONITOR"]["log_dir"]); log_root=log_root if log_root.is_absolute() else root/log_root
        worker_settings=jsonable({"root":settings["ROOT"],"model_path":str(model_path),"settings_dir":str(settings_dir),"parameters":settings})
        return launch_window(root/"run.py",worker_settings,log_root,settings["MONITOR"])
    return execute_training(settings,model_path,settings_dir)


def execute_training(settings,model_path,settings_dir,session_dir=None):
    from .training.monitor import TrainingMonitor
    if session_dir is None:
        log_root=Path(settings["MONITOR"]["log_dir"]); log_root=log_root if log_root.is_absolute() else Path(settings["ROOT"])/log_root
        session_dir=log_root/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    session_dir=Path(session_dir); session_dir.mkdir(parents=True,exist_ok=True); write_json(session_dir/"settings.json",settings)
    with TrainingMonitor(session_dir/"metrics.jsonl",session_dir/"control.json",interval=float(settings["MONITOR"].get("log_interval_s",1.0))) as monitor:
        return train(settings,Path(model_path),Path(settings_dir),monitor)


__all__=["build_background","execute_training","jsonable","launch","predict","train","write_json"]
