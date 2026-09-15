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

from .unified_corrected_final_audit import run_final_held_out_audit
from .unified_geometry import UnifiedUWPTGeometry, sample_geometry
from .unified_model import UnifiedNeuralElectroThermalModel
from .unified_open_boundary import OpenBoundaryBackground
from .unified_corrected_physics_gate import run_physics_gate
from .unified_corrected_truth import generate_tensor_dataset
from .unified_tensor_surrogate import TensorDataset, encode_geometry
from .unified_tensor_training import train_matrix_tensor_surrogate
from .unified_thermal import GeometryAwareThermalLibrary, build_geometry_aware_thermal_library
from .unified_corrected_truth_preflight import run_truth_preflight

_CACHE_FORMAT = 18
_SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"


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
    payload["TRAINING"]={
        k:v for k,v in payload["TRAINING"].items()
        if k not in {"network","optimizer","device","final_audit"}
    }
    payload["cache_format"]=_CACHE_FORMAT
    text=json.dumps(jsonable(payload),sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def build_background(settings,*,bounds=None):
    regions=settings["REGIONS"]; cfg=dict(settings["BACKGROUND"])
    if bounds is not None: cfg["bounds"]=np.asarray(bounds,float).tolist()
    bg=OpenBoundaryBackground.from_config(
        cfg,frequency_hz=settings["PHYSICS"]["frequency_hz"],materials=settings["MATERIALS"],
        coil_materials=regions["coil_materials"],package_materials=regions["package_materials"],
        seawater_material=regions["seawater_material"],ambient_temperature=settings["PHYSICS"]["ambient_temperature"],
    )
    bg.background_config=dict(cfg)
    bg.self_correction_config=dict(cfg.get("self_correction",{}))
    return bg


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


def _require_release_artifact(path):
    try:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data["metadata_json"]))
        release=dict(meta.get("metadata",{}))
        preflight=dict(release.get("truth_preflight",{})); gate=dict(release.get("physics_gate",{})); final=dict(release.get("final_held_out_audit",{}))
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise ValueError("model artifact lacks readable production release metadata") from exc
    checks={
        "truth_preflight":bool(preflight.get("certified",False)),
        "local_self_correction":bool(preflight.get("local_self_correction_converged",False)),
        "physics_gate":bool(gate.get("certified",False)),
        "self_correction_model":gate.get("self_correction_model")==_SELF_CORRECTION_MODEL,
        "final_held_out_audit":bool(final.get("certified",False)),
        "production_integrator":bool(final.get("production_integrator_ok",False)),
        "certificate_level":final.get("certificate_level")=="frozen_held_out_numerical_validation",
    }
    if not all(checks.values()):
        failed=[name for name,value in checks.items() if not value]
        raise ValueError("model artifact is not a certified production release: "+", ".join(failed))
    return release


def _require_effective_thermal_basis(report):
    get=report.get if isinstance(report,dict) else lambda name,default=None:getattr(report,name,default)
    trajectory=get("maximum_validation_trajectory_relative_error",None)
    if trajectory is None: raise RuntimeError("geometry-aware thermal cache lacks the required held-out trajectory audit; regenerate training cache")
    if bool(get("converged",False)): return
    energy=get("maximum_validation_relative_energy_error",None)
    if energy is None or float(energy)==0.0: energy=get("maximum_anchor_relative_energy_error",float("nan"))
    target=get("target_relative_error",float("nan")); rank=int(get("basis_dimension",0)); reason=str(get("stop_reason","unknown"))
    worst=(get("worst_validation_trajectory",{}) or get("worst_validation_anchor",{}) or get("worst_training_anchor",{}) or {})
    raise RuntimeError(f"geometry-aware thermal ROM 未达到训练要求：rank={rank}, energy error={float(energy):.3e}, trajectory error={float(trajectory):.3e}, target={float(target):.3e}, stop={reason}, worst={json.dumps(jsonable(worst),ensure_ascii=False,sort_keys=True)}。")


def _gate_sample_count(settings):
    background=settings["BACKGROUND"]
    sections=("open_boundary_check","formulation_check","mesh_check","geometry_continuity_check","self_correction")
    return max(1,*[int(dict(background.get(name,{})).get("samples",1)) for name in sections])


def train(settings,model_path,settings_dir,monitor=None):
    from .training.monitor import TrainingStopped
    settings_dir.mkdir(parents=True,exist_ok=True); sig=_signature(settings); meta_path,thermal_path,data_path=_cache_paths(settings_dir)
    checkpoint=Path(settings["FILES"]["training_checkpoint"]); checkpoint=checkpoint if checkpoint.is_absolute() else Path(settings["ROOT"])/checkpoint
    try:
        _progress("构建开放边界固定背景物理空间",0,monitor); bg=build_background(settings); _progress("构建开放边界固定背景物理空间",5,monitor)
        print(f"背景空间：{bg.n_cells} cells，{bg.n_edges} Maxwell edge DOFs；Maxwell 仅用于离线 open-boundary truth。",flush=True)
        seed=int(settings["TRAINING"].get("seed",17)); valid_cache=False; cache_meta={}
        if meta_path.is_file() and thermal_path.is_file() and data_path.is_file():
            try:
                cache_meta=json.loads(meta_path.read_text(encoding="utf-8"))
                cached_preflight=dict(cache_meta.get("truth_preflight",{}))
                valid_cache=(
                    cache_meta.get("signature")==sig
                    and int(cache_meta.get("cache_format",-1))==_CACHE_FORMAT
                    and bool(cached_preflight.get("certified",False))
                    and bool(cached_preflight.get("local_self_correction_converged",False))
                    and cache_meta.get("self_correction_model")==_SELF_CORRECTION_MODEL
                )
            except (OSError,ValueError,TypeError): valid_cache=False
        if valid_cache:
            preflight=dict(cache_meta["truth_preflight"]); _progress("复用已认证 pre-basis spatial truth preflight",6,monitor)
            print("Truth preflight：复用当前 physical-cache signature 下已通过的验证结果。",flush=True)
        else:
            preflight_rng=np.random.default_rng(seed+65537); preflight_geometries=_sample_geometries(settings,_gate_sample_count(settings),preflight_rng,bg)
            _progress("执行 pre-basis spatial truth preflight",6,monitor); preflight=run_truth_preflight(settings,bg,preflight_geometries,monitor=monitor)
            if not preflight["certified"]: raise RuntimeError("Spatial truth preflight failed; refusing thermal-basis construction: "+json.dumps(jsonable(preflight),sort_keys=True))
            print("Truth preflight：finite-support source / terminal continuity / loss partition / open-domain / MQS / local-self reference / corrected EM mesh convergence 全部通过。",flush=True)
        if valid_cache:
            _progress("复用 geometry-aware thermal library 与 tensor truth 数据",38,monitor)
            thermal_report=cache_meta.get("thermal_basis_report",{}); _require_effective_thermal_basis(thermal_report)
            library=GeometryAwareThermalLibrary.load(thermal_path); bg.set_thermal_library(library); dataset=TensorDataset.load(data_path)
            if dataset.thermal_rank!=bg.thermal_rank: raise RuntimeError("cached tensor dataset thermal rank 与 geometry-aware library 不一致")
        else:
            checkpoint.unlink(missing_ok=True); rng=np.random.default_rng(seed)
            n_basis=int(settings["TRAINING"].get("basis_samples",8)); n_basis_val=int(settings["TRAINING"].get("basis_validation_samples",6))
            basis_geometries=_sample_geometries(settings,n_basis,rng,bg); validation_geometries=_sample_geometries(settings,n_basis_val,rng,bg)
            _progress("构建 geometry-aware canonical thermal ROM",12,monitor)
            library,thermal_obj=build_geometry_aware_thermal_library(bg,settings["DEFAULT_GEOMETRY"],basis_geometries,validation_geometries=validation_geometries,
                target_relative_error=float(settings["TRAINING"].get("thermal_basis_energy_tolerance",5e-2)),time_scales=settings["TRAINING"].get("thermal_time_scales",[0.1,1.0,10.0]),
                trajectory_times=settings["TRAINING"].get("thermal_trajectory_times"),maximum_rank=settings["TRAINING"].get("thermal_basis_max_rank"),
                conditioning_limit=float(settings["TRAINING"].get("thermal_basis_conditioning_limit",1e10)),monitor=monitor)
            _require_effective_thermal_basis(thermal_obj); bg.set_thermal_library(library); library.save(thermal_path); thermal_report=asdict(thermal_obj)
            _progress("生成 matrix-tensor truth 数据集",38,monitor)
            n_train=int(settings["TRAINING"].get("samples",40)); geometries=_sample_geometries(settings,n_train,rng,bg)
            dataset=generate_tensor_dataset(bg,settings["DEFAULT_GEOMETRY"],geometries,settings["PHYSICS"]["frequencies"],monitor=monitor)
            dataset.save(data_path)
            cache_meta={"signature":sig,"cache_format":_CACHE_FORMAT,"self_correction_model":_SELF_CORRECTION_MODEL,"truth_preflight":preflight,"thermal_basis_report":thermal_report}
            write_json(meta_path,cache_meta)

        _progress("训练 geometry-to-tensor surrogate",55,monitor)
        network_cfg=settings["TRAINING"].get("network",{}); report=train_matrix_tensor_surrogate(dataset,network_cfg,monitor=monitor)
        _progress("执行 physics Gate",80,monitor); gate=run_physics_gate(settings,bg,dataset,report,monitor=monitor)
        if not gate["certified"]: raise RuntimeError("Physics Gate failed; refusing final held-out audit: "+json.dumps(jsonable(gate),sort_keys=True))
        _progress("执行 frozen held-out final audit",92,monitor); final=run_final_held_out_audit(settings,bg,dataset,report,gate,monitor=monitor)
        if not final["certified"]: raise RuntimeError("Final held-out audit failed; refusing release artifact: "+json.dumps(jsonable(final),sort_keys=True))
        release={"truth_preflight":preflight,"physics_gate":gate,"final_held_out_audit":final,"training_report":asdict(report),"thermal_basis_report":thermal_report,"self_correction_model":_SELF_CORRECTION_MODEL}
        model=UnifiedNeuralElectroThermalModel.from_training(bg,settings["DEFAULT_GEOMETRY"],dataset,report,release_metadata=release)
        model.save(model_path)
        _progress("训练完成并保存模型",100,monitor)
        return 0
    except TrainingStopped:
        print("训练已停止。",flush=True); return 130


def _worker_from_file(config_path):
    payload=json.loads(Path(config_path).read_text(encoding="utf-8")); wrapper=payload["payload"]; return execute_training(wrapper["parameters"],Path(wrapper["model_path"]),Path(wrapper["settings_dir"]),Path(payload["session_dir"]))


def execute_training(settings,model_path,settings_dir,session_dir):
    return train(settings,Path(model_path),Path(settings_dir),monitor=None)


def launch(settings,argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("train","predict","worker"),default="train"); parser.add_argument("--worker-config"); args=parser.parse_args(argv)
    if args.worker_config: return _worker_from_file(args.worker_config)
    if args.mode=="train": return execute_training(settings,Path(settings["FILES"]["model"]),Path(settings["FILES"]["settings_dir"]),Path(settings["FILES"].get("session_dir",settings["FILES"]["settings_dir"])))
    if args.mode=="predict":
        _require_release_artifact(Path(settings["FILES"]["model"])); print("Model release certificate verified.",flush=True); return 0
    return 0


__all__=["build_background","execute_training","jsonable","launch","train","write_json"]
