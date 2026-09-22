"""Runtime for geometry-to-spatial-Joule + geometry-local thermal ROM."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import uuid

import numpy as np

from .unified_spatial_final_audit import run_spatial_final_held_out_audit
from .unified_geometry import UnifiedUWPTGeometry, sample_geometry
from .unified_model import UnifiedNeuralElectroThermalModel
from .unified_open_boundary import OpenBoundaryBackground
from .unified_spatial_physics_gate import run_spatial_physics_gate
from .unified_corrected_truth import generate_spatial_tensor_dataset
from .unified_tensor_surrogate import (
    SpatialTensorDataset,
    encode_geometry,
    project_spatial_dataset_to_production_cone,
)
from .unified_tensor_training import train_spatial_tensor_surrogate
from .unified_thermal import configure_maxwell_field_cache
from .unified_online_thermal import merge_thermal_time_scales
from .unified_corrected_truth_preflight import run_truth_preflight

_CACHE_FORMAT = 56
_PREFLIGHT_CACHE_FORMAT = 37
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


def _background_signature_view(settings):
    """Physical BACKGROUND payload with solver-only v53 knobs removed.

    The two localized-transverse direct thresholds change only how the same
    certified linear system is solved.  Omitting them preserves compatibility
    with caches created before those knobs existed while keeping all physical
    geometry/material/self-correction settings in the signature.
    """
    background=jsonable(settings["BACKGROUND"])
    self_correction=dict(background.get("self_correction",{}) or {})
    self_correction.pop("linear_transverse_direct_max_dofs",None)
    self_correction.pop("linear_transverse_direct_fallback_max_dofs",None)
    if "self_correction" in background:
        background["self_correction"]=self_correction
    return background


def _signature(settings, *, cache_format=None):
    # Spatial-Joule truth is thermal-rank free.  Dataset cache identity depends
    # only on physical geometry/material settings plus the frozen truth sample
    # policy, not on neural optimizer choices or online thermal ROM tolerances.
    keys = (
        "BACKGROUND",
        "DEFAULT_GEOMETRY",
        "GEOMETRY_SAMPLING",
        "PHYSICS",
        "MATERIALS",
        "REGIONS",
    )
    payload = {k: settings[k] for k in keys}
    payload["BACKGROUND"] = _background_signature_view(settings)
    training = settings["TRAINING"]
    payload["TRAINING"] = {
        "seed": int(training.get("seed", 17)),
        "n_tensor_samples": int(training.get("n_tensor_samples", 96)),
        "spatial_tensor_schema": str(
            training.get(
                "spatial_tensor_schema",
                "cellwise_joule_tensor_v1",
            )
        ),
    }
    payload["cache_format"] = (
        _CACHE_FORMAT if cache_format is None else int(cache_format)
    )
    text = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _production_thermal_time_scales(settings, extra_times=()):
    """Thermal resolvent horizon declared by Gate, final audit and prediction."""
    training = settings["TRAINING"]
    final = dict(training.get("final_audit", {}) or {})
    prediction = dict(settings.get("PREDICTION", {}) or {})
    return merge_thermal_time_scales(
        training.get("thermal_time_scales", [0.1, 1.0, 10.0]),
        training.get("thermal_trajectory_times", ()),
        final.get("times", ()),
        prediction.get("times", ()),
        extra_times,
    )


def _preflight_signature(settings):
    # Numerical solver-policy knobs must not invalidate already certified
    # physical truth/preflight data.  Keep the historical signature stable when
    # only the new medium-system transverse direct threshold changes.
    background=jsonable(settings["BACKGROUND"])
    self_correction=dict(background.get("self_correction",{}) or {})
    self_correction.pop("linear_transverse_direct_max_dofs",None)
    self_correction.pop("linear_transverse_direct_fallback_max_dofs",None)
    if "self_correction" in background:
        background["self_correction"]=self_correction
    payload={
        "BACKGROUND":background,
        "DEFAULT_GEOMETRY":settings["DEFAULT_GEOMETRY"],
        "GEOMETRY_SAMPLING":settings.get("GEOMETRY_SAMPLING"),
        "PHYSICS":settings["PHYSICS"],
        "MATERIALS":settings["MATERIALS"],
        "REGIONS":settings["REGIONS"],
        "seed":int(settings["TRAINING"].get("seed",17)),
        "preflight_cache_format":int(_PREFLIGHT_CACHE_FORMAT),
    }
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


def _gate_sample_count(settings):
    background=settings["BACKGROUND"]
    sections=("open_boundary_check","formulation_check","mesh_check","geometry_continuity_check","self_correction")
    return max(1,*[int(dict(background.get(name,{})).get("samples",1)) for name in sections])


def train(settings, model_path, settings_dir, monitor=None):
    from .training.monitor import TrainingStopped

    settings_dir.mkdir(parents=True, exist_ok=True)
    sig = _signature(settings)
    meta_path, _legacy_thermal_path, data_path = _cache_paths(
        settings_dir
    )
    checkpoint = Path(settings["FILES"]["training_checkpoint"])
    checkpoint = (
        checkpoint
        if checkpoint.is_absolute()
        else Path(settings["ROOT"]) / checkpoint
    )

    try:
        _progress(
            "构建开放边界固定背景物理空间",
            0,
            monitor,
        )
        bg = build_background(settings)
        _progress(
            "构建开放边界固定背景物理空间",
            5,
            monitor,
        )
        print(
            f"背景空间：{bg.n_cells} cells，{bg.n_edges} Maxwell edge DOFs；"
            "Maxwell 仅用于离线 open-boundary truth；thermal ROM 按查询 geometry 在线构造。",
            flush=True,
        )

        seed = int(settings["TRAINING"].get("seed", 17))
        preflight_cache_valid = False
        valid_cache = False
        cache_meta = {}
        preflight_sig = _preflight_signature(settings)

        # Keep the already-certified EM field cache across thermal-architecture
        # changes.  Its signature contains physical EM settings only.
        configure_maxwell_field_cache(
            bg,
            settings_dir / "unified.thermal_maxwell_fields.npz",
            preflight_sig,
        )

        if meta_path.is_file():
            try:
                cache_meta = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
                cached_preflight = dict(
                    cache_meta.get("truth_preflight", {})
                )
                preflight_cache_valid = (
                    cache_meta.get("preflight_signature")
                    == preflight_sig
                    and bool(
                        cached_preflight.get("certified", False)
                    )
                    and bool(
                        cached_preflight.get(
                            "local_self_correction_converged",
                            False,
                        )
                    )
                    and cache_meta.get("self_correction_model")
                    == _SELF_CORRECTION_MODEL
                )
                valid_cache = (
                    cache_meta.get("signature") == sig
                    and int(
                        cache_meta.get("cache_format", -1)
                    )
                    == _CACHE_FORMAT
                    and preflight_cache_valid
                    and data_path.is_file()
                    and cache_meta.get("tensor_representation")
                    == "cellwise_joule_tensor_v1"
                )
            except (OSError, ValueError, TypeError):
                preflight_cache_valid = False
                valid_cache = False
                cache_meta = {}

        if preflight_cache_valid:
            preflight = dict(cache_meta["truth_preflight"])
            _progress(
                "复用已认证 spatial truth preflight",
                7,
                monitor,
            )
            print(
                "Truth preflight：复用当前 physical-cache signature 下已通过的验证结果。",
                flush=True,
            )
        else:
            preflight_rng = np.random.default_rng(seed + 65537)
            preflight_geometries = _sample_geometries(
                settings,
                _gate_sample_count(settings),
                preflight_rng,
                bg,
            )
            _progress(
                "执行 spatial truth preflight",
                7,
                monitor,
            )
            preflight = run_truth_preflight(
                settings,
                bg,
                preflight_geometries,
                monitor=monitor,
            )
            if not preflight["certified"]:
                raise RuntimeError(
                    "Spatial truth preflight failed; refusing tensor truth generation: "
                    + json.dumps(
                        jsonable(preflight),
                        sort_keys=True,
                    )
                )
            print(
                "Truth preflight：finite-support source / terminal continuity / "
                "loss partition / open-domain / MQS / local-self reference / "
                "corrected EM mesh convergence 全部通过。",
                flush=True,
            )
            cache_meta = {
                "cache_format": _CACHE_FORMAT,
                "signature": sig,
                "preflight_signature": preflight_sig,
                "truth_preflight": preflight,
                "self_correction_model": _SELF_CORRECTION_MODEL,
            }
            write_json(meta_path, cache_meta)

        if valid_cache:
            _progress(
                "复用 cellwise spatial-Joule tensor truth 数据",
                35,
                monitor,
            )
            dataset = SpatialTensorDataset.load(data_path)
            if (
                dataset.n_ports != len(bg.coil_materials)
                or dataset.n_cells != bg.n_cells
            ):
                raise RuntimeError(
                    "cached spatial tensor dataset dimensions differ from background"
                )
        else:
            checkpoint.unlink(missing_ok=True)
            tensor_rng = np.random.default_rng(seed + 131071)
            n_tensor = int(
                settings["TRAINING"].get(
                    "n_tensor_samples",
                    96,
                )
            )
            tensor_geometries = _sample_geometries(
                settings,
                n_tensor,
                tensor_rng,
                bg,
            )
            _progress(
                "生成 geometry→Z/D/cell-Joule truth 数据",
                12,
                monitor,
            )
            dataset = generate_spatial_tensor_dataset(
                bg,
                tensor_geometries,
                seed=seed,
                monitor=monitor,
            )
            dataset.save(data_path)
            cache_meta = {
                "cache_format": _CACHE_FORMAT,
                "signature": sig,
                "preflight_signature": preflight_sig,
                "truth_preflight": preflight,
                "tensor_representation": "cellwise_joule_tensor_v1",
                "thermal_representation": (
                    "geometry_local_rational_krylov_v1"
                ),
                "em_representation": (
                    "geometry_to_port_and_cellwise_joule_tensors"
                ),
                "em_boundary": "silver_muller_impedance",
                "source_model": bg.source_model,
                "self_correction_model": _SELF_CORRECTION_MODEL,
            }
            write_json(meta_path, cache_meta)
            _progress(
                "生成 geometry→Z/D/cell-Joule truth 数据",
                50,
                monitor,
            )

        # v56 truth caches are expensive Maxwell products.  Upgrade their packed
        # Z/D/cell labels through the exact production decoder instead of
        # invalidating and recomputing the physical truth.  The operation is
        # deterministic, idempotent and Maxwell-free.
        cache_projection = project_spatial_dataset_to_production_cone(dataset)
        dataset.save(data_path)
        print(
            "spatial truth production-cone projection: "
            f"maximum relative correction={cache_projection:.3e}",
            flush=True,
        )

        gate_rng = np.random.default_rng(seed + 104729)
        gate_geometries = _sample_geometries(
            settings,
            _gate_sample_count(settings),
            gate_rng,
            bg,
        )
        _progress(
            "执行 spatial-Joule / online-thermal Physics Gate",
            52,
            monitor,
        )
        gate = run_spatial_physics_gate(
            settings,
            bg,
            dataset,
            gate_geometries,
            preflight=preflight,
            monitor=monitor,
        )
        if not gate["certified"]:
            raise RuntimeError(
                "Spatial Physics Gate failed; refusing surrogate training: "
                + json.dumps(jsonable(gate), sort_keys=True)
            )
        print(
            "Physics Gate：corrected Z/D + cellwise PSD Joule / exact sum-to-D / "
            "held-out full-vs-online thermal transient 全部通过。",
            flush=True,
        )

        _progress(
            "训练 geometry→spatial-Joule POD-MLP",
            62,
            monitor,
        )
        surrogate, report = train_spatial_tensor_surrogate(
            dataset,
            background=bg,
            network_settings=settings["TRAINING"].get(
                "network"
            ),
            training_settings=settings["TRAINING"].get(
                "optimizer"
            ),
            device=settings["TRAINING"].get(
                "device",
                "cuda",
            ),
            monitor=monitor,
            checkpoint_path=checkpoint,
        )

        model = UnifiedNeuralElectroThermalModel(
            bg,
            surrogate,
            default_geometry=settings["DEFAULT_GEOMETRY"],
            current_offset=settings["PORTS"].get(
                "current_offset"
            ),
            current_matrix=settings["PORTS"].get(
                "current_matrix"
            ),
            production_domain=settings.get(
                "GEOMETRY_SAMPLING"
            ),
            thermal_time_scales=_production_thermal_time_scales(settings),
            thermal_conditioning_limit=float(
                settings["TRAINING"].get(
                    "online_thermal_conditioning_limit",
                    settings["TRAINING"].get(
                        "thermal_basis_conditioning_limit",
                        1e10,
                    ),
                )
            ),
            thermal_target_relative_error=float(
                settings["TRAINING"].get(
                    "online_thermal_relative_tolerance",
                    settings["TRAINING"].get(
                        "thermal_basis_energy_tolerance",
                        5e-2,
                    ),
                )
            ),
        )

        final_cfg = dict(
            settings["TRAINING"].get("final_audit", {})
        )
        final_count = max(
            1,
            int(final_cfg.get("samples", 2)),
        )
        final_rng = np.random.default_rng(seed + 524287)
        final_geometries = _sample_geometries(
            settings,
            final_count,
            final_rng,
            bg,
        )
        _progress(
            "执行 completely-held-out spatial Go/No-Go audit",
            92,
            monitor,
        )
        final_audit = run_spatial_final_held_out_audit(
            settings,
            model,
            final_geometries,
            monitor=monitor,
        )
        if not final_audit["certified"]:
            raise RuntimeError(
                "Final held-out audit failed; refusing model artifact: "
                + json.dumps(
                    jsonable(final_audit),
                    sort_keys=True,
                )
            )
        print(
            "Final audit：held-out spatial tensor/current contractions / "
            "full-vs-online thermal / current+circuit dynamics / production "
            "integrator / steady stability 全部通过。",
            flush=True,
        )

        _progress(
            "保存 spatial-Joule + online-thermal 统一模型",
            98,
            monitor,
        )
        model.save(
            model_path,
            metadata={
                "truth_preflight": preflight,
                "physics_gate": gate,
                "final_held_out_audit": final_audit,
                "training_report": asdict(report),
            },
        )
        checkpoint.unlink(missing_ok=True)
        default_rank = model.thermal_rank_for(
            settings["DEFAULT_GEOMETRY"]
        )
        write_json(
            settings_dir / "training.report.json",
            {
                "model": str(model_path),
                "background_cells": bg.n_cells,
                "offline_maxwell_dofs": bg.n_edges,
                "online_em_representation": (
                    "corrected Z_field + D_vol + cellwise Joule tensors"
                ),
                "tensor_representation": (
                    "cellwise_joule_tensor_v1"
                ),
                "thermal_representation": (
                    "geometry_local_rational_krylov_v1"
                ),
                "default_geometry_online_thermal_rank": (
                    default_rank
                ),
                "truth_preflight": preflight,
                "physics_gate": gate,
                "final_held_out_audit": final_audit,
                "training": report,
            },
        )
        _progress("训练完成", 100, monitor)
        print(
            "训练完成："
            f"default geometry online thermal rank={default_rank}，"
            f"tensor POD rank={report.pod_rank}，在线 Maxwell solve=0，"
            f"best epoch={report.best_epoch}，"
            f"validation matrix loss={report.best_validation_loss:.6g}，"
            f"test relative tensor error={report.test_relative_tensor_error:.6g}。",
            flush=True,
        )
        print(f"模型已保存：{model_path}", flush=True)
        if monitor is not None:
            monitor.finish(
                "completed",
                model=str(model_path),
            )
        return 0

    except TrainingStopped:
        if monitor is not None:
            monitor.finish(
                "stopped",
                checkpoint=(
                    str(checkpoint)
                    if checkpoint.is_file()
                    else None
                ),
            )
        print(
            (
                f"训练已停止；spatial tensor MLP 检查点：{checkpoint}"
                if checkpoint.is_file()
                else "训练已停止。"
            ),
            flush=True,
        )
        return 130


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
    rank=model.thermal_rank_for(geometry)
    if value is None or (isinstance(value,str) and value.lower()=="ambient"): return np.zeros(rank)
    array=np.asarray(value,float)
    if array.ndim==0 and float(array)==0.0: return np.zeros(rank)
    return model.project_initial_temperature(array,geometry)


def predict(settings,model_path,output_path,settings_dir):
    if not model_path.is_file(): raise FileNotFoundError(f"模型不存在：{model_path}")
    _require_release_artifact(model_path); device=_device(settings["TRAINING"].get("device","cuda")); model=UnifiedNeuralElectroThermalModel.load(model_path,device=device)
    p=settings["PREDICTION"]; geometry=p.get("geometry") or settings["DEFAULT_GEOMETRY"]
    requested_scales = merge_thermal_time_scales(model.thermal_time_scales, p.get("times", ()))
    if tuple(requested_scales) != tuple(model.thermal_time_scales):
        model.thermal_time_scales = tuple(requested_scales)
        model._contexts.clear()
    initial=_initial_state(model,p,geometry)
    operating=p.get("drive",p.get("operating",[1.0]+[0.0]*(model.current_dimension-1))); results=[]; tensors=model.tensors(geometry)
    print(f"加载 spatial-Joule online-thermal ROM：{model_path}  device={device}  thermal rank={model.thermal_rank_for(geometry)}  POD={model.surrogate.pod_rank}  tensor projection correction={tensors.projection_correction:.3e}",flush=True)
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
    parser=argparse.ArgumentParser(description="统一 geometry→spatial-Joule + geometry-local thermal ROM（在线无 Maxwell）")
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
