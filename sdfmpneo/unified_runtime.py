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

_CACHE_FORMAT = 19
_SELF_CORRECTION_MODEL = "canonical_local_fine_minus_coarse_self_defect_v1"


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