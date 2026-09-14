"""Online geometry-to-tensor electrothermal ROM with deterministic Phi(g)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .electrothermal_tensor.integrators import (
    GeneralizedThermalSpectrum,
    integrate_etd2,
    integrate_etd2_adaptive,
    integrate_imex_euler,
    integrate_reference,
)
from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .electrothermal_tensor.vector_field import ReducedThermalOperator
from .unified_geometry import UnifiedUWPTGeometry
from .unified_open_boundary import OpenBoundaryBackground
from .unified_tensor_surrogate import UnifiedTensorSurrogate
from .unified_thermal import GeometryAwareThermalLibrary

ARCHITECTURE = "unified-geometry-tensor-geometry-aware-thermal-rom"
FORMAT_VERSION = 8


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    return value


def _validate_rule(value, rule, label):
    if not isinstance(rule, dict): return
    if "choices" in rule:
        if value not in list(rule["choices"]): raise ValueError(f"geometry {label} lies outside the production choices")
        return
    if "bounds" not in rule: return
    bounds=np.asarray(rule["bounds"],float); x=np.asarray(value,float)
    if bounds.shape==(2,):
        if x.ndim!=0 or not bounds[0]<=float(x)<=bounds[1]: raise ValueError(f"geometry {label} lies outside the production bounds")
    elif bounds.shape==(3,2):
        x=x.reshape(-1)
        if x.shape!=(3,) or np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]): raise ValueError(f"geometry {label} lies outside the production bounds")
    else: raise ValueError(f"invalid production-domain rule for {label}")


@dataclass(frozen=True)
class UnifiedPrediction:
    time: float
    state: np.ndarray
    derivative: np.ndarray
    heat_source: np.ndarray
    maximum_temperature: float
    impedance: np.ndarray
    currents: np.ndarray
    volume_power: float
    wire_power: float
    outward_power: float
    tensor_projection_correction: float
    zd_projection_correction: float
    h_projection_correction: float
    steps: int
    rejected_steps: int


@dataclass(frozen=True)
class UnifiedSteadyState:
    state: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    stable: bool
    spectral_abscissa: float
    maximum_temperature: float
    impedance: np.ndarray
    currents: np.ndarray
    volume_power: float
    wire_power: float
    outward_power: float
    tensor_projection_correction: float
    zd_projection_correction: float
    h_projection_correction: float


class _ThermalFamily:
    geometry_dimension = 0
    def __init__(self, context):
        if not context.has_thermal_operators: raise ValueError("thermal integration requires reduced thermal operators")
        self._operator=ReducedThermalOperator(context.thermal_mass_reduced,context.thermal_stiffness_reduced)
    def operator(self,_): return self._operator


class _Field:
    def __init__(self, model, context, tensors, operating):
        self.model=model; self.context=context; self.tensors=tensors; self.operating=operating
        self.thermal_operators=_ThermalFamily(context)
    def heat_source(self,state,_geometry,_operating):
        return self.model._heat_source_with_tensors(state,self.context,self.tensors,self.operating)[0]
    def vector_field(self,state,_geometry,_operating):
        op=self.thermal_operators.operator(None); a=np.asarray(state,float)
        return np.linalg.solve(op.mass,-op.stiffness@a+self.heat_source(a,None,None))


class UnifiedNeuralElectroThermalModel:
    def __init__(
        self,
        background,
        surrogate,
        *,
        default_geometry,
        current_offset=None,
        current_matrix=None,
        production_domain=None,
        context_cache_size=16,
    ):
        if getattr(background,"thermal_library",None) is None or background.thermal_rank<1:
            raise ValueError("unified model requires a geometry-aware thermal library")
        if surrogate.thermal_rank!=background.thermal_rank:
            raise ValueError("tensor surrogate and geometry-aware thermal ranks differ")
        n_ports=len(background.coil_materials)
        if surrogate.n_ports!=n_ports: raise ValueError("tensor surrogate port count differs from background")
        self.background=background; self.surrogate=surrogate; self.default_geometry=dict(default_geometry)
        self.production_domain={} if production_domain is None else _plain(production_domain)
        self.current_offset=np.zeros(n_ports,complex) if current_offset is None else np.asarray(current_offset,complex).reshape(-1)
        self.current_matrix=np.eye(n_ports,dtype=complex) if current_matrix is None else np.asarray(current_matrix,complex)
        if self.current_offset.shape!=(n_ports,) or self.current_matrix.ndim!=2 or self.current_matrix.shape[0]!=n_ports:
            raise ValueError("current affine map does not match port count")
        self.context_cache_size=max(1,int(context_cache_size)); self._contexts=OrderedDict(); self._tensor_cache=OrderedDict()

    @property
    def thermal_rank(self): return self.background.thermal_rank
    @property
    def current_dimension(self): return self.current_matrix.shape[1]

    def _validate_production_geometry(self,mapping):
        if not self.production_domain: return
        candidate=dict(mapping)
        for section,rules in self.production_domain.items():
            if section not in candidate: raise ValueError(f"geometry is missing production-domain field {section}")
            if isinstance(rules,dict) and isinstance(candidate[section],dict):
                for key,rule in rules.items():
                    if key not in candidate[section]: raise ValueError(f"geometry is missing production-domain field {section}.{key}")
                    _validate_rule(candidate[section][key],rule,f"{section}.{key}")
            else: _validate_rule(candidate[section],rules,section)

    def geometry_context(self,geometry=None):
        mapping=self.default_geometry if geometry is None else geometry
        if isinstance(mapping,UnifiedUWPTGeometry): g=mapping
        else:
            self._validate_production_geometry(mapping); g=UnifiedUWPTGeometry.from_mapping(mapping)
        key=g.canonical_json()
        if key in self._contexts:
            self._contexts.move_to_end(key); return self._contexts[key]
        context=self.background.geometry_context(g,assemble_thermal=True)
        if not hasattr(context,"thermal_basis") or context.thermal_basis.shape[1]!=self.thermal_rank:
            raise RuntimeError("geometry-specific thermal basis was not assembled")
        self._contexts[key]=context
        if len(self._contexts)>self.context_cache_size: self._contexts.popitem(last=False)
        return context

    def tensors(self,geometry=None):
        context=self.geometry_context(geometry); key=context.geometry.canonical_json()
        if key in self._tensor_cache:
            self._tensor_cache.move_to_end(key); return self._tensor_cache[key]
        phi=np.asarray(context.thermal_basis,float)
        value=self.surrogate.predict(context.geometry,np.min(phi,axis=0),np.max(phi,axis=0))
        self._tensor_cache[key]=value
        if len(self._tensor_cache)>self.context_cache_size: self._tensor_cache.popitem(last=False)
        return value

    def project_initial_temperature(self,rise,geometry=None):
        context=self.geometry_context(geometry); value=np.asarray(rise,float)
        if value.ndim==0: value=np.full(self.background.n_cells,float(value))
        value=value.reshape(-1)
        if value.shape!=(self.background.n_cells,) or np.any(~np.isfinite(value)):
            raise ValueError("initial temperature rise must be scalar or one value per cell")
        phi=np.asarray(context.thermal_basis,float); rhs=phi.T@(context.thermal_mass_full@value)
        try: return np.linalg.solve(context.thermal_mass_reduced,rhs)
        except np.linalg.LinAlgError: return np.linalg.lstsq(context.thermal_mass_reduced,rhs,rcond=None)[0]

    def temperature_field(self,state,geometry=None,*,context=None):
        context=self.geometry_context(geometry) if context is None else context
        a=np.asarray(state,float).reshape(-1)
        if a.shape!=(self.thermal_rank,) or np.any(~np.isfinite(a)): raise ValueError("thermal state dimension mismatch")
        return self.background.ambient_temperature+np.asarray(context.thermal_basis,float)@a

    def maximum_temperature(self,state,geometry=None,*,context=None):
        field=self.temperature_field(state,geometry,context=context)
        return float(max(self.background.ambient_temperature,np.max(field)))

    def _current_drive(self,operating):
        u=np.asarray(operating,complex).reshape(-1)
        if u.shape!=(self.current_dimension,) or np.any(~np.isfinite(u)): raise ValueError("invalid operating current vector")
        return self.current_offset+self.current_matrix@u

    @staticmethod
    def _series_impedance(value,n_ports):
        if value is None: return np.zeros((n_ports,n_ports),complex)
        z=np.asarray(value,complex)
        if z.ndim==0: return np.eye(n_ports,dtype=complex)*z
        if z.ndim==1:
            if z.shape!=(n_ports,): raise ValueError("series impedance vector has wrong size")
            return np.diag(z)
        if z.shape!=(n_ports,n_ports): raise ValueError("series impedance matrix has wrong size")
        return z

    def _currents(self,state,context,tensors,operating):
        if isinstance(operating,dict):
            if "voltage" not in operating: raise ValueError("voltage-driven operating mapping requires 'voltage'")
            voltage=np.asarray(operating["voltage"],complex).reshape(-1)
            if voltage.shape!=(tensors.z_field.shape[0],): raise ValueError("drive voltage has wrong port dimension")
            resistance=self.background.wire_resistances(context,state)
            total=tensors.z_field+np.diag(resistance)+self._series_impedance(operating.get("series_impedance"),len(resistance))
            try: return np.linalg.solve(total,voltage)
            except np.linalg.LinAlgError: return np.linalg.lstsq(total,voltage,rcond=None)[0]
        return self._current_drive(operating)

    def impedance(self,state,context,tensors=None):
        tensors=self.tensors(context.geometry) if tensors is None else tensors
        return tensors.z_field+np.diag(self.background.wire_resistances(context,state))

    def _wire_modal_heat(self,state,context,currents):
        resistance=self.background.wire_resistances(context,state); phi=np.asarray(context.thermal_basis,float)
        reduced=np.zeros(self.thermal_rank,float); power=0.0
        for p,(r,weights) in enumerate(zip(resistance,context.line_heat_weights)):
            local_power=0.5*float(r)*float(abs(currents[p])**2); power+=local_power
            reduced+=local_power*(phi.T@np.asarray(weights,float))
        return reduced,np.asarray(resistance,float),float(power)

    def _heat_source_with_tensors(self,state,context,tensors,operating):
        a=np.asarray(state,float).reshape(-1)
        if a.shape!=(self.thermal_rank,) or np.any(~np.isfinite(a)): raise ValueError("invalid thermal state")
        currents=self._currents(a,context,tensors,operating); volume_modal=tensors.modal_heat(currents)
        wire_modal,resistance,wire_power=self._wire_modal_heat(a,context,currents)
        return np.asarray(volume_modal+wire_modal,float),currents,resistance,tensors.volume_power(currents),wire_power,tensors.implied_outward_power(currents)

    def heat_source(self,state,context,operating):
        return self._heat_source_with_tensors(state,context,self.tensors(context.geometry),operating)

    def evaluate(self,state,geometry,operating):
        context=self.geometry_context(geometry); tensors=self.tensors(context.geometry)
        q,currents,resistance,volume_power,wire_power,outward_power=self._heat_source_with_tensors(state,context,tensors,operating)
        operator=ReducedThermalOperator(context.thermal_mass_reduced,context.thermal_stiffness_reduced); a=np.asarray(state,float).reshape(-1)
        derivative=np.linalg.solve(operator.mass,-operator.stiffness@a+q)
        return {
            "heat_source":q,"derivative":derivative,"temperature":self.temperature_field(a,context=context),
            "maximum_temperature":self.maximum_temperature(a,context=context),"impedance":tensors.z_field+np.diag(resistance),
            "currents":currents,"volume_power":float(volume_power),"wire_power":float(wire_power),"outward_power":float(outward_power),
            "tensor_projection_correction":float(tensors.projection_correction),"zd_projection_correction":float(tensors.zd_projection_correction),
            "h_projection_correction":float(tensors.h_projection_correction),
        }

    def predict(self,time,*,initial_state,geometry,operating,max_step=100.0,method="etd2_adaptive",rtol=1e-5,atol=1e-8,initial_step=None):
        t=float(time); a0=np.asarray(initial_state,float).reshape(-1)
        if a0.shape!=(self.thermal_rank,) or np.any(~np.isfinite(a0)) or not np.isfinite(t) or t<0: raise ValueError("invalid prediction state/time")
        context=self.geometry_context(geometry); tensors=self.tensors(context.geometry); field=_Field(self,context,tensors,operating); empty=np.empty(0)
        spectrum=GeneralizedThermalSpectrum(field.thermal_operators.operator(empty)) if t>0 and method in {"etd2","etd2_adaptive","adaptive_etd2"} else None
        if method=="etd2": result=integrate_etd2(field,t,initial_state=a0,geometry=empty,operating=empty,max_step=max_step,spectrum=spectrum)
        elif method in {"etd2_adaptive","adaptive_etd2"}: result=integrate_etd2_adaptive(field,t,initial_state=a0,geometry=empty,operating=empty,max_step=max_step,rtol=rtol,atol=atol,initial_step=initial_step,spectrum=spectrum)
        elif method=="imex": result=integrate_imex_euler(field,t,initial_state=a0,geometry=empty,operating=empty,max_step=max_step)
        elif method=="reference": result=integrate_reference(field,t,initial_state=a0,geometry=empty,operating=empty,rtol=rtol,atol=atol)
        else: raise ValueError("method must be etd2, etd2_adaptive, imex, or reference")
        e=self.evaluate(result.state,context.geometry,operating)
        return UnifiedPrediction(t,result.state,e["derivative"],e["heat_source"],e["maximum_temperature"],e["impedance"],e["currents"],e["volume_power"],e["wire_power"],e["outward_power"],e["tensor_projection_correction"],e["zd_projection_correction"],e["h_projection_correction"],result.steps,result.rejected_steps)

    @staticmethod
    def _numerical_jacobian(function,state,value=None):
        a=np.asarray(state,float).reshape(-1); base=np.asarray(function(a) if value is None else value,float).reshape(-1)
        jacobian=np.empty((base.size,a.size),float); epsilon=np.sqrt(np.finfo(float).eps)*(1.0+np.abs(a))
        for k in range(a.size):
            trial=a.copy(); trial[k]+=epsilon[k]; jacobian[:,k]=(np.asarray(function(trial),float).reshape(-1)-base)/epsilon[k]
        return jacobian

    def steady_state(self,*,initial_guess,geometry,operating,tolerance=1e-10,max_iterations=40):
        state=np.asarray(initial_guess,float).reshape(-1).copy()
        if state.shape!=(self.thermal_rank,) or np.any(~np.isfinite(state)): raise ValueError("invalid steady-state initial guess")
        context=self.geometry_context(geometry); tensors=self.tensors(context.geometry); operator=ReducedThermalOperator(context.thermal_mass_reduced,context.thermal_stiffness_reduced)
        def residual(a):
            q=self._heat_source_with_tensors(a,context,tensors,operating)[0]
            return np.linalg.solve(operator.mass,-operator.stiffness@a+q)
        last_iteration=0
        for iteration in range(int(max_iterations)+1):
            last_iteration=iteration; value=residual(state); norm=float(np.linalg.norm(value))
            if norm<=tolerance or iteration==max_iterations: break
            jacobian=self._numerical_jacobian(residual,state,value)
            try: step=np.linalg.solve(jacobian,-value)
            except np.linalg.LinAlgError: step=np.linalg.lstsq(jacobian,-value,rcond=None)[0]
            factor=1.0; accepted=False
            for _ in range(12):
                trial=state+factor*step
                if np.linalg.norm(residual(trial))<norm: state=trial; accepted=True; break
                factor*=0.5
            if not accepted: break
        final_residual=residual(state); residual_norm=float(np.linalg.norm(final_residual)); converged=residual_norm<=tolerance
        if converged:
            jacobian=self._numerical_jacobian(residual,state,final_residual); spectral_abscissa=float(np.max(np.real(np.linalg.eigvals(jacobian))))
            stable=bool(np.isfinite(spectral_abscissa) and spectral_abscissa<0.0)
        else: spectral_abscissa=float("nan"); stable=False
        e=self.evaluate(state,context.geometry,operating)
        return UnifiedSteadyState(state,residual_norm,last_iteration,converged,stable,spectral_abscissa,e["maximum_temperature"],e["impedance"],e["currents"],e["volume_power"],e["wire_power"],e["outward_power"],e["tensor_projection_correction"],e["zd_projection_correction"],e["h_projection_correction"])

    def save(self,path,*,metadata=None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); checkpoint=self.surrogate.checkpoint(); library=self.background.thermal_library
        meta={
            "format_version":FORMAT_VERSION,"architecture":ARCHITECTURE,"frequency_hz":self.background.frequency_hz,
            "ambient_temperature":self.background.ambient_temperature,"materials":_plain(self.background.materials),
            "coil_materials":list(self.background.coil_materials),"package_materials":list(self.background.package_materials),
            "seawater_material":self.background.seawater_material,"thermal_rank":self.thermal_rank,
            "thermal_reference_geometry":library.reference_geometry.to_mapping(),"thermal_time_scales":list(library.time_scales),
            "thermal_conditioning_limit":float(library.conditioning_limit),"thermal_local_count":len(library.local_modes),
            "default_geometry":_plain(self.default_geometry),"production_domain":_plain(self.production_domain),
            "network_config":checkpoint["network_config"],"network_dtype":checkpoint["dtype"],"n_ports":checkpoint["n_ports"],
            "metadata":_plain(dict(metadata or {})),
        }
        arrays={
            "metadata_json":np.array(json.dumps(meta,sort_keys=True,allow_nan=False)),"background_x":self.background.x,"background_y":self.background.y,"background_z":self.background.z,
            "thermal_background_modes":library.background_modes,"current_offset":self.current_offset,"current_matrix":self.current_matrix,
            "input_mean":checkpoint["input_mean"],"input_scale":checkpoint["input_scale"],"output_mean":checkpoint["output_mean"],"output_scale":checkpoint["output_scale"],"pod_basis":checkpoint["pod_basis"],
        }
        for p,modes in enumerate(library.local_modes): arrays[f"thermal_local_modes_{p}"]=modes
        for key,value in checkpoint["network_state"].items(): arrays["network__"+key.replace(".","__DOT__")]=value.detach().cpu().numpy()
        temporary=path.with_suffix(path.suffix+".tmp")
        with temporary.open("wb") as handle: np.savez_compressed(handle,**arrays)
        temporary.replace(path); return path

    @classmethod
    def load(cls,path,*,device="cpu"):
        import torch
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data["metadata_json"]))
            if meta.get("architecture")!=ARCHITECTURE or int(meta.get("format_version",-1))!=FORMAT_VERSION:
                raise ValueError("model is not the current geometry-aware thermal tensor architecture")
            library=GeometryAwareThermalLibrary(
                UnifiedUWPTGeometry.from_mapping(meta["thermal_reference_geometry"]),
                np.asarray(data["thermal_background_modes"],float),
                tuple(np.asarray(data[f"thermal_local_modes_{p}"],float) for p in range(int(meta["thermal_local_count"]))),
                tuple(float(v) for v in meta["thermal_time_scales"]),
                float(meta.get("thermal_conditioning_limit",1e10)),
            )
            background=OpenBoundaryBackground(
                data["background_x"],data["background_y"],data["background_z"],frequency_hz=meta["frequency_hz"],materials=meta["materials"],
                coil_materials=meta["coil_materials"],package_materials=meta["package_materials"],seawater_material=meta["seawater_material"],ambient_temperature=meta["ambient_temperature"],
            )
            background.set_thermal_library(library)
            config=ResidualMLPConfig(**dict(meta["network_config"])); normalizer=FeatureNormalizer(np.asarray(data["input_mean"],float),np.asarray(data["input_scale"],float))
            network=build_residual_mlp(config,normalizer); dtype=torch.float32 if meta.get("network_dtype")=="float32" else torch.float64
            network=network.to(device=device,dtype=dtype); state={}
            for name in data.files:
                if name.startswith("network__"):
                    key=name[len("network__"):].replace("__DOT__","."); state[key]=torch.as_tensor(data[name],dtype=dtype,device=device)
            network.load_state_dict(state); network.eval()
            surrogate=UnifiedTensorSurrogate(network,np.asarray(data["output_mean"],float),np.asarray(data["output_scale"],float),np.asarray(data["pod_basis"],float),int(meta["n_ports"]),int(meta["thermal_rank"]))
            return cls(background,surrogate,default_geometry=meta["default_geometry"],current_offset=np.asarray(data["current_offset"],complex),current_matrix=np.asarray(data["current_matrix"],complex),production_domain=meta.get("production_domain"))


__all__=["ARCHITECTURE","FORMAT_VERSION","UnifiedNeuralElectroThermalModel","UnifiedPrediction","UnifiedSteadyState"]
