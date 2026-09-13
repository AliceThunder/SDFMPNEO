"""Unified geometry-independent residual-corrected neural electrothermal model."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .electrothermal_tensor.integrators import GeneralizedThermalSpectrum,integrate_etd2,integrate_etd2_adaptive,integrate_imex_euler,integrate_reference
from .electrothermal_tensor.network import FeatureNormalizer,ResidualMLPConfig,build_residual_mlp
from .electrothermal_tensor.vector_field import ReducedThermalOperator
from .unified_background import FixedMultiscaleBackground
from .unified_geometry import UnifiedUWPTGeometry
from .unified_maxwell import NeuralMaxwellAccelerator

ARCHITECTURE="unified-residual-corrected-neural-electrothermal-solver"
FORMAT_VERSION=1

@dataclass(frozen=True)
class UnifiedPrediction:
    time:float; state:np.ndarray; derivative:np.ndarray; heat_source:np.ndarray; maximum_temperature:float; impedance:np.ndarray; maxwell_initial_residual:tuple; maxwell_final_residual:tuple; maxwell_correction_iterations:tuple; steps:int; rejected_steps:int
@dataclass(frozen=True)
class UnifiedSteadyState:
    state:np.ndarray; residual_norm:float; iterations:int; converged:bool; maximum_temperature:float; impedance:np.ndarray; maxwell_final_residual:tuple

class _ThermalFamily:
    geometry_dimension=0
    def __init__(self,context): self._op=ReducedThermalOperator(context.thermal_mass_reduced,context.thermal_stiffness_reduced)
    def operator(self,_): return self._op

class _Field:
    def __init__(self,model,context,operating): self.model=model; self.context=context; self.operating=np.asarray(operating,float); self.thermal_operators=_ThermalFamily(context)
    def heat_source(self,state,_geometry,_operating): return self.model.heat_source(state,self.context,self.operating)[0]
    def vector_field(self,state,_geometry,_operating):
        op=self.thermal_operators.operator(None); a=np.asarray(state,float); return np.linalg.solve(op.mass,-op.stiffness@a+self.heat_source(a,None,None))

class UnifiedNeuralElectroThermalModel:
    def __init__(self,background,accelerator,*,default_geometry,current_offset=None,current_matrix=None,context_cache_size=16):
        self.background=background; self.accelerator=accelerator; self.default_geometry=dict(default_geometry); p=len(background.coil_materials); self.current_offset=np.zeros(p,complex) if current_offset is None else np.asarray(current_offset,complex).reshape(-1); self.current_matrix=np.eye(p,dtype=complex) if current_matrix is None else np.asarray(current_matrix,complex)
        if self.current_offset.shape!=(p,) or self.current_matrix.ndim!=2 or self.current_matrix.shape[0]!=p: raise ValueError("current affine map does not match port count")
        self.context_cache_size=max(1,int(context_cache_size)); self._contexts=OrderedDict(); self._em_cache=OrderedDict()
    @property
    def thermal_rank(self): return self.background.thermal_basis.shape[1]
    @property
    def current_dimension(self): return self.current_matrix.shape[1]
    def geometry_context(self,geometry=None):
        mapping=self.default_geometry if geometry is None else geometry; g=mapping if isinstance(mapping,UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(mapping); key=g.canonical_json()
        if key in self._contexts: self._contexts.move_to_end(key); return self._contexts[key]
        ctx=self.background.geometry_context(g); self._contexts[key]=ctx
        if len(self._contexts)>self.context_cache_size: self._contexts.popitem(last=False)
        return ctx
    def _currents(self,operating):
        u=np.asarray(operating,complex).reshape(-1)
        if u.shape!=(self.current_dimension,) or np.any(~np.isfinite(u)): raise ValueError("invalid operating current vector")
        return self.current_offset+self.current_matrix@u
    def _em_solution(self,state,context):
        a=np.asarray(state,float).reshape(-1)
        if a.shape!=(self.thermal_rank,) or np.any(~np.isfinite(a)): raise ValueError("invalid thermal state")
        key=(context.geometry.canonical_json(),a.tobytes())
        if key in self._em_cache: self._em_cache.move_to_end(key); return self._em_cache[key]
        A=self.background.em_operator(context,a); B=self.background.rhs_matrix(context); result=self.accelerator.solve(A,B); self._em_cache[key]=result
        if len(self._em_cache)>8: self._em_cache.popitem(last=False)
        return result
    def joule_matrices(self,state,context):
        X,report=self._em_solution(state,context); ex,ey,ez,weight=self.background.material_joule_cells(context,state,X); Phi=self.background.thermal_basis; p=X.shape[1]; r=self.thermal_rank; G=np.zeros((r,p,p),float)
        for j in range(r):
            w=weight*Phi[:,j]; G[j]=np.real(ex.conj().T@(w[:,None]*ex)+ey.conj().T@(w[:,None]*ey)+ez.conj().T@(w[:,None]*ez)); G[j]=.5*(G[j]+G[j].T)
        resist=self.background.wire_resistances(context,state)
        for port,(R,h) in enumerate(zip(resist,context.line_heat_weights)):
            projection=Phi.T@h
            for j in range(r): G[j,port,port]+=.5*R*projection[j]
        return G,X,report,resist
    def heat_source(self,state,context,operating):
        z=self._currents(operating); G,X,report,resist=self.joule_matrices(state,context); q=np.array([np.real(np.vdot(z,Gj@z)) for Gj in G],float); return q,(G,X,report,resist)
    def temperature_field(self,state): return self.background.ambient_temperature+self.background.thermal_basis@np.asarray(state,float)
    def impedance(self,state,context,X=None,resist=None):
        if X is None or resist is None: _,X,_,resist=self.joule_matrices(state,context)
        return context.source_shape.T@X+np.diag(resist)
    def evaluate(self,state,geometry,operating):
        ctx=self.geometry_context(geometry); q,(G,X,report,resist)=self.heat_source(state,ctx,operating); op=ReducedThermalOperator(ctx.thermal_mass_reduced,ctx.thermal_stiffness_reduced); a=np.asarray(state,float); derivative=np.linalg.solve(op.mass,-op.stiffness@a+q); T=self.temperature_field(a)
        return dict(heat_source=q,derivative=derivative,temperature=T,maximum_temperature=float(np.max(T)),impedance=self.impedance(a,ctx,X,resist),maxwell=report,joule_matrices=G)
    def predict(self,time,*,initial_state,geometry,operating,max_step=100.,method="etd2_adaptive",rtol=1e-5,atol=1e-8,initial_step=None):
        t=float(time); a0=np.asarray(initial_state,float).reshape(-1)
        if a0.shape!=(self.thermal_rank,) or np.any(~np.isfinite(a0)) or not np.isfinite(t) or t<0: raise ValueError("invalid prediction state/time")
        ctx=self.geometry_context(geometry); field=_Field(self,ctx,operating); empty=np.empty(0); spectrum=GeneralizedThermalSpectrum(field.thermal_operators.operator(empty)) if t>0 and method in {"etd2","etd2_adaptive","adaptive_etd2"} else None
        if method=="etd2": result=integrate_etd2(field,t,initial_state=a0,geometry=empty,operating=np.empty(0),max_step=max_step,spectrum=spectrum)
        elif method in {"etd2_adaptive","adaptive_etd2"}: result=integrate_etd2_adaptive(field,t,initial_state=a0,geometry=empty,operating=np.empty(0),max_step=max_step,rtol=rtol,atol=atol,initial_step=initial_step,spectrum=spectrum)
        elif method=="imex": result=integrate_imex_euler(field,t,initial_state=a0,geometry=empty,operating=np.empty(0),max_step=max_step)
        elif method=="reference": result=integrate_reference(field,t,initial_state=a0,geometry=empty,operating=np.empty(0),rtol=rtol,atol=atol)
        else: raise ValueError("method must be etd2, etd2_adaptive, imex, or reference")
        e=self.evaluate(result.state,ctx.geometry,operating); mr=e["maxwell"]
        return UnifiedPrediction(t,result.state,e["derivative"],e["heat_source"],e["maximum_temperature"],e["impedance"],mr.initial_relative_residual,mr.final_relative_residual,mr.correction_iterations,result.steps,result.rejected_steps)
    def steady_state(self,*,initial_guess,geometry,operating,tolerance=1e-10,max_iterations=40):
        state=np.asarray(initial_guess,float).copy(); ctx=self.geometry_context(geometry); op=ReducedThermalOperator(ctx.thermal_mass_reduced,ctx.thermal_stiffness_reduced); last=0
        def residual(a): return np.linalg.solve(op.mass,-op.stiffness@a+self.heat_source(a,ctx,operating)[0])
        for it in range(int(max_iterations)+1):
            last=it; f=residual(state); norm=float(np.linalg.norm(f))
            if norm<=tolerance: break
            if it==max_iterations: break
            J=np.empty((self.thermal_rank,self.thermal_rank)); eps=np.sqrt(np.finfo(float).eps)*(1+np.abs(state))
            for k in range(self.thermal_rank):
                trial=state.copy(); trial[k]+=eps[k]; J[:,k]=(residual(trial)-f)/eps[k]
            try: step=np.linalg.solve(J,-f)
            except np.linalg.LinAlgError: step=np.linalg.lstsq(J,-f,rcond=None)[0]
            factor=1.; accepted=False
            for _ in range(12):
                trial=state+factor*step
                if np.linalg.norm(residual(trial))<norm: state=trial; accepted=True; break
                factor*=.5
            if not accepted: break
        f=residual(state); e=self.evaluate(state,ctx.geometry,operating); return UnifiedSteadyState(state,float(np.linalg.norm(f)),last,float(np.linalg.norm(f))<=tolerance,e["maximum_temperature"],e["impedance"],e["maxwell"].final_relative_residual)
    def save(self,path,*,metadata=None):
        import torch
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); net=self.accelerator.network; first=next(net.parameters()); meta={"format_version":FORMAT_VERSION,"architecture":ARCHITECTURE,"network_config":net.config.to_dict(),"network_dtype":"float64" if first.dtype==torch.float64 else "float32","frequency_hz":self.background.frequency_hz,"ambient_temperature":self.background.ambient_temperature,"materials":self.background.materials,"coil_materials":self.background.coil_materials,"package_materials":self.background.package_materials,"seawater_material":self.background.seawater_material,"thermal_rank":self.thermal_rank,"default_geometry":self.default_geometry,"residual_tolerance":self.accelerator.residual_tolerance,"maxwell_max_iterations":self.accelerator.max_iterations,"metadata":dict(metadata or {})}
        arrays={"metadata_json":np.array(json.dumps(meta,sort_keys=True,allow_nan=False)),"background_x":self.background.x,"background_y":self.background.y,"background_z":self.background.z,"em_basis":self.accelerator.basis,"current_offset":self.current_offset,"current_matrix":self.current_matrix}
        for key,value in net.state_dict().items(): arrays["network__"+key.replace(".","__DOT__")]=value.detach().cpu().numpy()
        with path.open("wb") as f: np.savez_compressed(f,**arrays)
        return path
    @classmethod
    def load(cls,path,*,device="cpu"):
        import torch
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d["metadata_json"]));
            if meta.get("architecture")!=ARCHITECTURE or int(meta.get("format_version",-1))!=FORMAT_VERSION: raise ValueError("model is not the unified residual-corrected architecture")
            bg=FixedMultiscaleBackground(d["background_x"],d["background_y"],d["background_z"],frequency_hz=meta["frequency_hz"],materials=meta["materials"],coil_materials=meta["coil_materials"],package_materials=meta["package_materials"],seawater_material=meta["seawater_material"],thermal_rank=meta["thermal_rank"],ambient_temperature=meta["ambient_temperature"])
            config=ResidualMLPConfig(**meta["network_config"]); normalizer=FeatureNormalizer(d["network__input_mean"],d["network__input_scale"]); net=build_residual_mlp(config,normalizer); dtype=torch.float64 if meta.get("network_dtype")=="float64" else torch.float32; net=net.to(device=device,dtype=dtype); state={}
            for key in d.files:
                if key.startswith("network__"): state[key[len("network__"):].replace("__DOT__",".")]=torch.as_tensor(d[key],dtype=dtype,device=device)
            net.load_state_dict(state,strict=True); net.eval(); acc=NeuralMaxwellAccelerator(net,d["em_basis"],residual_tolerance=meta["residual_tolerance"],max_iterations=meta["maxwell_max_iterations"])
            return cls(bg,acc,default_geometry=meta["default_geometry"],current_offset=d["current_offset"],current_matrix=d["current_matrix"])

__all__=["UnifiedNeuralElectroThermalModel","UnifiedPrediction","UnifiedSteadyState","ARCHITECTURE"]
