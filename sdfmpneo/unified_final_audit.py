"""Completely-held-out final Go/No-Go audit for the production tensor-ROM."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .unified_tensor_surrogate import DecodedTensors, solve_port_truth_tensors, solve_truth_tensors
from .unified_thermal import audit_geometry_aware_thermal_trajectories


def _relative(value, reference, natural_scale=None):
    a = np.asarray(value); b = np.asarray(reference)
    floor = np.finfo(float).tiny
    if natural_scale is not None: floor = max(floor, 1e-10 * float(natural_scale))
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), floor))


def _current_vectors(n_ports):
    n=int(n_ports); eye=np.eye(n,dtype=complex); values=[eye[:,i] for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            values.extend([eye[:,i]+eye[:,j],eye[:,i]-eye[:,j],eye[:,i]+1j*eye[:,j],eye[:,i]-1j*eye[:,j]])
    return values


def _truth_tensors(background,geometry):
    z,d,h,phi_min,phi_max,audit=solve_truth_tensors(background,geometry)
    z2,d2,d_out,port_audit=solve_port_truth_tensors(background,geometry)
    if _relative(z,z2)>1e-12 or _relative(d,d2)>1e-12: raise RuntimeError("final audit truth tensor paths disagree")
    return DecodedTensors(z,d,h,d_out,0.0,0.0),phi_min,phi_max,{**audit,**port_audit}


def _tensor_case(model,geometry):
    truth,phi_min,phi_max,truth_audit=_truth_tensors(model.background,geometry)
    predicted=model.surrogate.predict(geometry,phi_min,phi_max)
    z_scale=max(float(np.linalg.norm(truth.z_field)),np.finfo(float).tiny)
    d_scale=max(float(np.linalg.norm(truth.d_vol)),z_scale*1e-12)
    h_scale=max(float(np.linalg.norm(truth.modal_h)),d_scale*1e-12)
    row={"z_relative_error":_relative(predicted.z_field,truth.z_field,z_scale),
         "d_relative_error":_relative(predicted.d_vol,truth.d_vol,d_scale),
         "h_relative_error":_relative(predicted.modal_h,truth.modal_h,h_scale),
         "outward_relative_error":_relative(predicted.implied_d_out,truth.implied_d_out,z_scale),
         "zd_projection_correction":float(predicted.zd_projection_correction),
         "h_projection_correction":float(predicted.h_projection_correction)}
    current_error=0.0
    for current in _current_vectors(truth.z_field.shape[0]):
        c=np.asarray(current,complex); c2=float(np.linalg.norm(c)**2)
        current_error=max(current_error,_relative(predicted.z_field@c,truth.z_field@c,z_scale*np.sqrt(c2)))
        p_truth=truth.volume_power(c); p_pred=predicted.volume_power(c)
        current_error=max(current_error,abs(p_pred-p_truth)/max(abs(p_truth),1e-10*d_scale*c2,np.finfo(float).tiny))
        current_error=max(current_error,_relative(predicted.modal_heat(c),truth.modal_heat(c),h_scale*c2))
    row["maximum_current_space_relative_error"]=float(current_error); row["truth_audit"]=truth_audit
    return truth,predicted,row


def _rhs(model,context,tensors,operating,state):
    a=np.asarray(state,float).reshape(-1); q=model._heat_source_with_tensors(a,context,tensors,operating)[0]
    return np.linalg.solve(context.thermal_mass_reduced,-context.thermal_stiffness_reduced@a+q)


def _integrate(model,context,tensors,operating,times):
    times=np.asarray(times,float)
    if times.ndim!=1 or times.size==0 or np.any(times<=0.0) or np.any(np.diff(times)<=0.0): raise ValueError("final audit times must be strictly increasing and positive")
    result=solve_ivp(lambda t,a:_rhs(model,context,tensors,operating,a),(0.0,float(times[-1])),np.zeros(model.thermal_rank),
                     t_eval=times,method="BDF",rtol=1e-8,atol=1e-10)
    if not result.success or result.y.shape!=(model.thermal_rank,len(times)): raise RuntimeError("final audit reduced reference integration failed: "+str(result.message))
    return np.asarray(result.y.T,float)


def _mass_relative(diff,reference,mass):
    diff=np.asarray(diff,float); reference=np.asarray(reference,float); mass=np.asarray(mass,float)
    numerator=max(float(diff@(mass@diff)),0.0); denominator=max(float(reference@(mass@reference)),np.finfo(float).tiny)
    return float(np.sqrt(numerator/denominator))


def _numerical_jacobian(function,state,value=None):
    a=np.asarray(state,float).reshape(-1); base=np.asarray(function(a) if value is None else value,float).reshape(-1)
    jacobian=np.empty((base.size,a.size),float); epsilon=np.sqrt(np.finfo(float).eps)*(1.0+np.abs(a))
    for k in range(a.size):
        trial=a.copy(); trial[k]+=epsilon[k]; jacobian[:,k]=(np.asarray(function(trial),float).reshape(-1)-base)/epsilon[k]
    return jacobian


def _steady(model,context,tensors,operating,initial):
    function=lambda a:_rhs(model,context,tensors,operating,a); state=np.asarray(initial,float).reshape(-1).copy()
    for _ in range(40):
        value=function(state)
        if np.linalg.norm(value)<=1e-10: break
        jacobian=_numerical_jacobian(function,state,value)
        try: step=np.linalg.solve(jacobian,-value)
        except np.linalg.LinAlgError: step=np.linalg.lstsq(jacobian,-value,rcond=None)[0]
        factor=1.0; norm=float(np.linalg.norm(value))
        for _ in range(12):
            trial=state+factor*step
            if np.linalg.norm(function(trial))<norm: state=trial; break
            factor*=0.5
        else: break
    residual=function(state); residual_norm=float(np.linalg.norm(residual))
    if residual_norm<=1e-10:
        spectral_abscissa=float(np.max(np.real(np.linalg.eigvals(_numerical_jacobian(function,state,residual)))))
        stable=bool(np.isfinite(spectral_abscissa) and spectral_abscissa<0.0)
    else: spectral_abscissa=float("nan"); stable=False
    return state,residual_norm,stable,spectral_abscissa


def _circuit_condition(model,context,tensors,operating,state):
    if not isinstance(operating,dict): return 1.0
    resistance=model.background.wire_resistances(context,state)
    total=tensors.z_field+np.diag(resistance)+model._series_impedance(operating.get("series_impedance"),len(resistance))
    return float(np.linalg.cond(total))


def _trajectory_case(model,geometry,truth,predicted,operating,times,*,integrator_rtol,integrator_atol,integrator_max_step):
    context=model.geometry_context(geometry); truth_states=_integrate(model,context,truth,operating,times); predicted_states=_integrate(model,context,predicted,operating,times)
    phi=np.asarray(context.thermal_basis,float); mass=np.asarray(context.thermal_mass_reduced,float)
    worst=0.0; maximum_circuit_condition=1.0; maximum_integrator_error=0.0; rows=[]
    zero=np.zeros(model.thermal_rank,float)
    for time,a_truth,a_pred in zip(times,truth_states,predicted_states):
        state_error=_mass_relative(a_pred-a_truth,a_truth,mass)
        theta_truth=phi@a_truth; theta_pred=phi@a_pred; scale=max(float(np.max(np.abs(theta_truth))),np.finfo(float).tiny)
        tmin=abs(float(np.min(theta_pred)-np.min(theta_truth)))/scale; tmax=abs(float(np.max(theta_pred)-np.max(theta_truth)))/scale
        wire_error=0.0
        for weights in context.line_heat_weights:
            weights=np.asarray(weights,float); ref=float(weights@theta_truth); val=float(weights@theta_pred)
            wire_error=max(wire_error,abs(val-ref)/max(abs(ref),1e-10*scale,np.finfo(float).tiny))
        z_truth=truth.z_field+np.diag(model.background.wire_resistances(context,a_truth)); z_pred=predicted.z_field+np.diag(model.background.wire_resistances(context,a_pred))
        z_error=_relative(z_pred,z_truth,np.linalg.norm(z_truth))
        currents_truth=model._currents(a_truth,context,truth,operating); currents_pred=model._currents(a_pred,context,predicted,operating)
        current_error=_relative(currents_pred,currents_truth,np.linalg.norm(currents_truth))
        condition=max(_circuit_condition(model,context,truth,operating,a_truth),_circuit_condition(model,context,predicted,operating,a_pred)); maximum_circuit_condition=max(maximum_circuit_condition,condition)
        production=model.predict(float(time),initial_state=zero,geometry=geometry,operating=operating,
                                 max_step=min(float(integrator_max_step),float(time)),method="etd2_adaptive",
                                 rtol=float(integrator_rtol),atol=float(integrator_atol))
        integrator_error=_mass_relative(np.asarray(production.state)-a_pred,a_pred,mass); maximum_integrator_error=max(maximum_integrator_error,integrator_error)
        composite=max(state_error,tmin,tmax,wire_error,z_error,current_error); worst=max(worst,composite)
        rows.append({"time":float(time),"state_mass_relative_error":state_error,"minimum_temperature_relative_error":float(tmin),
                     "maximum_temperature_relative_error":float(tmax),"maximum_wire_temperature_relative_error":float(wire_error),
                     "impedance_relative_error":z_error,"current_relative_error":current_error,"integrator_reference_relative_error":integrator_error,
                     "circuit_condition":condition,"composite_relative_error":float(composite)})
    a_truth,r_truth,stable_truth,alpha_truth=_steady(model,context,truth,operating,truth_states[-1])
    a_pred,r_pred,stable_pred,alpha_pred=_steady(model,context,predicted,operating,predicted_states[-1])
    theta_truth=phi@a_truth; theta_pred=phi@a_pred; scale=max(float(np.max(np.abs(theta_truth))),np.finfo(float).tiny)
    steady_error=max(_relative(a_pred,a_truth,np.linalg.norm(a_truth)),abs(float(np.min(theta_pred)-np.min(theta_truth)))/scale,abs(float(np.max(theta_pred)-np.max(theta_truth)))/scale)
    for weights in context.line_heat_weights:
        weights=np.asarray(weights,float); ref=float(weights@theta_truth); val=float(weights@theta_pred)
        steady_error=max(steady_error,abs(val-ref)/max(abs(ref),1e-10*scale,np.finfo(float).tiny))
    z_truth=truth.z_field+np.diag(model.background.wire_resistances(context,a_truth)); z_pred=predicted.z_field+np.diag(model.background.wire_resistances(context,a_pred))
    steady_error=max(steady_error,_relative(z_pred,z_truth,np.linalg.norm(z_truth)))
    maximum_circuit_condition=max(maximum_circuit_condition,_circuit_condition(model,context,truth,operating,a_truth),_circuit_condition(model,context,predicted,operating,a_pred)); worst=max(worst,steady_error)
    return {"maximum_relative_error":float(worst),"maximum_integrator_relative_error":float(maximum_integrator_error),
            "maximum_circuit_condition":float(maximum_circuit_condition),"truth_steady_residual":r_truth,"predicted_steady_residual":r_pred,
            "truth_steady_stable":bool(stable_truth),"predicted_steady_stable":bool(stable_pred),
            "truth_steady_spectral_abscissa":alpha_truth,"predicted_steady_spectral_abscissa":alpha_pred,
            "steady_relative_error":float(steady_error),"times":rows}


def _audit_operating_cases(settings,cfg):
    configured=cfg.get("operating_cases")
    if configured is not None:
        cases=[]
        for index,item in enumerate(list(configured)):
            if not isinstance(item,dict): raise ValueError("final_audit.operating_cases entries must be mappings")
            name=str(item.get("name",f"case_{index}"))
            if "drive" in item: value=item["drive"]
            elif "operating" in item: value=item["operating"]
            else: raise ValueError("each final audit operating case needs 'operating' or 'drive'")
            cases.append((name,value))
        if not cases: raise ValueError("final_audit.operating_cases cannot be empty")
        return cases
    if "drive" in cfg: return [("circuit",cfg["drive"])]
    if "operating" in cfg: return [("current",cfg["operating"])]
    prediction=settings["PREDICTION"]; value=prediction.get("drive",prediction.get("operating"))
    if value is None: raise ValueError("final audit requires an explicit operating or drive setting")
    return [("prediction_default",value)]


def run_final_held_out_audit(settings,model,geometries,monitor=None):
    geometries=list(geometries)
    if not geometries: raise ValueError("final held-out audit needs at least one geometry")
    cfg=dict(settings["TRAINING"].get("final_audit",{}))
    times=np.asarray(cfg.get("times",settings["TRAINING"].get("thermal_trajectory_times",[0.1,1.0,10.0,100.0])),float)
    tensor_tol=float(cfg.get("tensor_relative_tolerance",2e-1)); current_tol=float(cfg.get("current_space_relative_tolerance",2e-1)); outward_tol=float(cfg.get("outward_relative_tolerance",2e-1))
    projection_limit=float(cfg.get("projection_correction_limit",2e-1)); dynamic_tol=float(cfg.get("reduced_dynamic_relative_tolerance",1e-1))
    thermal_tol=float(cfg.get("full_vs_rom_thermal_tolerance",settings["TRAINING"].get("thermal_basis_energy_tolerance",5e-2))); circuit_limit=float(cfg.get("circuit_condition_limit",1e8))
    integrator_tol=float(cfg.get("integrator_relative_tolerance",1e-4)); integrator_rtol=float(cfg.get("integrator_rtol",1e-7)); integrator_atol=float(cfg.get("integrator_atol",1e-9)); integrator_max_step=float(cfg.get("integrator_max_step",10.0))
    if integrator_tol<=0.0 or integrator_rtol<=0.0 or integrator_atol<=0.0 or integrator_max_step<=0.0: raise ValueError("final audit integrator tolerances/max_step must be positive")
    operating_cases=_audit_operating_cases(settings,cfg)
    thermal_error,thermal_worst,thermal_diag,audited_times=audit_geometry_aware_thermal_trajectories(model.background,model.background.thermal_library,geometries,times=times,monitor=monitor)
    rows=[]
    for index,geometry in enumerate(geometries):
        if monitor is not None: monitor.checkpoint()
        truth,predicted,tensor=_tensor_case(model,geometry); trajectories=[]
        for name,operating in operating_cases:
            trajectory=_trajectory_case(model,geometry,truth,predicted,operating,times,integrator_rtol=integrator_rtol,integrator_atol=integrator_atol,integrator_max_step=integrator_max_step)
            trajectories.append({"name":name,"operating":operating,**trajectory})
        dynamic=max(row["maximum_relative_error"] for row in trajectories); integrator=max(row["maximum_integrator_relative_error"] for row in trajectories)
        rows.append({"index":int(index),"tensor":tensor,"operating_cases":trajectories})
        print(f"completely-held-out final audit……{index+1}/{len(geometries)} tensor={max(tensor['z_relative_error'],tensor['d_relative_error'],tensor['h_relative_error']):.3e} dynamic={dynamic:.3e} integrator={integrator:.3e}",flush=True)
    trajectory_rows=[case for row in rows for case in row["operating_cases"]]
    maximum_tensor=max(max(row["tensor"][key] for key in ("z_relative_error","d_relative_error","h_relative_error")) for row in rows)
    maximum_current=max(row["tensor"]["maximum_current_space_relative_error"] for row in rows); maximum_outward=max(row["tensor"]["outward_relative_error"] for row in rows)
    maximum_projection=max(max(row["tensor"]["zd_projection_correction"],row["tensor"]["h_projection_correction"]) for row in rows)
    maximum_dynamic=max(row["maximum_relative_error"] for row in trajectory_rows); maximum_integrator=max(row["maximum_integrator_relative_error"] for row in trajectory_rows); maximum_circuit=max(row["maximum_circuit_condition"] for row in trajectory_rows)
    steady_ok=all(row["truth_steady_residual"]<=1e-10 and row["predicted_steady_residual"]<=1e-10 and row["truth_steady_stable"] and row["predicted_steady_stable"] for row in trajectory_rows)
    checks={"full_vs_rom_thermal_ok":thermal_error<=thermal_tol,"tensor_surrogate_ok":maximum_tensor<=tensor_tol,
            "current_space_contractions_ok":maximum_current<=current_tol,"physical_outward_loss_ok":maximum_outward<=outward_tol,
            "decoder_projection_correction_ok":maximum_projection<=projection_limit,"reduced_end_to_end_dynamics_ok":maximum_dynamic<=dynamic_tol,
            "production_integrator_ok":maximum_integrator<=integrator_tol,"steady_state_ok":steady_ok,"circuit_conditioning_ok":maximum_circuit<=circuit_limit}
    certified=all(bool(value) for value in checks.values())
    return {**{key:bool(value) for key,value in checks.items()},"sample_count":len(rows),"times":[float(v) for v in times],
            "operating_case_names":[name for name,_ in operating_cases],"maximum_full_vs_rom_thermal_relative_error":float(thermal_error),
            "worst_full_vs_rom_thermal":thermal_worst,"full_vs_rom_thermal_diagnostics":thermal_diag,"full_vs_rom_audited_times":[float(v) for v in audited_times],
            "maximum_tensor_relative_error":float(maximum_tensor),"maximum_current_space_relative_error":float(maximum_current),
            "maximum_outward_relative_error":float(maximum_outward),"maximum_projection_correction":float(maximum_projection),
            "maximum_reduced_dynamic_relative_error":float(maximum_dynamic),"maximum_integrator_relative_error":float(maximum_integrator),
            "maximum_circuit_condition":float(maximum_circuit),
            "thresholds":{"full_vs_rom_thermal_tolerance":thermal_tol,"tensor_relative_tolerance":tensor_tol,"current_space_relative_tolerance":current_tol,
                          "outward_relative_tolerance":outward_tol,"projection_correction_limit":projection_limit,"reduced_dynamic_relative_tolerance":dynamic_tol,
                          "integrator_relative_tolerance":integrator_tol,"integrator_rtol":integrator_rtol,"integrator_atol":integrator_atol,
                          "integrator_max_step":integrator_max_step,"circuit_condition_limit":circuit_limit},
            "samples":rows,"certified":bool(certified),"status":"certified" if certified else "final_audit_failed",
            "certificate_level":"frozen_held_out_numerical_validation"}


__all__=["run_final_held_out_audit"]