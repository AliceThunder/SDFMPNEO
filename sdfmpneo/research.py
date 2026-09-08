"""Reproducible electromagnetic--thermal training and inference for research."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve

from .analytic import ParametricAnalyticEvolutionGraph, evaluate_parametric_stable
from .electrothermal import CertifiedElectroThermalVectorField
from .em import (ConductivityRegion, ConstantConductivity, ReciprocalLinearResistivity,
                 AffineConductivity, NonlinearTetrahedralApsiProblem, ImpressedCurrentPortSet,
                 SolidTerminalPortSet, SparseEnergyReducedEMModel, SparseEnergyResidualGreedyEMReducer,
                 tetra_face_loop_source)
from .em.riesz_action import SparseLUReferenceRieszAction
from .em.energy_solver import apsi_physical_energy_metric
from .em.tetra_nonlinear_diagnostics import NonlinearTetrahedralRegionLossEvaluator
from .model import ParametricExecutableSDFMPNEOModel
from .spatial import TetrahedralComplex3D, read_gmsh_v22_ascii
from .thermal import ThermalSpectralModel
from .tetra_core import TetrahedralElectroThermalCore
from .training import AffineOperatingRHSMap
from .training.research import ResearchTrainingConfig, ResearchTrainingReport, train_research_graph


class ResearchElectroThermalModel:
    """One fixed spatial geometry/frequency, arbitrary initial state/current/time.

    The trained DAG alone produces the thermal state. Optional diagnostics use
    the same reduced electromagnetic equilibrium for impedance and Joule losses.
    Save/load preserves the actual thermal and EM bases, without retraining or
    recomputing a potentially sign-ambiguous eigenspace.
    """
    def __init__(self, core, electromagnetic_model, ports, rhs_map, *, graph=None,
                 training_config=None, training_report=None):
        self.core = core
        self.em = electromagnetic_model
        self.ports = ports
        self.rhs_map = rhs_map
        baseline = self.reference_temperature
        if np.any(~np.isfinite(baseline)) or not np.all(baseline == baseline[0]):
            raise ValueError("research workflow requires a constant reference temperature; "
                             "a nonuniform reference needs an explicit conductive forcing term")
        self.field = CertifiedElectroThermalVectorField(core.thermal_model, self.em, rhs_map=rhs_map)
        self.graph = graph
        self.training_config = training_config
        self.training_report = training_report

    @classmethod
    def build(cls, core, ports, *, candidate_states, requested_em_error,
              current_offset=None, current_matrix=None):
        # Direct sparse Riesz factorization is an explicit scientific reference
        # backend. It never uses an electromagnetic solution snapshot A^-1 b.
        reducer = SparseEnergyResidualGreedyEMReducer(core.electromagnetic_problem,
                                                     riesz_action_factory=SparseLUReferenceRieszAction)
        p = ports.n_ports
        offset = np.zeros(p, complex) if current_offset is None else np.asarray(current_offset, complex)
        matrix = np.eye(p, dtype=complex) if current_matrix is None else np.asarray(current_matrix, complex)
        if offset.shape != (p,) or matrix.ndim != 2 or matrix.shape[0] != p:
            raise ValueError("current offset/matrix must match port count")
        rhs = AffineOperatingRHSMap(ports.coordinate_rhs @ offset, ports.coordinate_rhs @ matrix)
        em = reducer.build_multi_rhs(candidate_states, ports.coordinate_rhs,
                                      requested_energy_state_error=requested_em_error)
        if not em.reduction_certificate.certified:
            raise RuntimeError("EM basis did not reach requested error; inspect the reduction certificate")
        return cls(core, em, ports, rhs)

    @property
    def reference_temperature(self):
        problem = self.core.electromagnetic_problem
        baseline = np.zeros(self.core.mesh.n_nodes)
        for tet, values in zip(self.core.mesh.tetrahedra, problem.temperature_reference_local):
            baseline[tet] = values
        return baseline

    def temperature(self, a):
        deviation = self.core.thermal_assembly.expand_free(self.core.thermal_model.reconstruct(a))
        return self.reference_temperature + deviation

    def project_initial_temperature(self, temperature_nodal):
        T = np.asarray(temperature_nodal, float)
        if T.shape != (self.core.mesh.n_nodes,):
            raise ValueError("initial temperature must contain every mesh node")
        assembly = self.core.thermal_assembly
        baseline = self.reference_temperature
        if not np.allclose(T[assembly.boundary_nodes], baseline[assembly.boundary_nodes]):
            raise ValueError("initial boundary temperature differs from the prescribed ambient")
        return self.core.thermal_model.project((T-baseline)[assembly.free_nodes])

    def train(self, config: ResearchTrainingConfig, *, progress=None, monitor=None):
        from .training.monitor import TrainingStopped
        try:
            graph, report = train_research_graph(self.field, config, graph=self.graph,
                                                progress=progress, monitor=monitor)
        except TrainingStopped:
            if monitor is not None and monitor.best_graph is not None:
                self.graph = monitor.best_graph
                self.training_config = config
                self.training_report = None
            raise
        self.graph, self.training_config, self.training_report = graph, config, report
        return report

    def predict(self, time, *, a0, operating, diagnostics=True, allow_extrapolation=False):
        if self.graph is None:
            raise ValueError("train or load a trained model first")
        initial = np.asarray(a0, float)
        u = np.asarray(operating, float)
        time = float(time)
        if not np.isfinite(time) or time < 0 or np.any(~np.isfinite(initial)) or np.any(~np.isfinite(u)):
            raise ValueError("inference inputs must be finite and time non-negative")
        if self.training_config is not None and not allow_extrapolation:
            c = self.training_config
            p = np.concatenate([initial, u, [time]])
            lo = np.array(c.initial_lower+c.operating_lower+(0.,))
            hi = np.array(c.initial_upper+c.operating_upper+(c.time_horizon,))
            if p.shape != lo.shape or np.any(p < lo) or np.any(p > hi):
                raise ValueError("query is outside the trained box; set allow_extrapolation=True for an explicit study")
        if not diagnostics:
            a, da = evaluate_parametric_stable(self.graph, time, a0=initial, operating=u)
            T = self.temperature(a)
            return {'time': time, 'thermal_coordinates': a, 'thermal_derivative': da,
                    'temperature_field': T, 'maximum_temperature': float(np.max(T))}
        online = ParametricExecutableSDFMPNEOModel(
            evolution=self.graph, thermal_model=self.core.thermal_model,
            electromagnetic_model=self.em, rhs_map=self.rhs_map, ports=self.ports,
            region_loss_projector=NonlinearTetrahedralRegionLossEvaluator(self.em.problem),
            thermal_assembly=self.core.thermal_assembly, temperature_reference=self.reference_temperature)
        return online.evaluate(time, a0=initial, operating=u)

    def validate_trajectory(self, times, *, a0, operating, full_electromagnetics=True,
                            rtol=1e-8, atol=1e-10):
        """Validation-only transient reference; never called during training.

        The reference uses an independent implicit time integrator and, by
        default, full-order sparse electromagnetic equilibrium. Thermal spatial
        coordinates are the saved thermal ROM: this separates NN/EM-ROM error
        from the additional thermal projection and mesh errors.
        """
        times = np.asarray(times, float)
        if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must be a nonempty finite non-negative vector")
        rhs = self.field.rhs(np.asarray(operating, float))
        problem = self.em.problem
        def reference_rhs(t, a):
            if full_electromagnetics:
                x = spsolve(problem.operator_sparse(a).tocsc(), rhs)
                q = np.array([np.vdot(x, problem.loss_operator_sparse(j,a) @ x).real
                              for j in range(problem.n_thermal)])
            else:
                q = self.em.heat_source_for_rhs(a, rhs)
            return -self.core.thermal_model.lambdas*a + q
        if times.max() == 0:
            reference = np.tile(np.asarray(a0,float), (len(times),1))
        else:
            solution = solve_ivp(reference_rhs, (0.,float(times.max())), np.asarray(a0,float),
                                 method='Radau', dense_output=True, rtol=rtol, atol=atol)
            if not solution.success:
                raise RuntimeError(solution.message)
            reference = solution.sol(times).T
        predicted = np.array([self.predict(t,a0=a0,operating=operating,diagnostics=False)['thermal_coordinates']
                              for t in times])
        temperature_error = np.array([self.temperature(p)-self.temperature(r)
                                      for p,r in zip(predicted,reference)])
        return {'times': times, 'predicted_coordinates': predicted, 'reference_coordinates': reference,
                'maximum_coordinate_error': float(np.max(np.linalg.norm(predicted-reference,axis=1))),
                'maximum_temperature_error': float(np.max(np.abs(temperature_error))),
                'reference': 'Radau + full sparse EM' if full_electromagnetics else 'Radau + EM ROM',
                'used_for_training': False}

    def save(self, path):
        if self.graph is None:
            raise ValueError("no trained graph to save")
        core, problem = self.core, self.em.problem
        regions = []
        arrays = {}
        for i, region in enumerate(problem.conductivity_regions):
            regions.append({'name':region.name,'law':type(region.law).__name__,'parameters':asdict(region.law)})
            arrays[f'region_{i}'] = np.asarray(region.mask, bool)
        metadata = {'format_version':1, 'omega':problem.omega, 'regions':regions,
                    'constitutive_error':problem.constitutive_relative_error_budget,
                    'port_names':list(self.ports.names), 'operating_names':list(self.graph.operating_names),
                    'thermal_backend':core.thermal_spectrum_backend,
                    'nodes':[{'name':n.name,'target_mode':n.target_mode,'parents':list(n.parents),
                              'weight':[n.weight.real,n.weight.imag]} for n in self.graph.response_nodes],
                    'training_config':None if self.training_config is None else asdict(self.training_config),
                    'training_report':None if self.training_report is None else self.training_report.to_dict()}
        arrays.update(vertices=core.mesh.vertices, tetrahedra=core.mesh.tetrahedra,
                      thermal_phi=core.thermal_model.Phi, thermal_lambdas=core.thermal_model.lambdas,
                      reference_temperature=self.reference_temperature, reluctivity=problem.reluctivity_tetra,
                      source_current=problem.source_current, em_basis=self.em.V,
                      port_edges=self.ports.edge_currents, port_rhs=self.ports.coordinate_rhs,
                      rhs_offset=self.rhs_map.offset, rhs_matrix=self.rhs_map.matrix)
        # Persist the assembled thermal matrices; loading preserves the selected
        # eigenspace exactly and does not need the original material JSON.
        for name,matrix in [('M',core.thermal_assembly.M),('K',core.thermal_assembly.K)]:
            csr=sp.csr_matrix(matrix)
            arrays.update({name+'_data':csr.data,name+'_indices':csr.indices,name+'_indptr':csr.indptr})
        arrays['thermal_free_nodes']=core.thermal_assembly.free_nodes
        arrays['metadata']=np.array(json.dumps(metadata))
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as output:
            np.savez_compressed(output,**arrays)
        return path

    @classmethod
    def load(cls,path):
        from .spatial.tetra3d import TetrahedralThermalAssembly
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta['format_version'] != 1:
                raise ValueError('unsupported model format version')
            mesh=TetrahedralComplex3D.build(data['vertices'],data['tetrahedra'])
            matrices={name:sp.csr_matrix((data[name+'_data'],data[name+'_indices'],data[name+'_indptr']),
                                        shape=(len(data['thermal_free_nodes']),len(data['thermal_free_nodes']))) for name in ['M','K']}
            free=data['thermal_free_nodes']
            M,K=matrices['M'],matrices['K']
            assembly=TetrahedralThermalAssembly(M=M,K=K,free_nodes=free,boundary_nodes=mesh.boundary_nodes(),
                                                n_full_nodes=mesh.n_nodes,tetrahedra=mesh.tetrahedra)
            thermal=ThermalSpectralModel(assembly.M,assembly.K,data['thermal_phi'],data['thermal_lambdas'])
            local=np.array([assembly.expand_free(thermal.Phi[:,i])[mesh.tetrahedra] for i in range(thermal.rank)])
            laws={c.__name__:c for c in (ConstantConductivity,AffineConductivity,ReciprocalLinearResistivity)}
            regions=tuple(ConductivityRegion(r['name'],data[f'region_{i}'],laws[r['law']](**r['parameters']))
                          for i,r in enumerate(meta['regions']))
            problem=NonlinearTetrahedralApsiProblem(mesh,omega=meta['omega'],reluctivity_tetra=data['reluctivity'],
                        source_current=data['source_current'],temperature_reference_local=data['reference_temperature'][mesh.tetrahedra],
                        thermal_modes_local=local,conductivity_regions=regions,constitutive_relative_error_budget=meta['constitutive_error'])
            core=TetrahedralElectroThermalCore(mesh,assembly,None,thermal,None,local,meta['thermal_backend'],
                                               'certified_nonlinear',None,None,regions,problem,problem)
            em=SparseEnergyReducedEMModel(problem,data['em_basis'],reference_energy_metric=apsi_physical_energy_metric(
                problem.operator_sparse(np.zeros(thermal.rank))),riesz_action_factory=SparseLUReferenceRieszAction)
            ports=ImpressedCurrentPortSet(tuple(meta['port_names']),data['port_edges'],data['port_rhs'],meta['omega'])
            rhs=AffineOperatingRHSMap(data['rhs_offset'],data['rhs_matrix'])
            graph=ParametricAnalyticEvolutionGraph(thermal.lambdas,meta['operating_names'])
            for node in meta['nodes']:
                graph.add_product_response(node['name'],node['target_mode'],node['parents'],complex(*node['weight']))
            config=meta['training_config']
            if config is not None:
                for key in ['initial_lower','initial_upper','operating_lower','operating_upper']:
                    config[key]=tuple(config[key])
                config=ResearchTrainingConfig(**config)
            report=meta['training_report']
            if report is not None:
                report['objective_history']=tuple(report['objective_history'])
                report=ResearchTrainingReport(**report)
            return cls(core,em,ports,rhs,graph=graph,training_config=config,training_report=report)


def demo_research_model():
    """Small volumetric two-port Cu/seawater electrothermal regression case.

    Four tetrahedra and one interior thermal mode make the complete workflow
    inexpensive. This is a discretization example, not a calibrated coil design.
    """
    xyz=.03*np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.25,.25,.25]],float)
    mesh=TetrahedralComplex3D.build(xyz,np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]]))
    copper=np.array([True,True,False,False])
    regions=(ConductivityRegion('copper',copper,ReciprocalLinearResistivity(5.8e7,.00393,293.15)),
             ConductivityRegion('seawater',~copper,ConstantConductivity(5.)))
    currents=np.column_stack([tetra_face_loop_source(mesh,int(f)) for f in mesh.boundary_face_indices[:2]])
    core=TetrahedralElectroThermalCore.build_nonlinear(mesh,omega=2*np.pi*1e5,
         reluctivity_tetra=np.ones(4)/(4e-7*np.pi),conductivity_regions=regions,
         temperature_reference_nodal=np.full(5,293.15),constitutive_relative_error_budget=1e-8,
         rho_cp_tetra=np.where(copper,3.45e6,4.1e6),thermal_conductivity_tetra=np.where(copper,400.,.6),
         source_current=currents[:,0])
    ports=core.build_ports(currents,names=['tx','rx'])
    return ResearchElectroThermalModel.build(core,ports,candidate_states=[np.array([v]) for v in [0.,2.,5.]],
        requested_em_error=1e-8)


def model_from_config(path):
    """Read a tagged Gmsh mesh + SI material/terminal/training configuration."""
    path=Path(path)
    config=json.loads(path.read_text())
    tagged=read_gmsh_v22_ascii(path.parent/config['mesh'])
    mesh=tagged.mesh
    nu=np.zeros(mesh.n_tetrahedra); capacity=nu.copy(); conductivity=nu.copy()
    regions=[]
    ambient=float(config.get('ambient_temperature',293.15))
    covered=np.zeros(mesh.n_tetrahedra,bool)
    for tag, material in config['materials'].items():
        mask=tagged.tetra_mask(int(tag)); covered |= mask
        sigma=float(material['electrical_conductivity'])
        alpha=float(material.get('resistivity_temperature_coefficient',0.))
        law=(ReciprocalLinearResistivity(sigma,alpha,float(material.get('reference_temperature',ambient)))
             if alpha else ConstantConductivity(sigma))
        regions.append(ConductivityRegion(material['name'],mask,law))
        nu[mask]=1/(4e-7*np.pi*float(material.get('relative_permeability',1.)))
        capacity[mask]=float(material['volumetric_heat_capacity'])
        conductivity[mask]=float(material['thermal_conductivity'])
    if not np.all(covered):
        raise ValueError('material definitions must cover all tetrahedral physical tags')
    core=TetrahedralElectroThermalCore.build_nonlinear(mesh,omega=2*np.pi*float(config['frequency_hz']),
        reluctivity_tetra=nu,conductivity_regions=regions,temperature_reference_nodal=np.full(mesh.n_nodes,ambient),
        constitutive_relative_error_budget=float(config.get('constitutive_relative_error',1e-8)),
        rho_cp_tetra=capacity,thermal_conductivity_tetra=conductivity,source_current=np.zeros(mesh.n_edges),
        thermal_rank=config.get('thermal_rank'), **config.get('thermal_truncation',{}))
    ports=SolidTerminalPortSet.build(tagged,core.electromagnetic_problem,config['terminal_pairs'],
                                    names=config.get('port_names'))
    train=dict(config['training'])
    for key in ['initial_lower','initial_upper','operating_lower','operating_upper']:
        train[key]=tuple(train[key])
    training=ResearchTrainingConfig(**train)
    n=core.thermal_model.rank
    if len(training.initial_lower)!=n:
        raise ValueError(f'training initial-state bounds must contain {n} retained thermal coordinates')
    candidates=config.get('em_candidate_states',[training.initial_lower,training.initial_upper,
                ((np.array(training.initial_lower)+training.initial_upper)/2).tolist()])
    # Complex current coefficients are represented by JSON strings, e.g. "1j".
    offset=config.get('current_offset')
    matrix=config.get('current_matrix')
    result=ResearchElectroThermalModel.build(core,ports,candidate_states=np.asarray(candidates,float),
        requested_em_error=float(config['em_energy_error']),current_offset=offset,current_matrix=matrix)
    return result,training
