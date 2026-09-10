"""Reproducible electromagnetic-thermal training and segmented inference."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve

from .analytic.fixed_response_network import FixedAnalyticResponseNetwork
from .electrothermal import CertifiedElectroThermalVectorField
from .em import (
    AffineConductivity,
    ConductivityRegion,
    ConstantConductivity,
    ImpressedCurrentPortSet,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SolidTerminalPortSet,
    SparseEnergyReducedEMModel,
    SparseEnergyResidualGreedyEMReducer,
    tetra_face_loop_source,
)
from .em.energy_solver import apsi_physical_energy_metric
from .em.riesz_action import SparseLUReferenceRieszAction
from .em.tetra_nonlinear_diagnostics import NonlinearTetrahedralRegionLossEvaluator
from .model import _port_outputs
from .rollout import rollout_fixed_network, solve_physical_steady_state
from .spatial import TetrahedralComplex3D, read_gmsh_v22_ascii
from .thermal import ThermalSpectralModel
from .tetra_core import TetrahedralElectroThermalCore
from .training import AffineOperatingRHSMap
from .training.automatic_thermal_rank import (
    automatic_thermal_rank,
    resolve_training_bounds,
    strict_truncation_kwargs,
)
from .training.research import (
    ResearchTrainingConfig,
    ResearchTrainingReport,
    train_research_network,
)

_MODEL_FORMAT_VERSION = 5


class ResearchElectroThermalModel:
    """One fixed geometry/frequency with one finite-horizon analytic flow network."""

    def __init__(
        self,
        core,
        electromagnetic_model,
        ports,
        rhs_map,
        *,
        network=None,
        training_config=None,
        training_report=None,
    ):
        self.core = core
        self.em = electromagnetic_model
        self.ports = ports
        self.rhs_map = rhs_map
        baseline = self.reference_temperature
        if np.any(~np.isfinite(baseline)) or not np.all(baseline == baseline[0]):
            raise ValueError("research workflow requires a constant reference temperature")
        self.field = CertifiedElectroThermalVectorField(
            core.thermal_model, self.em, rhs_map=rhs_map
        )
        self.network = network
        self.training_config = training_config
        self.training_report = training_report
        self.thermal_rank_report = None

    @classmethod
    def build(
        cls,
        core,
        ports,
        *,
        candidate_states,
        requested_em_error,
        current_offset=None,
        current_matrix=None,
    ):
        reducer = SparseEnergyResidualGreedyEMReducer(
            core.electromagnetic_problem,
            riesz_action_factory=SparseLUReferenceRieszAction,
        )
        p = ports.n_ports
        offset = (
            np.zeros(p, complex)
            if current_offset is None
            else np.asarray(current_offset, complex)
        )
        matrix = (
            np.eye(p, dtype=complex)
            if current_matrix is None
            else np.asarray(current_matrix, complex)
        )
        if offset.shape != (p,) or matrix.ndim != 2 or matrix.shape[0] != p:
            raise ValueError("current offset/matrix must match port count")
        rhs = AffineOperatingRHSMap(
            ports.coordinate_rhs @ offset,
            ports.coordinate_rhs @ matrix,
        )
        em = reducer.build_multi_rhs(
            candidate_states,
            ports.coordinate_rhs,
            requested_energy_state_error=requested_em_error,
        )
        if not em.reduction_certificate.certified:
            raise RuntimeError("EM basis did not reach requested error")
        return cls(core, em, ports, rhs)

    @property
    def reference_temperature(self):
        problem = self.core.electromagnetic_problem
        baseline = np.zeros(self.core.mesh.n_nodes)
        for tet, values in zip(
            self.core.mesh.tetrahedra, problem.temperature_reference_local
        ):
            baseline[tet] = values
        return baseline

    def temperature(self, a):
        deviation = self.core.thermal_assembly.expand_free(
            self.core.thermal_model.reconstruct(a)
        )
        return self.reference_temperature + deviation

    def project_initial_temperature(self, temperature_nodal):
        T = np.asarray(temperature_nodal, float)
        if T.shape != (self.core.mesh.n_nodes,):
            raise ValueError("initial temperature must contain every mesh node")
        assembly = self.core.thermal_assembly
        baseline = self.reference_temperature
        if not np.allclose(
            T[assembly.boundary_nodes], baseline[assembly.boundary_nodes]
        ):
            raise ValueError(
                "initial boundary temperature differs from the prescribed ambient"
            )
        return self.core.thermal_model.project(
            (T - baseline)[assembly.free_nodes]
        )

    def prepare_training_contexts(self, points):
        return None

    def train(self, config: ResearchTrainingConfig, *, progress=None, monitor=None):
        from .training.monitor import TrainingStopped

        try:
            network, report = train_research_network(
                self.field,
                config,
                network=self.network,
                operating_names=tuple(
                    f"current_{i}" for i in range(self.rhs_map.n_operating)
                ),
                progress=progress,
                monitor=monitor,
            )
        except TrainingStopped:
            if monitor is not None and monitor.best_network is not None:
                self.network = monitor.best_network
                self.training_config = config
                self.training_report = None
            raise
        self.network = network
        self.training_config = config
        self.training_report = report
        return report

    def _validated_inputs(self, a0, operating, *, allow_extrapolation):
        if self.network is None:
            raise ValueError("train or load a trained model first")
        initial = np.asarray(a0, float)
        u = np.asarray(operating, float)
        if initial.shape != (self.network.n_modes,):
            raise ValueError("initial-state dimension does not match the saved model")
        if u.shape != (len(self.network.operating_names),):
            raise ValueError("operating dimension does not match the saved model")
        if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(u)):
            raise ValueError("initial/current inputs must be finite")
        cfg = self.training_config
        if cfg is not None and not allow_extrapolation:
            p = np.concatenate([initial, u])
            lo = np.asarray(cfg.initial_lower + cfg.operating_lower)
            hi = np.asarray(cfg.initial_upper + cfg.operating_upper)
            if np.any(p < lo) or np.any(p > hi):
                raise ValueError(
                    "query is outside the trained restart/operating box; "
                    "set allow_extrapolation=True explicitly"
                )
        return initial, u

    def _diagnostics(self, state, derivative, operating):
        rhs = self.rhs_map.evaluate(operating)
        physical = self.field.vector_field(state, operating)
        ports, certificate = _port_outputs(self.ports, self.em, state, 1e-6)
        x = self.em.state_for_rhs(state, rhs)
        return {
            "physical_residual": np.asarray(derivative - physical),
            "physical_residual_norm": float(np.linalg.norm(derivative - physical)),
            "impedance": ports,
            "impedance_certificate": certificate,
            "drive_rhs_residual_dual_norm": float(
                self.em.residual_dual_norm_for_rhs(state, rhs)
            ),
            "region_losses": NonlinearTetrahedralRegionLossEvaluator(
                self.em.problem
            ).evaluate_state(x, state),
        }

    def steady_state(
        self,
        *,
        a0,
        operating,
        diagnostics=True,
        allow_extrapolation=False,
        tolerance=None,
        max_iterations=40,
    ):
        initial, u = self._validated_inputs(
            a0, operating, allow_extrapolation=allow_extrapolation
        )
        if tolerance is None:
            tolerance = (
                self.training_config.residual_tolerance
                if self.training_config is not None
                else 1e-10
            )
        solved = solve_physical_steady_state(
            self.field,
            u,
            initial_guess=initial,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if not solved.converged:
            raise RuntimeError(
                f"physical steady-state solve did not converge; residual={solved.residual_norm:.6g}"
            )
        state = solved.state
        derivative = np.zeros_like(state)
        T = self.temperature(state)
        result = {
            "time": "steady",
            "thermal_coordinates": state,
            "thermal_derivative": derivative,
            "temperature_field": T,
            "maximum_temperature": float(np.max(T)),
            "steady_state": True,
            "steady_iterations": solved.iterations,
            "steady_residual_norm": solved.residual_norm,
            "segment_count": 0,
            "segment_durations": (),
            "max_response_time": self.network.max_response_time,
        }
        if diagnostics:
            result.update(self._diagnostics(state, derivative, u))
        return result

    def predict(
        self,
        time,
        *,
        a0,
        operating,
        diagnostics=True,
        allow_extrapolation=False,
    ):
        initial, u = self._validated_inputs(
            a0, operating, allow_extrapolation=allow_extrapolation
        )
        t = float(time)
        if np.isnan(t) or t < 0:
            raise ValueError("time must be non-negative or positive infinity")
        if np.isposinf(t):
            return self.steady_state(
                a0=initial,
                operating=u,
                diagnostics=diagnostics,
                allow_extrapolation=allow_extrapolation,
            )
        if not np.isfinite(t):
            raise ValueError("time must be finite or positive infinity")
        cfg = self.training_config
        lower = None if cfg is None else cfg.initial_lower
        upper = None if cfg is None else cfg.initial_upper
        rolled = rollout_fixed_network(
            self.network,
            t,
            a0=initial,
            operating=u,
            state_lower=lower,
            state_upper=upper,
            allow_extrapolation=allow_extrapolation,
        )
        state, derivative = rolled.state, rolled.derivative
        T = self.temperature(state)
        result = {
            "time": t,
            "thermal_coordinates": state,
            "thermal_derivative": derivative,
            "temperature_field": T,
            "maximum_temperature": float(np.max(T)),
            "steady_state": False,
            "segment_count": rolled.segment_count,
            "segment_durations": rolled.segment_durations,
            "max_response_time": self.network.max_response_time,
        }
        if diagnostics:
            result.update(self._diagnostics(state, derivative, u))
        return result

    def validate_trajectory(
        self,
        times,
        *,
        a0,
        operating,
        full_electromagnetics=True,
        rtol=1e-8,
        atol=1e-10,
    ):
        times = np.asarray(times, float)
        if (
            times.ndim != 1
            or times.size == 0
            or np.any(~np.isfinite(times))
            or np.any(times < 0)
        ):
            raise ValueError("times must be a nonempty finite non-negative vector")
        rhs = self.field.rhs(np.asarray(operating, float))
        problem = self.em.problem

        def reference_rhs(_t, a):
            if full_electromagnetics:
                x = spsolve(problem.operator_sparse(a).tocsc(), rhs)
                q = np.asarray(
                    [
                        np.vdot(
                            x, problem.loss_operator_sparse(j, a) @ x
                        ).real
                        for j in range(problem.n_thermal)
                    ]
                )
            else:
                q = self.em.heat_source_for_rhs(a, rhs)
            return -self.core.thermal_model.lambdas * a + q

        if times.max() == 0:
            reference = np.tile(np.asarray(a0, float), (len(times), 1))
        else:
            solution = solve_ivp(
                reference_rhs,
                (0.0, float(times.max())),
                np.asarray(a0, float),
                method="Radau",
                dense_output=True,
                rtol=rtol,
                atol=atol,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            reference = solution.sol(times).T
        predicted = np.asarray(
            [
                self.predict(
                    t,
                    a0=a0,
                    operating=operating,
                    diagnostics=False,
                )["thermal_coordinates"]
                for t in times
            ]
        )
        temperature_error = np.asarray(
            [
                self.temperature(p) - self.temperature(r)
                for p, r in zip(predicted, reference)
            ]
        )
        return {
            "times": times,
            "predicted_coordinates": predicted,
            "reference_coordinates": reference,
            "maximum_coordinate_error": float(
                np.max(np.linalg.norm(predicted - reference, axis=1))
            ),
            "maximum_temperature_error": float(
                np.max(np.abs(temperature_error))
            ),
            "reference": (
                "Radau + full sparse EM"
                if full_electromagnetics
                else "Radau + EM ROM"
            ),
            "used_for_training": False,
            "prediction_mode": "segmented finite-horizon rollout",
        }

    def save(self, path):
        if self.network is None:
            raise ValueError("no trained fixed analytic network to save")
        core, problem = self.core, self.em.problem
        arrays = {}
        regions = []
        for i, region in enumerate(problem.conductivity_regions):
            regions.append(
                {
                    "name": region.name,
                    "law": type(region.law).__name__,
                    "parameters": asdict(region.law),
                }
            )
            arrays[f"region_{i}"] = np.asarray(region.mask, bool)
        metadata = {
            "model_type": "research",
            "format_version": _MODEL_FORMAT_VERSION,
            "omega": problem.omega,
            "regions": regions,
            "constitutive_error": problem.constitutive_relative_error_budget,
            "port_names": list(self.ports.names),
            "thermal_backend": core.thermal_spectrum_backend,
            "network": self.network.to_metadata(),
            "training_config": (
                None if self.training_config is None else asdict(self.training_config)
            ),
            "training_report": (
                None if self.training_report is None else self.training_report.to_dict()
            ),
            "thermal_rank_report": self.thermal_rank_report,
        }
        arrays.update(
            vertices=core.mesh.vertices,
            tetrahedra=core.mesh.tetrahedra,
            thermal_phi=core.thermal_model.Phi,
            thermal_lambdas=core.thermal_model.lambdas,
            reference_temperature=self.reference_temperature,
            reluctivity=problem.reluctivity_tetra,
            source_current=problem.source_current,
            em_basis=self.em.V,
            port_edges=self.ports.edge_currents,
            port_rhs=self.ports.coordinate_rhs,
            rhs_offset=self.rhs_map.offset,
            rhs_matrix=self.rhs_map.matrix,
            network_parameters=self.network.parameters,
        )
        for name, matrix in (
            ("M", core.thermal_assembly.M),
            ("K", core.thermal_assembly.K),
        ):
            csr = sp.csr_matrix(matrix)
            arrays.update(
                {
                    name + "_data": csr.data,
                    name + "_indices": csr.indices,
                    name + "_indptr": csr.indptr,
                }
            )
        arrays["thermal_free_nodes"] = core.thermal_assembly.free_nodes
        arrays["metadata"] = np.array(json.dumps(metadata))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez_compressed(output, **arrays)
        return path

    @classmethod
    def load(cls, path):
        from .spatial.tetra3d import TetrahedralThermalAssembly

        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("model_type") == "geometry_research":
                from .geometry_research import GeometryResearchModel
                return GeometryResearchModel.load(path)
            if (
                meta.get("model_type") != "research"
                or meta.get("format_version") != _MODEL_FORMAT_VERSION
            ):
                raise ValueError(
                    "unsupported model format; retrain with the current segmented fixed network"
                )
            mesh = TetrahedralComplex3D.build(
                data["vertices"], data["tetrahedra"]
            )
            free = data["thermal_free_nodes"]
            matrices = {
                name: sp.csr_matrix(
                    (
                        data[name + "_data"],
                        data[name + "_indices"],
                        data[name + "_indptr"],
                    ),
                    shape=(len(free), len(free)),
                )
                for name in ("M", "K")
            }
            assembly = TetrahedralThermalAssembly(
                M=matrices["M"],
                K=matrices["K"],
                free_nodes=free,
                boundary_nodes=mesh.boundary_nodes(),
                n_full_nodes=mesh.n_nodes,
                tetrahedra=mesh.tetrahedra,
            )
            thermal = ThermalSpectralModel(
                assembly.M,
                assembly.K,
                data["thermal_phi"],
                data["thermal_lambdas"],
            )
            local = np.asarray(
                [
                    assembly.expand_free(thermal.Phi[:, i])[mesh.tetrahedra]
                    for i in range(thermal.rank)
                ]
            )
            laws = {
                c.__name__: c
                for c in (
                    ConstantConductivity,
                    AffineConductivity,
                    ReciprocalLinearResistivity,
                )
            }
            regions = tuple(
                ConductivityRegion(
                    r["name"],
                    data[f"region_{i}"],
                    laws[r["law"]](**r["parameters"]),
                )
                for i, r in enumerate(meta["regions"])
            )
            problem = NonlinearTetrahedralApsiProblem(
                mesh,
                omega=meta["omega"],
                reluctivity_tetra=data["reluctivity"],
                source_current=data["source_current"],
                temperature_reference_local=data["reference_temperature"][
                    mesh.tetrahedra
                ],
                thermal_modes_local=local,
                conductivity_regions=regions,
                constitutive_relative_error_budget=meta["constitutive_error"],
            )
            core = TetrahedralElectroThermalCore(
                mesh,
                assembly,
                None,
                thermal,
                None,
                local,
                meta["thermal_backend"],
                "certified_nonlinear",
                None,
                None,
                regions,
                problem,
                problem,
            )
            em = SparseEnergyReducedEMModel(
                problem,
                data["em_basis"],
                reference_energy_metric=apsi_physical_energy_metric(
                    problem.operator_sparse(np.zeros(thermal.rank))
                ),
                riesz_action_factory=SparseLUReferenceRieszAction,
            )
            ports = ImpressedCurrentPortSet(
                tuple(meta["port_names"]),
                data["port_edges"],
                data["port_rhs"],
                meta["omega"],
            )
            rhs = AffineOperatingRHSMap(
                data["rhs_offset"], data["rhs_matrix"]
            )
            network_meta = meta["network"]
            if (
                network_meta.get("format_version")
                != FixedAnalyticResponseNetwork.format_version
            ):
                raise ValueError(
                    "unsupported fixed-network checkpoint version; retrain with the current format"
                )
            network = FixedAnalyticResponseNetwork.from_metadata(
                network_meta, data["network_parameters"]
            )
            cfg = meta.get("training_config")
            if cfg is not None:
                for key in (
                    "initial_lower",
                    "initial_upper",
                    "operating_lower",
                    "operating_upper",
                ):
                    cfg[key] = tuple(cfg[key])
                cfg = ResearchTrainingConfig(**cfg)
            report = meta.get("training_report")
            if report is not None:
                report["objective_history"] = tuple(report["objective_history"])
                report = ResearchTrainingReport(**report)
            out = cls(
                core,
                em,
                ports,
                rhs,
                network=network,
                training_config=cfg,
                training_report=report,
            )
            out.thermal_rank_report = meta.get("thermal_rank_report")
            return out


def demo_research_model():
    xyz = 0.03 * np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.25, 0.25, 0.25],
        ],
        float,
    )
    mesh = TetrahedralComplex3D.build(
        xyz,
        np.array(
            [
                [4, 1, 2, 3],
                [0, 4, 2, 3],
                [0, 1, 4, 3],
                [0, 1, 2, 4],
            ]
        ),
    )
    copper = np.array([True, True, False, False])
    regions = (
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(5.8e7, 0.00393, 293.15),
        ),
        ConductivityRegion(
            "seawater",
            ~copper,
            ConstantConductivity(5.0),
        ),
    )
    currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, int(f))
            for f in mesh.boundary_face_indices[:2]
        ]
    )
    core = TetrahedralElectroThermalCore.build_nonlinear(
        mesh,
        omega=2 * np.pi * 1e5,
        reluctivity_tetra=np.ones(4) / (4e-7 * np.pi),
        conductivity_regions=regions,
        temperature_reference_nodal=np.full(5, 293.15),
        constitutive_relative_error_budget=1e-8,
        rho_cp_tetra=np.where(copper, 3.45e6, 4.1e6),
        thermal_conductivity_tetra=np.where(copper, 400.0, 0.6),
        source_current=currents[:, 0],
    )
    ports = core.build_ports(currents, names=["tx", "rx"])
    return ResearchElectroThermalModel.build(
        core,
        ports,
        candidate_states=[np.array([v]) for v in [0.0, 2.0, 5.0]],
        requested_em_error=1e-8,
    )


def model_from_config(path, *, monitor=None):
    """Build the current model; automatic thermal rank is resolved before EM reduction."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    tagged = read_gmsh_v22_ascii(path.parent / config["mesh"])
    mesh = tagged.mesh
    truncation = dict(config.get("thermal_truncation") or {})
    thermal_rank = config.get("thermal_rank")
    strict_keys = (
        "initial_temperature_deviation_free",
        "source_dual_bound",
        "requested_state_tolerance",
    )
    strict = [truncation.get(key) is not None for key in strict_keys]
    if any(strict) and not all(strict):
        raise ValueError("strict thermal certificate inputs must be supplied together")
    rank_report = None
    training_payload = dict(config["training"])
    mode = truncation.get("mode")
    if (
        thermal_rank is None
        and mode in {
            "auto",
            "automatic",
            "physics_envelope",
            "automatic_physics_envelope",
        }
        and not all(strict)
    ):
        thermal_rank, rank_report = automatic_thermal_rank(
            path, config, monitor=monitor
        )
        training_payload = resolve_training_bounds(
            training_payload,
            thermal_rank,
            truncation,
            rank_report=rank_report,
        )
        thermal_kwargs = {}
    else:
        thermal_kwargs = strict_truncation_kwargs(truncation)

    nu = np.zeros(mesh.n_tetrahedra)
    capacity = np.zeros(mesh.n_tetrahedra)
    conductivity = np.zeros(mesh.n_tetrahedra)
    regions = []
    ambient = float(config.get("ambient_temperature", 293.15))
    covered = np.zeros(mesh.n_tetrahedra, bool)
    for tag, material in config["materials"].items():
        mask = tagged.tetra_mask(int(tag))
        covered |= mask
        sigma = float(material["electrical_conductivity"])
        alpha = float(
            material.get("resistivity_temperature_coefficient", 0.0)
        )
        law = (
            ReciprocalLinearResistivity(
                sigma,
                alpha,
                float(material.get("reference_temperature", ambient)),
            )
            if alpha
            else ConstantConductivity(sigma)
        )
        regions.append(
            ConductivityRegion(material["name"], mask, law)
        )
        nu[mask] = 1 / (
            4e-7
            * np.pi
            * float(material.get("relative_permeability", 1.0))
        )
        capacity[mask] = float(material["volumetric_heat_capacity"])
        conductivity[mask] = float(material["thermal_conductivity"])
    if not np.all(covered):
        raise ValueError(
            "material definitions must cover all tetrahedral physical tags"
        )

    core = TetrahedralElectroThermalCore.build_nonlinear(
        mesh,
        omega=2 * np.pi * float(config["frequency_hz"]),
        reluctivity_tetra=nu,
        conductivity_regions=regions,
        temperature_reference_nodal=np.full(mesh.n_nodes, ambient),
        constitutive_relative_error_budget=float(
            config.get("constitutive_relative_error", 1e-8)
        ),
        rho_cp_tetra=capacity,
        thermal_conductivity_tetra=conductivity,
        source_current=np.zeros(mesh.n_edges),
        thermal_rank=thermal_rank,
        **thermal_kwargs,
    )
    ports = SolidTerminalPortSet.build(
        tagged,
        core.electromagnetic_problem,
        config["terminal_pairs"],
        names=config.get("port_names"),
    )
    if thermal_rank is None:
        training_payload = resolve_training_bounds(
            training_payload,
            core.thermal_model.rank,
            truncation,
            rank_report=rank_report,
        )
    for key in (
        "initial_lower",
        "initial_upper",
        "operating_lower",
        "operating_upper",
    ):
        training_payload[key] = tuple(training_payload[key])
    training = ResearchTrainingConfig(**training_payload)
    n = core.thermal_model.rank
    if len(training.initial_lower) != n:
        raise ValueError(
            f"training restart-state bounds must contain {n} retained thermal coordinates"
        )
    candidates = config.get("em_candidate_states")
    if candidates is None:
        candidates = [
            training.initial_lower,
            training.initial_upper,
            (
                (
                    np.asarray(training.initial_lower)
                    + np.asarray(training.initial_upper)
                )
                / 2
            ).tolist(),
        ]
    result = ResearchElectroThermalModel.build(
        core,
        ports,
        candidate_states=np.asarray(candidates, float),
        requested_em_error=float(config["em_energy_error"]),
        current_offset=config.get("current_offset"),
        current_matrix=config.get("current_matrix"),
    )
    result.thermal_rank_report = rank_report or {
        "method": "strict_or_explicit",
        "selected_rank": int(core.thermal_model.rank),
        "certified_continuous_domain": bool(
            core.thermal_tail_certificate is not None
        ),
    }
    return result, training
