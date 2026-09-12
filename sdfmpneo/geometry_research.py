"""One finite-horizon analytic surrogate over a certified UWPT geometry family."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json

import numpy as np
from scipy.stats import qmc

from .em import NonlinearTetrahedralApsiProblem, SolidTerminalPortSet, SparseEnergyReducedEMModel
from .em.sparse_reduced import SparseEnergyResidualGreedyEMReducer
from .em.riesz_action import SparseLUReferenceRieszAction
from .em.tetra_nonlinear_diagnostics import NonlinearTetrahedralRegionLossEvaluator
from .model import _port_outputs
from .rollout import rollout_fixed_network, solve_physical_steady_state
from .spatial import TaggedTetrahedralMesh
from .spatial.geometry_chart import AffineTetrahedralGeometryChart
from .training import AffineOperatingRHSMap
from .training.research import train_research_network
from .research import ResearchElectroThermalModel

_GEOMETRY_FORMAT_VERSION = 5


class GeometryResearchModel:
    """Shared thermal coordinates, EM basis and finite-horizon network across geometry."""

    def __init__(self, reference, tagged, chart, geometry_reference, lower, upper,
                 capacity, conductivity, terminal_pairs, current_offset, current_matrix,
                 *, cache_size=128):
        self.reference, self.tagged, self.chart = reference, tagged, chart
        self.geometry_reference = np.asarray(geometry_reference, float)
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        n = chart.n_parameters
        if self.lower.shape != (n,) or self.upper.shape != (n,) or np.any(~np.isfinite(self.lower + self.upper)) or np.any(self.upper <= self.lower):
            raise ValueError("geometry bounds must be finite, strictly ordered and match parameter names")
        self.certificate = chart.certify_box(self.lower - self.geometry_reference, self.upper - self.geometry_reference)
        if not self.certificate.certified_nondegenerate:
            raise ValueError("geometry box cannot be proved non-inverting on this mesh")
        self.capacity, self.conductivity = np.asarray(capacity), np.asarray(conductivity)
        self.terminal_pairs = tuple(tuple(pair) for pair in terminal_pairs)
        self.current_offset = np.asarray(current_offset, complex)
        self.current_matrix = np.asarray(current_matrix, complex)
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        self._cache = OrderedDict()
        self.thermal_model = reference.core.thermal_model
        self.n_operating = n + self.current_matrix.shape[1]
        self.rhs_map = None
        self.network = reference.network
        self.training_config = reference.training_config
        self.training_report = reference.training_report
        self.em_basis_report = None

    @property
    def geometry_names(self):
        return self.chart.parameter_names

    def geometry_vector(self, geometry):
        if isinstance(geometry, dict):
            if set(geometry) != set(self.geometry_names):
                raise ValueError(f"geometry must specify exactly {self.geometry_names}")
            geometry = [geometry[name] for name in self.geometry_names]
        g = np.asarray(geometry, float)
        if g.shape != self.lower.shape or np.any(~np.isfinite(g)):
            raise ValueError("invalid geometry input")
        if np.any(g < self.lower) or np.any(g > self.upper):
            raise ValueError("geometry is outside the saved non-inverting training chart")
        return g

    def normalize(self, g):
        return (2 * np.asarray(g) - self.lower - self.upper) / (self.upper - self.lower)

    def denormalize(self, z):
        z = np.asarray(z)
        g = (self.lower + self.upper + z * (self.upper - self.lower)) / 2
        return np.where(z == -1, self.lower, np.where(z == 1, self.upper, g))

    def context(self, geometry):
        g = self.geometry_vector(geometry)
        key = tuple(float(v) for v in g)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        mesh = self.chart.mesh(g - self.geometry_reference)
        assembly = mesh.assemble_p1_thermal(
            rho_cp_tetra=self.capacity,
            conductivity_tetra=self.conductivity,
            homogeneous_dirichlet_boundary=True,
        )
        ref, phi = self.reference, self.thermal_model.Phi
        if not np.array_equal(assembly.free_nodes, ref.core.thermal_assembly.free_nodes):
            raise RuntimeError("geometry changed the reference thermal coordinate topology")
        Mr, Kr = phi.T @ (assembly.M @ phi), phi.T @ (assembly.K @ phi)
        np.linalg.cholesky(Mr)
        original = ref.em.problem
        problem = NonlinearTetrahedralApsiProblem(
            mesh,
            omega=original.omega,
            reluctivity_tetra=original.reluctivity_tetra,
            source_current=original.source_current,
            temperature_reference_local=original.temperature_reference_local,
            thermal_modes_local=ref.core.thermal_mode_local_values,
            conductivity_regions=original.conductivity_regions,
            constitutive_relative_error_budget=original.constitutive_relative_error_budget,
        )
        tagged = TaggedTetrahedralMesh(
            mesh,
            self.tagged.tetra_physical_tags,
            self.tagged.boundary_triangles,
            self.tagged.boundary_physical_tags,
        )
        ports = SolidTerminalPortSet.build(tagged, problem, self.terminal_pairs, names=ref.ports.names)
        rhs = AffineOperatingRHSMap(
            ports.coordinate_rhs @ self.current_offset,
            ports.coordinate_rhs @ self.current_matrix,
        )
        em = SparseEnergyReducedEMModel(
            problem,
            ref.em.V,
            reference_energy_metric=ref.em.reference_energy_metric,
            riesz_action_factory=SparseLUReferenceRieszAction,
        )
        result = SimpleNamespace(mesh=mesh, assembly=assembly, M=Mr, K=Kr, em=em, ports=ports, rhs=rhs)
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def split(self, static):
        value = np.asarray(static, float)
        if value.shape != (self.n_operating,):
            raise ValueError("static input must concatenate normalized geometry and currents")
        n = len(self.geometry_names)
        return self.context(self.denormalize(value[:n])), value[n:]

    def prepare_training_contexts(self, points):
        points = np.asarray(points, float)
        n_modes = self.thermal_model.rank
        n_geometry = len(self.geometry_names)
        seen = set()
        for point in points:
            z = tuple(float(v) for v in point[n_modes:n_modes + n_geometry])
            if z not in seen:
                seen.add(z)
                self.context(self.denormalize(np.asarray(z)))

    def vector_field(self, a, static):
        context, u = self.split(static)
        q = context.em.heat_source_for_rhs(a, context.rhs.evaluate(u))
        return np.linalg.solve(context.M, -context.K @ a + q)

    def evaluate(self, a, static):
        context, u = self.split(static)
        q, Jq = context.em.heat_source_and_jacobian_for_rhs(a, context.rhs.evaluate(u))
        return SimpleNamespace(
            vector_field=np.linalg.solve(context.M, -context.K @ a + q),
            vector_field_jacobian=np.linalg.solve(context.M, -context.K + Jq),
        )

    def build_joint_em_basis(self, states, *, requested_error, anchor_count=4, monitor=None):
        """Build one shared physical-energy residual-greedy EM space over geometry anchors."""
        if not np.isfinite(requested_error) or requested_error <= 0:
            raise ValueError("EM error must be finite and positive")
        n = len(self.geometry_names)
        anchors = np.vstack([
            np.zeros(n), -np.ones(n), np.ones(n), np.eye(n), -np.eye(n),
            2 * qmc.Halton(n, scramble=True, seed=17).random(anchor_count) - 1,
        ])
        ref = self.reference
        reducer = SparseEnergyResidualGreedyEMReducer(ref.em.problem, riesz_action_factory=SparseLUReferenceRieszAction)
        V = ref.em.V.copy()
        while True:
            start_rank = V.shape[1]
            maximum = 0.0
            for z in anchors:
                if monitor is not None:
                    monitor.phase("geometry_em_basis")
                context = self.context(self.denormalize(z))
                local = SparseEnergyResidualGreedyEMReducer(context.em.problem, riesz_action_factory=SparseLUReferenceRieszAction)
                for a in states:
                    state_context = local._state_context(np.asarray(a, float))
                    for b in context.ports.coordinate_rhs.T:
                        while True:
                            if monitor is not None:
                                monitor.checkpoint()
                            residual = local._residual(state_context, b, V)
                            decision = state_context.riesz_action.decide_dual_norm(residual, threshold=requested_error / np.sqrt(2))
                            bound = np.sqrt(2) * decision.result.dual_norm_upper_bound
                            maximum = max(maximum, bound)
                            if decision.relation == "below":
                                break
                            if V.shape[1] >= context.em.problem.n_em:
                                raise RuntimeError("joint geometry EM space exhausted before residual acceptance")
                            V = reducer._append_h0_independent(decision.result.vector, V)
            if V.shape[1] == start_rank:
                break
        ref.em.V = V
        self._cache.clear()
        self.em_basis_report = {
            "anchor_count": len(anchors),
            "basis_dimension": V.shape[1],
            "maximum_anchor_energy_error": float(maximum),
            "requested_energy_error": requested_error,
            "scope": "finite geometry anchors and thermal candidate states",
        }
        return self.em_basis_report

    def training_domain(self, config):
        n = len(self.geometry_names)
        return replace(
            config,
            operating_lower=(-1.0,) * n + config.operating_lower,
            operating_upper=(1.0,) * n + config.operating_upper,
        )

    def train(self, config, *, progress=None, monitor=None):
        from .training.monitor import TrainingStopped
        if len(config.operating_lower) == self.current_matrix.shape[1]:
            config = self.training_domain(config)
        names = tuple(self.geometry_names) + tuple(
            f"current_{i}" for i in range(self.current_matrix.shape[1])
        )
        try:
            self.network, self.training_report = train_research_network(
                self,
                config,
                network=self.network,
                operating_names=names,
                progress=progress,
                monitor=monitor,
            )
        except TrainingStopped:
            if monitor is not None and monitor.best_network is not None:
                self.network = monitor.best_network
            self.training_report = None
            raise
        finally:
            self.training_config = config
        return self.training_report

    def project_initial_temperature(self, temperature_nodal, *, geometry):
        context = self.context(geometry)
        T = np.asarray(temperature_nodal, float)
        ref = self.reference.reference_temperature
        if T.shape != ref.shape or not np.all(np.isfinite(T)):
            raise ValueError("initial temperature must contain every mesh node in K")
        if not np.allclose(T[context.assembly.boundary_nodes], ref[context.assembly.boundary_nodes]):
            raise ValueError("initial boundary temperature must equal the prescribed ambient")
        phi = self.thermal_model.Phi
        return np.linalg.solve(
            context.M,
            phi.T @ (context.assembly.M @ (T - ref)[context.assembly.free_nodes]),
        )

    def _validated_inputs(self, geometry, a0, operating, *, allow_extrapolation):
        if self.network is None:
            raise ValueError("train or load a trained geometry model first")
        g = self.geometry_vector(geometry)
        initial = np.asarray(a0, float)
        u = np.asarray(operating, float)
        if initial.shape != (self.network.n_modes,) or u.shape != (self.current_matrix.shape[1],):
            raise ValueError("initial/current dimensions do not match the saved model")
        if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(u)):
            raise ValueError("initial/current inputs must be finite")
        static = np.concatenate([self.normalize(g), u])
        cfg = self.training_config
        if cfg is not None and not allow_extrapolation:
            p = np.concatenate([initial, static])
            lo = np.asarray(cfg.initial_lower + cfg.operating_lower)
            hi = np.asarray(cfg.initial_upper + cfg.operating_upper)
            if np.any(p < lo) or np.any(p > hi):
                raise ValueError("initial/geometry/current input is outside the trained restart box")
        return g, initial, u, static

    def _diagnostics(self, g, state, derivative, u, static):
        context = self.context(g)
        rhs = context.rhs.evaluate(u)
        F = self.vector_field(state, static)
        ports, certificate = _port_outputs(context.ports, context.em, state, 1e-6)
        x = context.em.state_for_rhs(state, rhs)
        return {
            "physical_residual": np.asarray(derivative - F),
            "physical_residual_norm": float(np.linalg.norm(derivative - F)),
            "impedance": ports,
            "impedance_certificate": certificate,
            "drive_rhs_residual_dual_norm": float(context.em.residual_dual_norm_for_rhs(state, rhs)),
            "region_losses": NonlinearTetrahedralRegionLossEvaluator(context.em.problem).evaluate_state(x, state),
        }

    def steady_state(self, *, geometry, a0, operating, diagnostics=True,
                     allow_extrapolation=False, tolerance=None, max_iterations=40):
        g, initial, u, static = self._validated_inputs(
            geometry, a0, operating, allow_extrapolation=allow_extrapolation
        )
        if tolerance is None:
            tolerance = self.training_config.residual_tolerance if self.training_config is not None else 1e-10
        solved = solve_physical_steady_state(
            self,
            static,
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
        T = self.reference.temperature(state)
        result = {
            "time": "steady",
            "geometry": dict(zip(self.geometry_names, g)),
            "thermal_coordinates": state,
            "thermal_derivative": derivative,
            "temperature_field": T,
            "maximum_temperature": float(np.max(T)),
            "mesh_vertices": self.chart.vertices(g - self.geometry_reference),
            "steady_state": True,
            "steady_iterations": solved.iterations,
            "steady_residual_norm": solved.residual_norm,
            "segment_count": 0,
            "segment_durations": (),
            "max_response_time": self.network.max_response_time,
        }
        if diagnostics:
            result.update(self._diagnostics(g, state, derivative, u, static))
        return result

    def predict(self, time, *, geometry, a0, operating, diagnostics=True, allow_extrapolation=False):
        g, initial, u, static = self._validated_inputs(
            geometry, a0, operating, allow_extrapolation=allow_extrapolation
        )
        t = float(time)
        if np.isnan(t) or t < 0:
            raise ValueError("time must be non-negative or positive infinity")
        if np.isposinf(t):
            return self.steady_state(
                geometry=g,
                a0=initial,
                operating=u,
                diagnostics=diagnostics,
                allow_extrapolation=allow_extrapolation,
            )
        if not np.isfinite(t):
            raise ValueError("time must be finite or positive infinity")
        cfg = self.training_config
        rolled = rollout_fixed_network(
            self.network,
            t,
            a0=initial,
            operating=static,
            state_lower=None if cfg is None else cfg.initial_lower,
            state_upper=None if cfg is None else cfg.initial_upper,
            allow_extrapolation=allow_extrapolation,
        )
        state, derivative = rolled.state, rolled.derivative
        T = self.reference.temperature(state)
        result = {
            "time": t,
            "geometry": dict(zip(self.geometry_names, g)),
            "thermal_coordinates": state,
            "thermal_derivative": derivative,
            "temperature_field": T,
            "maximum_temperature": float(np.max(T)),
            "mesh_vertices": self.chart.vertices(g - self.geometry_reference),
            "steady_state": False,
            "segment_count": rolled.segment_count,
            "segment_durations": rolled.segment_durations,
            "max_response_time": self.network.max_response_time,
        }
        if diagnostics:
            result.update(self._diagnostics(g, state, derivative, u, static))
        return result

    def save(self, path):
        if self.network is None:
            raise ValueError("no trained fixed analytic network to save")
        self.reference.network = self.network
        self.reference.training_config = self.training_config
        self.reference.training_report = self.training_report
        with TemporaryDirectory() as directory:
            ref_path = self.reference.save(Path(directory) / "reference.npz")
            reference_bytes = np.frombuffer(ref_path.read_bytes(), dtype=np.uint8)
        metadata = {
            "model_type": "geometry_research",
            "format_version": _GEOMETRY_FORMAT_VERSION,
            "names": self.geometry_names,
            "terminal_pairs": self.terminal_pairs,
            "cache_size": self.cache_size,
            "em_basis_report": self.em_basis_report,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.array(json.dumps(metadata)),
                reference_npz=reference_bytes,
                directions=self.chart.vertex_directions,
                geometry_reference=self.geometry_reference,
                lower=self.lower,
                upper=self.upper,
                capacity=self.capacity,
                conductivity=self.conductivity,
                tetra_tags=self.tagged.tetra_physical_tags,
                triangles=self.tagged.boundary_triangles,
                triangle_tags=self.tagged.boundary_physical_tags,
                current_offset=self.current_offset,
                current_matrix=self.current_matrix,
            )
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("model_type") != "geometry_research" or meta.get("format_version") != _GEOMETRY_FORMAT_VERSION:
                raise ValueError("unsupported geometry model format; retrain with the current segmented fixed network")
            with TemporaryDirectory() as directory:
                ref_path = Path(directory) / "reference.npz"
                ref_path.write_bytes(data["reference_npz"].tobytes())
                ref = ResearchElectroThermalModel.load(ref_path)
            tagged = TaggedTetrahedralMesh(ref.core.mesh, data["tetra_tags"], data["triangles"], data["triangle_tags"])
            chart = AffineTetrahedralGeometryChart(
                ref.core.mesh.vertices,
                ref.core.mesh.tetrahedra,
                data["directions"],
                tuple(meta["names"]),
            )
            out = cls(
                ref,
                tagged,
                chart,
                data["geometry_reference"],
                data["lower"],
                data["upper"],
                data["capacity"],
                data["conductivity"],
                meta["terminal_pairs"],
                data["current_offset"],
                data["current_matrix"],
                cache_size=meta["cache_size"],
            )
            out.em_basis_report = meta.get("em_basis_report")
            return out


def geometry_model_from_config(path, *, monitor=None):
    from .research import model_from_config
    from .spatial import read_gmsh_v22_ascii
    from .spatial.uwpt_family import uwpt_geometry_chart

    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    reference, training = model_from_config(path, monitor=monitor)
    family = config["geometry_family"]
    tagged = read_gmsh_v22_ascii(path.parent / config["mesh"])
    chart, g0 = uwpt_geometry_chart(
        tagged,
        family["transmitter"],
        family["receiver"],
        family["physical_tags"],
        family["parameters"],
    )
    capacity = np.zeros(tagged.mesh.n_tetrahedra)
    conductivity = capacity.copy()
    for tag, material in config["materials"].items():
        mask = tagged.tetra_mask(int(tag))
        capacity[mask] = material["volumetric_heat_capacity"]
        conductivity[mask] = material["thermal_conductivity"]
    p = reference.ports.n_ports
    offset = np.zeros(p, complex) if config.get("current_offset") is None else np.asarray(config["current_offset"], complex)
    matrix = np.eye(p, dtype=complex) if config.get("current_matrix") is None else np.asarray(config["current_matrix"], complex)
    lower, upper = [], []
    for name, nominal in zip(chart.parameter_names, g0):
        bounds = family["parameters"][name]
        lo, hi = nominal * np.asarray(bounds["relative"], float) if "relative" in bounds else bounds["bounds"]
        lower.append(lo)
        upper.append(hi)
    out = GeometryResearchModel(
        reference,
        tagged,
        chart,
        g0,
        lower,
        upper,
        capacity,
        conductivity,
        config["terminal_pairs"],
        offset,
        matrix,
        cache_size=family.get("cache_size", 128),
    )
    states = config.get("em_candidate_states")
    if states is None:
        states = [
            training.initial_lower,
            training.initial_upper,
            ((np.asarray(training.initial_lower) + np.asarray(training.initial_upper)) / 2).tolist(),
        ]
    out.build_joint_em_basis(
        states,
        requested_error=float(config["em_energy_error"]),
        anchor_count=family.get("em_anchor_count", 4),
        monitor=monitor,
    )
    return out, training
