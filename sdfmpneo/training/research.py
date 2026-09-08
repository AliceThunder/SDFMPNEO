"""Solution-data-free fitting with explicitly numerical residual acceptance.

No trajectory solver or labelled states are used here. Collocation points are
inputs (a0, currents, time); all targets are governing-equation residuals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from types import SimpleNamespace

import numpy as np
from scipy.stats import qmc

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph, evaluate_parametric_stable
from sdfmpneo.analytic.parametric_realization import (compile_parametric_realization,
    evaluate_parametric_stable_with_jacobians)
from sdfmpneo.analytic.realization import AnalyticRealization


@dataclass(frozen=True)
class ResearchTrainingConfig:
    initial_lower: tuple[float, ...]
    initial_upper: tuple[float, ...]
    operating_lower: tuple[float, ...]
    operating_upper: tuple[float, ...]
    time_horizon: float
    residual_tolerance: float
    sample_count: int = 32
    validation_count: int = 32
    max_nodes: int = 24
    max_degree: int = 3
    time_sampling: str = "linear"
    time_min: float = 1e-6
    include_steady_state: bool = False
    max_parent_responses: int = 3
    max_realization_dimension: int = 64

    def __post_init__(self):
        for key in ('initial_lower','initial_upper','operating_lower','operating_upper'):
            object.__setattr__(self,key,tuple(float(v) for v in getattr(self,key)))
        if len(self.initial_lower)!=len(self.initial_upper) or len(self.operating_lower)!=len(self.operating_upper):
            raise ValueError("lower and upper dimensions must match")
        if not np.isfinite(self.time_horizon+self.residual_tolerance):
            raise ValueError("time horizon and residual tolerance must be finite")
        for key in ('sample_count','validation_count','max_nodes','max_degree','max_parent_responses','max_realization_dimension'):
            if int(getattr(self,key))!=getattr(self,key):
                raise ValueError("sample and work budgets must be integers")
            object.__setattr__(self,key,int(getattr(self,key)))
        lo = np.array(self.initial_lower + self.operating_lower, float)
        hi = np.array(self.initial_upper + self.operating_upper, float)
        if lo.shape != hi.shape or np.any(~np.isfinite(lo + hi)) or np.any(lo > hi):
            raise ValueError("invalid training parameter box")
        if self.time_horizon <= 0 or self.residual_tolerance <= 0:
            raise ValueError("time_horizon and residual_tolerance must be positive")
        if self.max_parent_responses < 0 or self.max_realization_dimension < 2:
            raise ValueError("invalid analytic candidate complexity budget")
        if min(self.sample_count, self.validation_count, self.max_nodes, self.max_degree) < 1:
            raise ValueError("sample and work budgets must be positive")

        if self.time_sampling not in ('linear', 'mixed_log'):
            raise ValueError('time_sampling must be linear or mixed_log')
        if not np.isfinite(self.time_min) or self.time_min <= 0 or (
                self.time_sampling == 'mixed_log' and self.time_min > self.time_horizon):
            raise ValueError('time_min must be positive and no larger than time_horizon for mixed_log')

    def points(self, validation=False, seed=None):
        # Two distinct deterministic low-discrepancy sets; these are numerical
        # checks, never continuous-domain certificates. Include t=0 explicitly.
        lo = np.array(self.initial_lower + self.operating_lower + (0.,))
        hi = np.array(self.initial_upper + self.operating_upper + (self.time_horizon,))
        count = self.validation_count if validation else self.sample_count
        engine = qmc.Halton(len(lo), scramble=True, seed=(1 if validation else 0) if seed is None else seed)
        unit = engine.random(count)
        if not validation:
            unit = np.vstack([unit, np.full(len(lo), .5), np.zeros(len(lo)), np.ones(len(lo))])
            unit[-3, -1] = 0.
        points = lo + unit * (hi-lo)
        if self.time_sampling == 'mixed_log':
            # Half linear for late transients, half logarithmic for early scales.
            ids = np.arange(0, count, 2)
            points[ids, -1] = np.exp(np.log(self.time_min) + unit[ids, -1] *
                                    (np.log(self.time_horizon)-np.log(self.time_min)))
        if self.include_steady_state:
            steady = points[:count].copy()
            steady[:, -1] = np.inf
            points = np.vstack([points, steady])
        return points


@dataclass(frozen=True)
class ResearchTrainingReport:
    status: str
    accepted_nodes: int
    initial_rms_residual: float
    final_rms_residual: float
    maximum_training_residual: float
    maximum_validation_residual: float
    objective_history: tuple[float, ...]
    # True only describes this finite numerical check, not PDE certification.
    numerical_tolerance_met: bool

    def to_dict(self):
        return asdict(self)


def _clone(graph):
    out = ParametricAnalyticEvolutionGraph(graph.lambdas.copy(), graph.operating_names)
    for n in graph.response_nodes:
        out.add_product_response(n.name, n.target_mode, n.parents, n.weight)
    return out


def _evaluate(graph, field, points, *, jacobian=False, monitor=None):
    records = []
    n = graph.n_modes
    for point in points:
        if monitor is not None:
            monitor.checkpoint()
        initial, u, time = point[:n], point[n:-1], float(point[-1])
        a, da = evaluate_parametric_stable(graph, time, a0=initial, operating=u)
        if jacobian:
            physical = field.evaluate(a, u)
            F, J = physical.vector_field, physical.vector_field_jacobian
        elif hasattr(field, "vector_field"):
            F = field.vector_field(a, u)
            J = None
        else:
            rhs = field.rhs(u)
            q = field.em_model.heat_source_for_rhs(a, rhs)
            F = -field.thermal_model.lambdas * a + q + field.thermal_forcing
            J = None
        records.append(SimpleNamespace(a=a, residual=da-F, J=J, initial=initial, u=u, time=time))
    return records


def _metrics(records):
    norms = np.array([np.linalg.norm(x.residual) for x in records])
    return float(np.mean(norms**2)), float(np.max(norms))


def _refine_weights(graph, field, points, max_iterations=12, monitor=None, tolerance=0.):
    """Joint Gauss--Newton correction using exact DAG and physical Jacobians."""
    if not graph.response_nodes:
        return graph
    for _ in range(max_iterations):
        if monitor is not None:
            monitor.phase("weight_refinement")
        residuals, jacobians = [], []
        for point in points:
            if monitor is not None:
                monitor.checkpoint()
            n = graph.n_modes
            a, da, ja, jda = evaluate_parametric_stable_with_jacobians(
                graph, float(point[-1]), a0=point[:n], operating=point[n:-1], weight_derivatives=True)
            physical = field.evaluate(a, point[n:-1])
            residuals.append(da-physical.vector_field)
            jacobians.append(jda-physical.vector_field_jacobian @ ja)
        if max(np.linalg.norm(r) for r in residuals) <= tolerance:
            break
        residual = np.concatenate(residuals)
        J = np.vstack(jacobians)
        scales = np.linalg.norm(J, axis=0)
        scales[scales == 0] = 1.
        # Column scaling changes coordinates, not the objective or physics.
        delta = np.linalg.lstsq(J/scales, -residual, rcond=None)[0]/scales
        objective = float(residual @ residual)
        accepted = False
        for _ in range(20):
            trial = ParametricAnalyticEvolutionGraph(graph.lambdas, graph.operating_names)
            for node, change in zip(graph.response_nodes, delta):
                trial.add_product_response(node.name,node.target_mode,node.parents,node.weight+change)
            try:
                values = _evaluate(trial,field,points,monitor=monitor)
                trial_objective = sum(float(r.residual @ r.residual) for r in values)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                trial_objective = float('inf')
            if trial_objective < objective:
                graph = trial
                if monitor is not None:
                    obj, maximum = _metrics(values)
                    monitor.record(graph, obj, maximum, len(points))
                accepted = True
                break
            delta *= .5
        if not accepted or objective-trial_objective <= 1e-10*max(objective,np.finfo(float).tiny):
            break
    return graph


def _quadratic_heating_seed(graph, field, monitor=None):
    """Exact reference-state Joule polynomial in the affine current inputs.

    For x(0,U)=x0+sum U_k*xk, q_j=x^H H_j x is quadratic. These response
    neurons come from the governing equation, not from solution trajectories.
    """
    if hasattr(field, "seed_graph") and not graph.response_nodes:
        return field.seed_graph(graph, monitor=monitor)
    if graph.response_nodes or field.rhs_map is None:
        return graph
    source = np.column_stack([field.rhs_map.offset,field.rhs_map.matrix])
    state = np.zeros(graph.n_modes)
    X = np.column_stack([field.em_model.state_for_rhs(state,b) for b in source.T])
    trial = _clone(graph)
    problem = field.em_model.problem
    for j in range(graph.n_modes):
        if monitor is not None:
            monitor.checkpoint()
        loss = getattr(problem,'loss_operator_sparse',None)
        H = problem.loss_operator(j,state) if loss is None else loss(j,state)
        Q = np.real(X.conj().T @ (H @ X))
        for k in range(source.shape[1]):
            for ell in range(k,source.shape[1]):
                weight = Q[k,ell] if k==ell else Q[k,ell]+Q[ell,k]
                if k==ell==0:
                    weight += field.thermal_forcing[j]
                if weight==0:
                    continue
                parents=tuple(graph.operating_names[i-1] for i in [k,ell] if i>0)
                trial.add_product_response(f'response_{len(trial.response_nodes)}',j,parents,weight)
    return trial


def train_research_graph(field, config: ResearchTrainingConfig, *, graph=None, progress=None, monitor=None):
    """Grow analytic response neurons using the exact coupled residual tangent.

    Polynomial degree and node counts are computational budgets; a run that
    exhausts them is reported as budget_exhausted, not converged. All accepted
    weights decrease the re-evaluated nonlinear residual. No labelled FEM or
    transient samples, optimizer-generated trajectories or time marching enter
    this routine.
    """
    if graph is None:
        graph = ParametricAnalyticEvolutionGraph(field.thermal_model.lambdas,
                                               [f"u{i}" for i in range(field.n_operating)])
    if len(config.initial_lower) != graph.n_modes or len(config.operating_lower) != field.n_operating:
        raise ValueError("training domain does not match the physical model")
    if len(graph.response_nodes) > config.max_nodes:
        raise ValueError("existing graph exceeds the requested node budget")
    if monitor is not None:
        monitor.retain(graph)
        monitor.phase("initial_residual")
    points = config.points()
    checks = config.points(validation=True)
    records = _evaluate(graph, field, points, jacobian=True, monitor=monitor)
    objective, maximum = _metrics(records)
    initial_rms = np.sqrt(objective)
    history = [objective]
    accepted = len(graph.response_nodes)
    if monitor is not None:
        monitor.record(graph, objective, maximum, len(points))
        monitor.phase("quadratic_seed")
    if graph.response_nodes and maximum > config.residual_tolerance:
        graph = _refine_weights(graph,field,points,monitor=monitor,tolerance=config.residual_tolerance)
        records = _evaluate(graph,field,points,jacobian=True,monitor=monitor)
        objective,maximum = _metrics(records)
        history.append(objective)
        if monitor is not None:
            monitor.record(graph,objective,maximum,len(points))
    seed = _quadratic_heating_seed(graph,field,monitor=monitor) if config.max_degree >= 2 else graph
    if len(seed.response_nodes) <= config.max_nodes and seed is not graph:
        try:
            seed_objective,seed_max = _metrics(_evaluate(seed,field,points,monitor=monitor))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError):
            seed_objective = float('inf')
        if seed_objective < objective:
            if monitor is not None:
                monitor.record(seed, seed_objective, seed_max, len(points))
            graph = _refine_weights(seed,field,points,monitor=monitor,tolerance=config.residual_tolerance)
            records = _evaluate(graph,field,points,jacobian=True,monitor=monitor)
            objective,maximum = _metrics(records)
            accepted = len(graph.response_nodes)
            history.append(objective)
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points))
            if progress is not None:
                progress(accepted,np.sqrt(objective),maximum)
    check_max = float('inf')
    while True:
        if monitor is not None:
            monitor.checkpoint()
        if maximum <= config.residual_tolerance or accepted >= config.max_nodes:
            if monitor is not None:
                monitor.phase("validation")
            _, check_max = _metrics(_evaluate(graph, field, checks, monitor=monitor))
            if monitor is not None:
                monitor.validation(check_max, len(checks))
            if check_max <= config.residual_tolerance and maximum <= config.residual_tolerance:
                status = 'numerically_converged'
                break
            if accepted >= config.max_nodes:
                status = 'budget_exhausted'
                break
            # Refine collocation at the independent violating points, then
            # reserve a fresh independent validation set for the next check.
            points = np.vstack([points, checks])
            checks = config.points(validation=True, seed=2+accepted)
            records = _evaluate(graph, field, points, jacobian=True, monitor=monitor)
            objective, maximum = _metrics(records)
            # A new sample set changes the objective, so start a new monotone
            # history on that set; do not compare unlike quadrature objectives.
            history = [objective]
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points), new_points=True)

        if monitor is not None:
            monitor.phase("candidate_search")
        compiled = [compile_parametric_realization(graph, a0=r.initial, operating=r.u) for r in records]
        names = graph.known_names()
        existing = {(n.target_mode, tuple(sorted(n.parents))) for n in graph.response_nodes}
        scored = []
        response_names = {node.name for node in graph.response_nodes}
        # Empty product includes static heating even when the RHS has an offset.
        for degree in range(config.max_degree+1):
            for parents in combinations_with_replacement(names, degree):
                if sum(p in response_names for p in parents) > config.max_parent_responses:
                    continue
                dimension = int(np.prod([compiled[0].node_realizations[p].dimension for p in parents]))+1
                if dimension > config.max_realization_dimension:
                    continue
                for target in range(graph.n_modes):
                    if (target, tuple(sorted(parents))) in existing:
                        continue
                    inner = norm2 = 0.
                    for r, realization in zip(records, compiled):
                        if monitor is not None:
                            monitor.checkpoint()
                        source = AnalyticRealization.constant(1.)
                        for parent in parents:
                            source = source.product(realization.node_realizations[parent])
                        response = source.response(graph.lambdas[target])
                        h = response.evaluate(r.time).real
                        psi = source.evaluate(r.time).real
                        e = np.zeros(graph.n_modes); e[target] = 1.
                        tangent = e*psi - (graph.lambdas[target]*e + r.J[:, target])*h
                        inner += float(r.residual @ tangent)
                        norm2 += float(tangent @ tangent)
                    if norm2 > 0 and np.isfinite(norm2 + inner):
                        scored.append((inner*inner/norm2, -inner/norm2, target, parents))
        scored.sort(key=lambda item: item[0], reverse=True)
        success = False
        # Nonlinear re-evaluation with backtracking, including the material
        # admissibility domain. Tangent score alone never accepts an update.
        for score, weight, target, parents in scored:
            if score <= 0:
                continue
            for _ in range(24):
                trial = _clone(graph)
                trial.add_product_response(f"response_{len(graph.response_nodes)}", target, parents, weight)
                try:
                    trial_records = _evaluate(trial, field, points, monitor=monitor)
                    trial_objective, trial_max = _metrics(trial_records)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    trial_objective = float('inf')
                if np.isfinite(trial_objective) and trial_objective < objective:
                    if monitor is not None:
                        monitor.record(trial, trial_objective, trial_max, len(points))
                    graph = _refine_weights(trial, field, points, monitor=monitor, tolerance=config.residual_tolerance)
                    objective, maximum = _metrics(_evaluate(graph,field,points,monitor=monitor))
                    history.append(objective)
                    accepted += 1
                    if monitor is not None:
                        monitor.record(graph, objective, maximum, len(points))
                    success = True
                    if progress is not None:
                        progress(accepted, np.sqrt(objective), maximum)
                    break
                weight *= .5
            if success:
                break
        if not success:
            status = 'stalled'
            if monitor is not None:
                monitor.phase("validation")
            _, check_max = _metrics(_evaluate(graph, field, checks, monitor=monitor))
            if monitor is not None:
                monitor.validation(check_max, len(checks))
            break
        records = _evaluate(graph, field, points, jacobian=True, monitor=monitor)
    report = ResearchTrainingReport(status, accepted, float(initial_rms), float(np.sqrt(objective)),
                                    maximum, check_max, tuple(history), status=='numerically_converged')
    return graph, report
