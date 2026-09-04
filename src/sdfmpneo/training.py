from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .config import RankLevel
from .contracts import BasisBundle, GeometryEncoding
from .model import SDFMPNEO
from .physics.em import solve_em_matrix_free
from .reduction import build_physical_bases, truncate_bases
from .seeds import validate_seed_prefix_ranks
from .solvers.equilibrium import EquilibriumResult, solve_pseudo_transient_newton
from .topology import validate_topology


@dataclass
class TrainingStepResult:
    loss: float
    training_residual_norm: float
    equilibrium: EquilibriumResult
    rank: RankLevel


def _rank_tuple(level: RankLevel) -> tuple[int, int, int, int, int]:
    return level.em, level.thermal, level.velocity, level.tke, level.omega


def _build_rank_bases(
    model: SDFMPNEO,
    geometry: GeometryEncoding,
    level: RankLevel,
) -> BasisBundle:
    topology = model.backend.topology(geometry)
    validate_topology(topology)
    raw = model.basis_generator(geometry, topology)
    metrics = model.backend.basis_metrics(geometry, topology)
    seeds = model.backend.physics_seed_bases(geometry, topology)
    validate_seed_prefix_ranks(seeds, model.config.ranks.levels[0])
    full = build_physical_bases(
        raw,
        topology,
        metrics,
        seeds,
        jitter=model.config.metric_jitter,
        rank_rtol=model.config.basis_rank_rtol,
    )
    return truncate_bases(full, _rank_tuple(level))


def _solve_training_equilibrium(
    model: SDFMPNEO,
    geometry: GeometryEncoding,
    level: RankLevel,
) -> tuple[EquilibriumResult, BasisBundle]:
    with torch.no_grad():
        detached_bases = _build_rank_bases(model, geometry, level)

    initial = model.backend.initial_slow_state(
        geometry, detached_bases, previous_state=None
    )
    if initial.ndim != 1:
        raise ValueError("training equilibrium currently expects one sample at a time")

    def residual(z: Tensor) -> Tensor:
        assembly = model.backend.assemble_reduced(geometry, detached_bases, z)
        em = solve_em_matrix_free(detached_bases.current, assembly.em_actions)
        return assembly.slow_residual_from_em(em)

    mass = model.backend.pseudo_mass(geometry, detached_bases, initial)
    equilibrium = solve_pseudo_transient_newton(
        residual,
        initial,
        model.config.solver,
        pseudo_mass=mass,
    )
    return equilibrium, detached_bases


def _replace_none_with_zeros(
    gradients: Iterable[Tensor | None],
    parameters: list[Tensor],
) -> list[Tensor]:
    result: list[Tensor] = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        result.append(torch.zeros_like(parameter) if gradient is None else gradient)
    return result


def implicit_parameter_gradients(
    loss: Tensor,
    reduced_residual: Callable[[Tensor], Tensor],
    equilibrium_state: Tensor,
    parameters: list[Tensor],
) -> tuple[list[Tensor], Tensor]:
    """Differentiate a scalar objective through F(z*,theta)=0."""
    if loss.ndim != 0:
        raise ValueError("loss must be scalar")
    if equilibrium_state.ndim != 1:
        raise ValueError("equilibrium_state must be one-dimensional")
    if not equilibrium_state.requires_grad:
        raise ValueError("equilibrium_state must require gradients")
    if not parameters:
        raise ValueError("at least one trainable parameter is required")

    direct = torch.autograd.grad(
        loss,
        [equilibrium_state, *parameters],
        retain_graph=True,
        allow_unused=True,
    )
    grad_z = direct[0]
    if grad_z is None:
        grad_z = torch.zeros_like(equilibrium_state)
    direct_parameters = _replace_none_with_zeros(direct[1:], parameters)

    f = reduced_residual(equilibrium_state)
    jacobian = torch.autograd.functional.jacobian(
        reduced_residual,
        equilibrium_state,
        create_graph=False,
        vectorize=True,
    ).detach()
    adjoint = torch.linalg.solve(
        jacobian.conj().transpose(-2, -1),
        grad_z.detach(),
    )

    implicit_raw = torch.autograd.grad(
        f,
        parameters,
        grad_outputs=adjoint,
        allow_unused=True,
    )
    implicit_parameters = _replace_none_with_zeros(implicit_raw, parameters)
    total_gradients = [
        direct_gradient - implicit_gradient
        for direct_gradient, implicit_gradient in zip(
            direct_parameters, implicit_parameters, strict=True
        )
    ]
    return total_gradients, adjoint.detach()


def _assign_gradients(parameters: list[Tensor], gradients: list[Tensor]) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.detach()


def solution_data_free_training_step(
    model: SDFMPNEO,
    geometry: GeometryEncoding,
    level: RankLevel,
    optimizer: Optimizer,
) -> TrainingStepResult:
    """Perform one solution-data-free neural enrichment optimisation step."""
    optimizer.zero_grad(set_to_none=True)
    equilibrium, _ = _solve_training_equilibrium(model, geometry, level)
    if not equilibrium.converged:
        raise RuntimeError(
            "Reduced equilibrium did not converge; do not train on an invalid state."
        )

    bases = _build_rank_bases(model, geometry, level)
    z = equilibrium.state.detach().requires_grad_(True)

    def reduced_residual(state: Tensor) -> Tensor:
        assembly = model.backend.assemble_reduced(geometry, bases, state)
        em = solve_em_matrix_free(bases.current, assembly.em_actions)
        return assembly.slow_residual_from_em(em)

    assembly = model.backend.assemble_reduced(geometry, bases, z)
    em = solve_em_matrix_free(bases.current, assembly.em_actions)
    training_residual = model.backend.training_residual(geometry, bases, z, em)
    loss = 0.5 * training_residual.abs().square().sum()

    parameters = [
        parameter
        for parameter in model.basis_generator.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("The basis generator has no trainable parameters.")

    total_gradients, _ = implicit_parameter_gradients(
        loss,
        reduced_residual,
        z,
        parameters,
    )
    _assign_gradients(parameters, total_gradients)
    optimizer.step()

    return TrainingStepResult(
        loss=float(loss.detach().item()),
        training_residual_norm=float(
            torch.linalg.vector_norm(training_residual.detach()).item()
        ),
        equilibrium=equilibrium,
        rank=level,
    )
