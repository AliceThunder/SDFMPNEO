from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from ..config import SolverConfig


Residual = Callable[[Tensor], Tensor]


@dataclass
class EquilibriumResult:
    state: Tensor
    residual_norm: Tensor
    converged: bool
    steps: int


def _jacobian(residual: Residual, z: Tensor, create_graph: bool = False) -> Tensor:
    return torch.autograd.functional.jacobian(
        residual, z, create_graph=create_graph, vectorize=True
    )


def solve_pseudo_transient_newton(
    residual: Residual,
    initial_state: Tensor,
    config: SolverConfig,
    pseudo_mass: Tensor | None = None,
) -> EquilibriumResult:
    """Solve F(z)=0 with pseudo-transient damped Newton.

    This implementation intentionally solves one reduced state at a time. The
    training driver can batch parameter samples while solving their small
    reduced nonlinear systems independently. That avoids constructing one
    artificial cross-sample Jacobian.
    """
    if initial_state.ndim != 1:
        raise ValueError("initial_state must be one-dimensional")

    z = initial_state.detach().clone()
    n = z.numel()
    if pseudo_mass is None:
        pseudo_mass = torch.eye(n, dtype=z.dtype, device=z.device)
    if pseudo_mass.shape != (n, n):
        raise ValueError("pseudo_mass must have shape [n,n]")

    pseudo_dt = config.initial_pseudo_dt
    last_norm = torch.tensor(float("inf"), dtype=z.dtype, device=z.device)

    for step in range(1, config.max_newton_steps + 1):
        z = z.detach().requires_grad_(True)
        f = residual(z)
        norm = torch.linalg.vector_norm(f)
        if not torch.isfinite(norm):
            return EquilibriumResult(z.detach(), norm.detach(), False, step)
        if norm.item() <= config.residual_tolerance:
            return EquilibriumResult(z.detach(), norm.detach(), True, step - 1)

        jac = _jacobian(residual, z, create_graph=False).detach()
        lhs = jac + pseudo_mass / pseudo_dt
        try:
            delta = torch.linalg.solve(lhs, -f.detach())
        except RuntimeError:
            pseudo_dt = max(pseudo_dt * config.line_search_shrink, 1.0e-12)
            continue

        alpha = 1.0
        accepted = False
        current = norm.detach()
        while alpha >= config.minimum_step:
            candidate = z.detach() + alpha * delta
            candidate_norm = torch.linalg.vector_norm(residual(candidate)).detach()
            if torch.isfinite(candidate_norm) and candidate_norm <= (
                1.0 - config.armijo * alpha
            ) * current:
                z = candidate
                last_norm = candidate_norm
                accepted = True
                break
            alpha *= config.line_search_shrink

        if accepted:
            pseudo_dt = min(
                pseudo_dt * config.pseudo_dt_growth, config.max_pseudo_dt
            )
        else:
            pseudo_dt = max(pseudo_dt * config.line_search_shrink, 1.0e-12)
            z = z.detach()
            last_norm = current

    return EquilibriumResult(
        state=z.detach(),
        residual_norm=last_norm.detach(),
        converged=False,
        steps=config.max_newton_steps,
    )


def implicit_adjoint(
    residual: Residual,
    equilibrium: Tensor,
    output_gradient: Tensor,
) -> Tensor:
    """Solve J_F(z*)^H lambda = dQ/dz for implicit differentiation/certification."""
    z = equilibrium.detach().requires_grad_(True)
    jac = _jacobian(residual, z, create_graph=False).detach()
    return torch.linalg.solve(jac.conj().transpose(-2, -1), output_gradient)
