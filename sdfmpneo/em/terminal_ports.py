from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from sdfmpneo.spatial import TaggedTetrahedralMesh

from .ports import ImpressedCurrentPortSet


def _active_components(problem) -> tuple[list[list[int]], np.ndarray]:
    mesh = problem.mesh
    support = np.zeros(mesh.n_tetrahedra, dtype=bool)
    T0 = np.asarray(problem.temperature_reference_local, dtype=float)
    for region in problem.conductivity_regions:
        mask = np.asarray(region.mask, dtype=bool)
        if not np.any(mask):
            continue
        values = np.asarray(region.law.evaluate(T0[mask]), dtype=float)
        dynamic = np.any(np.asarray(region.law.derivative(T0[mask])) != 0.0, axis=1)
        nonzero = np.any(values > 0.0, axis=1)
        idx = np.flatnonzero(mask)
        support[idx[nonzero | dynamic]] = True

    active_nodes = sorted(set(map(int, mesh.tetrahedra[support].ravel())))
    adjacency = {node: set() for node in active_nodes}
    for tet in mesh.tetrahedra[support]:
        for u, v in combinations(map(int, tet), 2):
            adjacency[u].add(v)
            adjacency[v].add(u)

    components: list[list[int]] = []
    seen: set[int] = set()
    kept: list[int] = []
    for root in active_nodes:
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in sorted(adjacency[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        component.sort()
        components.append(component)
        kept.extend(component[1:])  # same reference-node rule as mesh.conductive_gradient
    return components, np.asarray(kept, dtype=int)


def _surface_nodal_weights(tagged: TaggedTetrahedralMesh, physical_tag: int) -> np.ndarray:
    triangles = tagged.boundary_triangles[tagged.boundary_mask(physical_tag)]
    if triangles.size == 0:
        raise ValueError(f"terminal physical tag {physical_tag} has no boundary triangles")
    weights = np.zeros(tagged.mesh.n_nodes, dtype=float)
    xyz = tagged.mesh.vertices
    for tri in triangles:
        a, b, c = xyz[tri]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        if area <= 0.0:
            raise ValueError("terminal contains a degenerate boundary triangle")
        weights[tri] += area / 3.0
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("terminal surface has zero area")
    return weights / total


@dataclass(frozen=True)
class SolidTerminalPortSet(ImpressedCurrentPortSet):
    """Work-conjugate solid-conductor terminal-current ports.

    A unit port current is imposed weakly through the scalar continuity equation.
    The positive and negative terminal surfaces receive opposite P1-consistent
    area loads, so each port has exactly zero net current.  In the scaled
    ``psi=phi/(j omega)`` coordinate the inherited output ``j omega B^T x`` is
    exactly the work-conjugate terminal-voltage functional.
    """

    terminal_pairs: tuple[tuple[int, int], ...] = ()

    @classmethod
    def build(
        cls,
        tagged: TaggedTetrahedralMesh,
        problem,
        terminal_pairs: Sequence[tuple[int, int]],
        *,
        names: Sequence[str] | None = None,
    ) -> "SolidTerminalPortSet":
        pairs = tuple((int(p), int(n)) for p, n in terminal_pairs)
        if not pairs:
            raise ValueError("at least one terminal pair is required")
        port_names = tuple(f"terminal_{k}" for k in range(len(pairs))) if names is None else tuple(names)
        if len(port_names) != len(pairs) or len(set(port_names)) != len(port_names):
            raise ValueError("port names must be unique and match terminal_pairs")

        components, kept = _active_components(problem)
        if kept.size != problem.n_scalar:
            raise RuntimeError("terminal scalar coordinates do not match the electromagnetic problem")
        component_of = {}
        for cidx, component in enumerate(components):
            for node in component:
                component_of[node] = cidx

        B = np.zeros((problem.n_em, len(pairs)), dtype=complex)
        for k, (positive_tag, negative_tag) in enumerate(pairs):
            wp = _surface_nodal_weights(tagged, positive_tag)
            wn = _surface_nodal_weights(tagged, negative_tag)
            pnodes = np.flatnonzero(wp)
            nnodes = np.flatnonzero(wn)
            if any(int(node) not in component_of for node in np.concatenate([pnodes, nnodes])):
                raise ValueError("terminal surface contains nodes outside the conducting domain")
            components_used = {component_of[int(node)] for node in np.concatenate([pnodes, nnodes])}
            if len(components_used) != 1:
                raise ValueError("positive and negative terminals must belong to one conducting component")
            load = wp - wn
            if abs(float(np.sum(load))) > 64.0 * np.finfo(float).eps:
                raise FloatingPointError("terminal current load lost exact zero-net-current balance")
            B[problem.n_A :, k] = load[kept]

        # edge_currents is unused by the inherited algebra; retain its shape only
        # so the existing n_ports property and multiport API remain unchanged.
        placeholder = np.zeros((problem.mesh.n_edges, len(pairs)), dtype=float)
        return cls(port_names, placeholder, B, float(problem.omega), pairs)
