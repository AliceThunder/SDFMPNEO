"""Relative tree-cotree gauge for first-order 3D Nedelec edge elements.

Fix exactly one edge per interior nodal gradient degree of freedom, rooted
at the Dirichlet outer boundary. For the simply connected cubical D2
domain, this removes the redundant H(curl) gradient gauge without a
saddle-point Lagrange multiplier. Conducting scalar phi DOFs remain.
No H(curl) high-order, harmonic-cohomology or general disconnected-domain
claims: independently revalidate topology for other geometries.
"""
from __future__ import annotations

from collections import deque


def relative_tree_edges(edge_dictionary, boundary_nodes, n_vertices):
    """Return zero-A edge DOFs of a spanning forest rooted at boundary nodes.

    All boundary vertices are treated as roots (their relative P1 gauge is
    zero), and each *interior* node is reached by a unique parent edge.
    The fixed edge set has size n_vertices - len(boundary_nodes).
    This is a gauge choice, not a material boundary, coil-source or
    electrostatic-phi constraint.
    """
    adjacency = [[] for _ in range(n_vertices)]
    for (a, b), edge in edge_dictionary.items():
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    visited = set(boundary_nodes)
    if not visited:
        raise ValueError("Relative tree gauge needs Dirichlet boundary nodes")
    queue = deque(sorted(visited))
    fixed = set()
    while queue:
        node = queue.popleft()
        for neighbor, edge in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                fixed.add(edge)
                queue.append(neighbor)
    if len(visited) != n_vertices:
        raise ValueError("Disconnected mesh component without Dirichlet boundary")
    if len(fixed) != n_vertices - len(boundary_nodes):
        raise ValueError("Incomplete relative Nedelec gauge forest")
    return fixed
