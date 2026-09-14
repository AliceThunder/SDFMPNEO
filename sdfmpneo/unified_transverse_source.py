"""Transverse projection for open stranded port-current sources.

The analytic spiral centerlines are open curves.  Their raw impressed-current
edge deposition therefore contains a discrete gradient (longitudinal) part at
the two terminals.  In the low-frequency E formulation that part excites the
near-null gradient subspace inside insulating packages and can dominate the
port reaction with a mesh-dependent electrostatic terminal-charging mode.

For the UWPT magnetic model, terminal/circuit electrostatics are external to
the field truth.  We therefore keep the transverse current that drives curl/H
and remove only the algebraic gradient component.  The projection is the
orthogonal Helmholtz projection in the edge/source reaction pairing, so it
preserves ``curl(source)`` to roundoff and makes the reaction gauge-invariant.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


_PROJECTION_ATOL = 1e-11
_MAX_LONGITUDINAL_RELATIVE_RESIDUAL = 1e-8
_MAX_CURL_RELATIVE_ERROR = 1e-10
TERMINAL_MODEL = "transverse_impressed_port_external_terminal_circuit_v1"


def _node_id(background, i, j, k):
    return (int(i) * (background.ny + 1) + int(j)) * (background.nz + 1) + int(k)


def _gradient_operator(background):
    cached = getattr(background, "_transverse_source_gradient", None)
    if cached is not None:
        return cached
    n_nodes = (background.nx + 1) * (background.ny + 1) * (background.nz + 1)
    rows, cols, data = [], [], []
    for edge, (axis, i, j, k) in enumerate(background.edge_tuples):
        start = _node_id(background, i, j, k)
        if axis == 0:
            stop = _node_id(background, i + 1, j, k)
        elif axis == 1:
            stop = _node_id(background, i, j + 1, k)
        else:
            stop = _node_id(background, i, j, k + 1)
        rows.extend((edge, edge))
        cols.extend((start, stop))
        data.extend((-1.0, 1.0))
    gradient = sp.csr_matrix(
        (data, (rows, cols)), shape=(background.n_edges, n_nodes), dtype=float
    )
    background._transverse_source_gradient = gradient
    return gradient


def project_transverse(background, raw_source):
    """Return the edge source orthogonal to all discrete gradients plus an audit."""
    raw = np.asarray(raw_source, float).reshape(-1)
    if raw.shape != (background.n_edges,) or np.any(~np.isfinite(raw)):
        raise ValueError("transverse source projection requires one finite value per EM edge")
    raw_norm = max(float(np.linalg.norm(raw)), np.finfo(float).tiny)
    gradient = _gradient_operator(background)
    raw_normal = np.asarray(gradient.T @ raw, float).reshape(-1)
    raw_longitudinal = float(np.linalg.norm(raw_normal) / raw_norm)
    if np.linalg.norm(raw_normal) <= 1e-14 * raw_norm:
        return raw.copy(), {
            "projection_model": "discrete_edge_helmholtz_transverse_v1",
            "raw_longitudinal_relative_norm": raw_longitudinal,
            "transverse_longitudinal_relative_norm": raw_longitudinal,
            "curl_preservation_relative_error": 0.0,
            "projection_iterations": 0,
        }

    solve = spla.lsmr(
        gradient,
        raw,
        atol=_PROJECTION_ATOL,
        btol=_PROJECTION_ATOL,
        conlim=1e12,
        maxiter=max(500, min(5000, 12 * max(background.nx, background.ny, background.nz))),
    )
    longitudinal = np.asarray(gradient @ solve[0], float).reshape(-1)
    transverse = raw - longitudinal
    remaining = np.asarray(gradient.T @ transverse, float).reshape(-1)
    remaining_relative = float(np.linalg.norm(remaining) / raw_norm)

    curl_raw = np.asarray(background.curl @ raw, float).reshape(-1)
    curl_transverse = np.asarray(background.curl @ transverse, float).reshape(-1)
    curl_error = float(
        np.linalg.norm(curl_transverse - curl_raw)
        / max(float(np.linalg.norm(curl_raw)), raw_norm, np.finfo(float).tiny)
    )
    if remaining_relative > _MAX_LONGITUDINAL_RELATIVE_RESIDUAL:
        raise RuntimeError(
            "transverse impressed-current projection did not converge: "
            f"relative longitudinal residual={remaining_relative:.3e}"
        )
    if curl_error > _MAX_CURL_RELATIVE_ERROR:
        raise RuntimeError(
            "transverse impressed-current projection changed source curl: "
            f"relative error={curl_error:.3e}"
        )
    if np.linalg.norm(transverse) <= np.finfo(float).tiny:
        raise RuntimeError("transverse impressed-current projection removed the entire source")
    return transverse, {
        "projection_model": "discrete_edge_helmholtz_transverse_v1",
        "raw_longitudinal_relative_norm": raw_longitudinal,
        "transverse_longitudinal_relative_norm": remaining_relative,
        "curl_preservation_relative_error": curl_error,
        "projection_iterations": int(solve[2]),
    }


def install(open_boundary_class):
    """Install the source projection on ``OpenBoundaryBackground`` exactly once."""
    if bool(getattr(open_boundary_class, "_transverse_source_projection_installed", False)):
        return open_boundary_class
    original = open_boundary_class._spatial_context
    open_boundary_class.terminal_model = TERMINAL_MODEL

    def spatial_context_with_transverse_source(self, geometry):
        context = original(self, geometry)
        raw = np.asarray(context.source_shape, float)
        projected, audits = [], []
        source_rows = list(getattr(context, "source_regularization", ()))
        if raw.ndim != 2 or raw.shape[0] != self.n_edges:
            raise ValueError("source_shape must have shape (n_edges, n_ports)")
        if source_rows and len(source_rows) != raw.shape[1]:
            raise ValueError("source regularization metadata does not match port count")
        for port in range(raw.shape[1]):
            source, audit = project_transverse(self, raw[:, port])
            projected.append(source)
            audits.append(audit)
            if source_rows:
                row = dict(source_rows[port])
                row["raw_source_norm"] = float(row.get("source_norm", np.linalg.norm(raw[:, port])))
                row["source_norm"] = float(np.linalg.norm(source))
                row.update(audit)
                source_rows[port] = row
        context.source_shape = np.column_stack(projected)
        if source_rows:
            context.source_regularization = tuple(source_rows)
        context.transverse_source_projection = tuple(audits)
        return context

    open_boundary_class._spatial_context = spatial_context_with_transverse_source
    open_boundary_class._transverse_source_projection_installed = True
    open_boundary_class.transverse_source_model = "discrete_edge_helmholtz_transverse_v1"
    return open_boundary_class


__all__ = ["TERMINAL_MODEL", "install", "project_transverse"]
