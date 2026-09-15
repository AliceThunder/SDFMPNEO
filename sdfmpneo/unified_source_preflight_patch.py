"""Production source preflight for an open two-terminal stranded current.

The source must remain open and carry genuine terminal divergence.  Both pieces
of finite-support regularization are physical and must be independently
certified:

* feed/return divergence is distributed over a mesh-independent contact length;
* the stranded current occupies a mesh-independent rectangular cross section.

The numerical quadrature used to integrate that fixed rectangle may become more
resolved on finer meshes; changing quadrature resolution is not a change of the
physical source.  The continuity audit compares the nodal-divergence first
moment with the mesh-independent regularized source vector declared by source
deposition.
"""
from __future__ import annotations

import numpy as np

from .unified_gradient_block_maxwell import source_terminal_divergence


_MAX_NET_TERMINAL_BALANCE_ERROR = 1e-12
_MAX_TERMINAL_MOMENT_RELATIVE_ERROR = 1e-12
_MIN_TERMINAL_DIVERGENCE_RELATIVE_NORM = 1e-12


def install(truth_preflight_module):
    if bool(getattr(truth_preflight_module, "_open_terminal_source_preflight_installed", False)):
        return truth_preflight_module
    original = truth_preflight_module._source_and_loss_partition

    def source_and_loss_partition(background, geometry):
        result = dict(original(background, geometry))
        context = background.geometry_context(geometry, assemble_thermal=False)
        source = np.asarray(context.source_shape, float)
        if source.ndim != 2 or source.shape[0] != background.n_edges:
            raise ValueError("production source matrix has invalid edge/port shape")

        metadata = tuple(getattr(context, "source_regularization", ()))
        rows = []
        for port in range(source.shape[1]):
            q, net_error, moment = source_terminal_divergence(background, source[:, port])
            coil = context.geometry.coils[port]
            points = np.asarray(background._physical_centerline(coil), float)
            path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            meta = metadata[port] if port < len(metadata) else {}
            declared = meta.get("regularized_source_vector")
            if declared is None:
                expected_moment = np.asarray(points[-1] - points[0], float)
            else:
                expected_moment = np.asarray(declared, float).reshape(-1)
                if expected_moment.shape != (3,) or np.any(~np.isfinite(expected_moment)):
                    raise ValueError("terminal regularization declared an invalid source vector")

            moment_error = float(
                np.linalg.norm(moment - expected_moment)
                / max(path_length, np.finfo(float).tiny)
            )
            divergence_relative = float(
                np.linalg.norm(q)
                / max(float(np.linalg.norm(source[:, port])), np.finfo(float).tiny)
            )
            distributed = bool(
                meta.get("terminal_regularization_mesh_independent", False)
                and float(meta.get("terminal_contact_length", 0.0)) > 0.0
                and bool(meta.get("terminal_profile"))
            )
            requires_distributed = str(getattr(background, "terminal_model", "")).startswith(
                "distributed_terminal_contact"
            )
            terminal_support_ok = bool(distributed if requires_distributed else True)

            composite_source = str(getattr(background, "source_model", "")).startswith(
                "stranded_rectangular_cross_section_composite_gauss3"
            )
            cross_section_support_ok = bool(
                meta.get("cross_section_support_mesh_independent", False)
                and meta.get("cross_section_quadrature") == "composite_gauss3"
                and int(meta.get("cross_section_width_panels", 0)) >= 1
                and int(meta.get("cross_section_thickness_panels", 0)) >= 1
                and int(meta.get("cross_section_quadrature_points", 0)) >= 9
                and float(meta.get("source_quadrature_resolution", 0.0)) > 0.0
            )
            if not composite_source:
                cross_section_support_ok = True

            rows.append(
                {
                    "port": int(port),
                    "net_terminal_balance_relative_error": float(net_error),
                    "terminal_divergence_relative_norm": divergence_relative,
                    "terminal_first_moment_relative_error": moment_error,
                    "terminal_divergence_nonzero": bool(
                        divergence_relative >= _MIN_TERMINAL_DIVERGENCE_RELATIVE_NORM
                    ),
                    "terminal_support_mesh_independent": terminal_support_ok,
                    "cross_section_support_mesh_independent": cross_section_support_ok,
                    "terminal_contact_length": float(meta.get("terminal_contact_length", 0.0)),
                    "terminal_profile": meta.get("terminal_profile"),
                    "cross_section_quadrature": meta.get("cross_section_quadrature"),
                    "cross_section_width_panels": int(meta.get("cross_section_width_panels", 0)),
                    "cross_section_thickness_panels": int(meta.get("cross_section_thickness_panels", 0)),
                    "cross_section_quadrature_points": int(meta.get("cross_section_quadrature_points", 0)),
                    "source_quadrature_resolution": float(meta.get("source_quadrature_resolution", 0.0)),
                }
            )

        terminal_ok = bool(
            rows
            and all(
                row["net_terminal_balance_relative_error"] <= _MAX_NET_TERMINAL_BALANCE_ERROR
                and row["terminal_first_moment_relative_error"] <= _MAX_TERMINAL_MOMENT_RELATIVE_ERROR
                and row["terminal_divergence_nonzero"]
                and row["terminal_support_mesh_independent"]
                and row["cross_section_support_mesh_independent"]
                for row in rows
            )
        )
        result["open_two_terminal_charge_balance"] = terminal_ok
        result["terminal_support_mesh_independent"] = bool(
            rows and all(row["terminal_support_mesh_independent"] for row in rows)
        )
        result["cross_section_support_mesh_independent"] = bool(
            rows and all(row["cross_section_support_mesh_independent"] for row in rows)
        )
        result["minimum_terminal_contact_length"] = min(
            (row["terminal_contact_length"] for row in rows), default=0.0
        )
        result["minimum_cross_section_quadrature_points"] = min(
            (row["cross_section_quadrature_points"] for row in rows), default=0
        )
        result["maximum_source_quadrature_resolution"] = max(
            (row["source_quadrature_resolution"] for row in rows), default=float("inf")
        )
        result["maximum_net_terminal_balance_relative_error"] = max(
            (row["net_terminal_balance_relative_error"] for row in rows), default=float("inf")
        )
        result["minimum_terminal_divergence_relative_norm"] = min(
            (row["terminal_divergence_relative_norm"] for row in rows), default=0.0
        )
        result["maximum_terminal_first_moment_relative_error"] = max(
            (row["terminal_first_moment_relative_error"] for row in rows), default=float("inf")
        )
        result["terminal_divergence_audit"] = rows
        result["terminal_path_conservation"] = bool(
            result.get("terminal_path_conservation", False) and terminal_ok
        )
        result["finite_support_source"] = bool(
            result.get("finite_support_source", False)
            and result["cross_section_support_mesh_independent"]
        )
        return result

    truth_preflight_module._source_and_loss_partition = source_and_loss_partition
    truth_preflight_module._open_terminal_source_preflight_installed = True
    return truth_preflight_module


__all__ = ["install"]
