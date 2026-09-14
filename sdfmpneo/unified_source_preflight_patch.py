"""Add transverse-source invariants to the production source preflight."""
from __future__ import annotations

import numpy as np


_MAX_LONGITUDINAL_RELATIVE_RESIDUAL = 1e-8
_MAX_CURL_RELATIVE_ERROR = 1e-10


def install(truth_preflight_module):
    if bool(getattr(truth_preflight_module, "_transverse_source_preflight_installed", False)):
        return truth_preflight_module
    original = truth_preflight_module._source_and_loss_partition

    def source_and_loss_partition(background, geometry):
        result = dict(original(background, geometry))
        context = background.geometry_context(geometry, assemble_thermal=False)
        rows = tuple(getattr(context, "source_regularization", ()))
        expected_model = getattr(background, "transverse_source_model", None)
        projection_ok = bool(
            expected_model
            and len(rows) == context.source_shape.shape[1]
            and all(
                row.get("projection_model") == expected_model
                and float(row.get("transverse_longitudinal_relative_norm", np.inf))
                <= _MAX_LONGITUDINAL_RELATIVE_RESIDUAL
                and float(row.get("curl_preservation_relative_error", np.inf))
                <= _MAX_CURL_RELATIVE_ERROR
                for row in rows
            )
        )
        max_longitudinal = max(
            (float(row.get("transverse_longitudinal_relative_norm", np.inf)) for row in rows),
            default=float("inf"),
        )
        max_curl_error = max(
            (float(row.get("curl_preservation_relative_error", np.inf)) for row in rows),
            default=float("inf"),
        )
        result["transverse_source_projection"] = projection_ok
        result["maximum_transverse_longitudinal_relative_norm"] = float(max_longitudinal)
        result["maximum_source_curl_preservation_relative_error"] = float(max_curl_error)
        result["finite_support_source"] = bool(result.get("finite_support_source", False) and projection_ok)
        return result

    truth_preflight_module._source_and_loss_partition = source_and_loss_partition
    truth_preflight_module._transverse_source_preflight_installed = True
    return truth_preflight_module


__all__ = ["install"]
