"""Physics-aware initialization of analytic source-factor banks."""
from __future__ import annotations

import numpy as np


class _PhysicalFeatureInitMixin:
    def __init__(self, *args, **kwargs):
        supplied_parameters = kwargs.get("parameters") is not None
        super().__init__(*args, **kwargs)
        if supplied_parameters:
            return
        current_operating = [
            i for i, name in enumerate(self.operating_names)
            if str(name).startswith("current_")
        ]
        if not current_operating or self.quadratic_rank < 1:
            return

        theta = self.parameters.copy()
        qu_ids = self._indices("quadratic_u")
        qv_ids = self._indices("quadratic_v")
        feature_pairs = []
        for idx in current_operating:
            feature_pairs.append((idx, idx))
        for i, left in enumerate(current_operating):
            for right in current_operating[i + 1:]:
                feature_pairs.append((left, right))
        for row, (left, right) in enumerate(feature_pairs[:self.quadratic_rank]):
            theta[qu_ids[row]] = 0.0
            theta[qv_ids[row]] = 0.0
            theta[qu_ids[row, self.n_modes + left]] = 1.0
            theta[qv_ids[row, 1 + right]] = 1.0
        self.parameters = theta


__all__ = ["_PhysicalFeatureInitMixin"]
