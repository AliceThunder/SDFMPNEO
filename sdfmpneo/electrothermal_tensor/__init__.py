"""Structure-preserving electrothermal tensor surrogates.

The package starts with the exact current-quadratic Joule representation.  Neural
compression and time integration are intentionally layered on top of this
physics API rather than mixed into it.
"""

from .quadratic_joule import (
    augmented_operating_vector,
    quadratic_heat_source,
    quadratic_heat_source_batch,
    quadratic_joule_tensor,
)

__all__ = [
    "augmented_operating_vector",
    "quadratic_heat_source",
    "quadratic_heat_source_batch",
    "quadratic_joule_tensor",
]
