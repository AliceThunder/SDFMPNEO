"""Adapters from existing SDFMPNEO physical models to the new tensor pipeline."""
from __future__ import annotations

import numpy as np

from .quadratic_joule import quadratic_joule_tensor
from .vector_field import CallableThermalOperatorFamily, FixedThermalOperatorFamily


def fixed_research_tensor_factory(model):
    """Return ``factory(state, empty_geometry) -> G`` for ResearchElectroThermalModel."""
    if not hasattr(model, "em") or not hasattr(model, "rhs_map"):
        raise TypeError("model does not expose the fixed research EM/rhs interface")

    def factory(state, geometry):
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.size != 0:
            raise ValueError("fixed-geometry tensor factory expects empty geometry coordinates")
        return quadratic_joule_tensor(model.em, state, model.rhs_map)

    return factory


def fixed_research_thermal_family(model):
    thermal = model.core.thermal_model
    M, K = thermal.reduced_matrices()
    return FixedThermalOperatorFamily(M, K, geometry_dimension=0)


def geometry_research_tensor_factory(model, *, normalized_geometry: bool = True):
    """Return an exact tensor factory over a GeometryResearchModel family."""
    n = len(model.geometry_names)

    def context_from_coordinates(coordinates):
        z = np.asarray(coordinates, dtype=float).reshape(-1)
        if z.shape != (n,):
            raise ValueError("geometry coordinate dimension mismatch")
        geometry = model.denormalize(z) if normalized_geometry else z
        return model.context(geometry)

    def factory(state, geometry):
        context = context_from_coordinates(geometry)
        return quadratic_joule_tensor(context.em, state, context.rhs)

    return factory


def geometry_research_thermal_family(model, *, normalized_geometry: bool = True):
    n = len(model.geometry_names)

    def callback(coordinates):
        z = np.asarray(coordinates, dtype=float).reshape(-1)
        if z.shape != (n,):
            raise ValueError("geometry coordinate dimension mismatch")
        geometry = model.denormalize(z) if normalized_geometry else z
        return model.context(geometry)

    return CallableThermalOperatorFamily(n, callback)


__all__ = [
    "fixed_research_tensor_factory",
    "fixed_research_thermal_family",
    "geometry_research_tensor_factory",
    "geometry_research_thermal_family",
]
