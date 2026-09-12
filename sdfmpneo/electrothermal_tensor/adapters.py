"""Adapters from existing SDFMPNEO physical models to the new tensor pipeline."""
from __future__ import annotations

import numpy as np

from .quadratic_joule import quadratic_joule_tensor
from .vector_field import CallableThermalOperatorFamily, FixedThermalOperatorFamily


def fixed_research_tensor_factory(model):
    if not hasattr(model, "em") or not hasattr(model, "rhs_map"):
        raise TypeError("model does not expose the fixed research EM/rhs interface")

    def factory(state, geometry):
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.size != 0:
            raise ValueError("fixed-geometry tensor factory expects empty geometry coordinates")
        return quadratic_joule_tensor(model.em, state, model.rhs_map)

    return factory


def fixed_research_direct_heat_factory(model):
    def factory(state, geometry, operating):
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.size != 0:
            raise ValueError("fixed-geometry direct heat expects empty geometry")
        rhs = model.rhs_map.evaluate(np.asarray(operating, dtype=float))
        return model.em.heat_source_for_rhs(np.asarray(state, dtype=float), rhs)

    return factory


def fixed_research_vector_field_factory(model):
    def factory(state, geometry, operating):
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.size != 0:
            raise ValueError("fixed-geometry vector field expects empty geometry")
        return model.field.evaluate(
            np.asarray(state, dtype=float), np.asarray(operating, dtype=float)
        ).vector_field

    return factory


def fixed_research_jacobian_factory(model):
    def factory(state, geometry, operating):
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.size != 0:
            raise ValueError("fixed-geometry Jacobian expects empty geometry")
        return model.field.evaluate(
            np.asarray(state, dtype=float), np.asarray(operating, dtype=float)
        ).vector_field_jacobian

    return factory


def fixed_research_temperature_reconstructor(model):
    def reconstruct(state, geometry):
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.size != 0:
            raise ValueError("fixed-geometry temperature reconstruction expects empty geometry")
        return model.temperature(np.asarray(state, dtype=float))

    return reconstruct


def fixed_research_thermal_family(model):
    thermal = model.core.thermal_model
    M, K = thermal.reduced_matrices()
    return FixedThermalOperatorFamily(M, K, geometry_dimension=0)


def _geometry_context(model, coordinates, *, normalized_geometry: bool):
    n = len(model.geometry_names)
    z = np.asarray(coordinates, dtype=float).reshape(-1)
    if z.shape != (n,):
        raise ValueError("geometry coordinate dimension mismatch")
    geometry = model.denormalize(z) if normalized_geometry else z
    return model.context(geometry), geometry


def geometry_research_tensor_factory(model, *, normalized_geometry: bool = True):
    def factory(state, geometry):
        context, _ = _geometry_context(model, geometry, normalized_geometry=normalized_geometry)
        return quadratic_joule_tensor(context.em, state, context.rhs)

    return factory


def geometry_research_direct_heat_factory(model, *, normalized_geometry: bool = True):
    def factory(state, geometry, operating):
        context, _ = _geometry_context(model, geometry, normalized_geometry=normalized_geometry)
        rhs = context.rhs.evaluate(np.asarray(operating, dtype=float))
        return context.em.heat_source_for_rhs(np.asarray(state, dtype=float), rhs)

    return factory


def geometry_research_vector_field_factory(model, *, normalized_geometry: bool = True):
    def factory(state, geometry, operating):
        context, _ = _geometry_context(model, geometry, normalized_geometry=normalized_geometry)
        a = np.asarray(state, dtype=float)
        u = np.asarray(operating, dtype=float)
        q = context.em.heat_source_for_rhs(a, context.rhs.evaluate(u))
        return np.linalg.solve(context.M, -context.K @ a + q)

    return factory


def geometry_research_jacobian_factory(model, *, normalized_geometry: bool = True):
    def factory(state, geometry, operating):
        context, _ = _geometry_context(model, geometry, normalized_geometry=normalized_geometry)
        a = np.asarray(state, dtype=float)
        u = np.asarray(operating, dtype=float)
        _, jq = context.em.heat_source_and_jacobian_for_rhs(a, context.rhs.evaluate(u))
        return np.linalg.solve(context.M, -context.K + np.asarray(jq, dtype=float))

    return factory


def geometry_research_temperature_reconstructor(model, *, normalized_geometry: bool = True):
    def reconstruct(state, geometry):
        context, _ = _geometry_context(model, geometry, normalized_geometry=normalized_geometry)
        if hasattr(model, "reference") and hasattr(model.reference, "temperature"):
            return model.reference.temperature(np.asarray(state, dtype=float))
        phi = model.thermal_model.Phi
        baseline = model.reference.reference_temperature
        deviation = phi @ np.asarray(state, dtype=float)
        assembly = context.assembly
        full = np.zeros_like(baseline, dtype=float)
        full[assembly.free_nodes] = deviation
        return baseline + full

    return reconstruct


def geometry_research_thermal_family(model, *, normalized_geometry: bool = True):
    n = len(model.geometry_names)

    def callback(coordinates):
        context, _ = _geometry_context(model, coordinates, normalized_geometry=normalized_geometry)
        return context

    return CallableThermalOperatorFamily(n, callback)


__all__ = [
    "fixed_research_direct_heat_factory",
    "fixed_research_jacobian_factory",
    "fixed_research_temperature_reconstructor",
    "fixed_research_tensor_factory",
    "fixed_research_thermal_family",
    "fixed_research_vector_field_factory",
    "geometry_research_direct_heat_factory",
    "geometry_research_jacobian_factory",
    "geometry_research_temperature_reconstructor",
    "geometry_research_tensor_factory",
    "geometry_research_thermal_family",
    "geometry_research_vector_field_factory",
]
