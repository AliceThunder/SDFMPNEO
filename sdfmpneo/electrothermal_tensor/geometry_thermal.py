"""Serializable exact thermal operators over an affine tetrahedral geometry chart."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..spatial.geometry_chart import AffineTetrahedralGeometryChart
from .vector_field import ReducedThermalOperator


class AffineGeometryThermalOperatorFamily:
    """Reassemble exact reduced thermal ``M(g),K(g)`` without electromagnetic data.

    Geometry coordinates exposed to the neural ROM are normalized to ``[-1,1]``.
    The family stores only the affine mesh chart, tetrahedral thermal material
    coefficients and the shared pullback thermal basis.  It therefore preserves
    the exact thermal geometry dependence while keeping EM out of online
    inference and out of the neural model forward pass.
    """

    def __init__(
        self,
        *,
        reference_vertices: np.ndarray,
        tetrahedra: np.ndarray,
        vertex_directions: np.ndarray,
        parameter_names,
        geometry_reference: np.ndarray,
        geometry_lower: np.ndarray,
        geometry_upper: np.ndarray,
        volumetric_heat_capacity: np.ndarray,
        thermal_conductivity: np.ndarray,
        thermal_basis: np.ndarray,
        free_nodes: np.ndarray,
        cache_size: int = 64,
    ) -> None:
        self.chart = AffineTetrahedralGeometryChart(
            np.asarray(reference_vertices, dtype=float),
            np.asarray(tetrahedra, dtype=int),
            np.asarray(vertex_directions, dtype=float),
            tuple(str(v) for v in parameter_names),
        )
        self.geometry_reference = np.asarray(geometry_reference, dtype=float).reshape(-1)
        self.lower = np.asarray(geometry_lower, dtype=float).reshape(-1)
        self.upper = np.asarray(geometry_upper, dtype=float).reshape(-1)
        self.capacity = np.asarray(volumetric_heat_capacity, dtype=float).reshape(-1)
        self.conductivity = np.asarray(thermal_conductivity, dtype=float).reshape(-1)
        self.thermal_basis = np.asarray(thermal_basis, dtype=float)
        self.free_nodes = np.asarray(free_nodes, dtype=int).reshape(-1)
        self.cache_size = int(cache_size)
        n = self.chart.n_parameters
        if self.geometry_reference.shape != (n,) or self.lower.shape != (n,) or self.upper.shape != (n,):
            raise ValueError("geometry chart/domain dimensions do not match")
        if np.any(~np.isfinite(self.lower + self.upper + self.geometry_reference)) or np.any(self.upper <= self.lower):
            raise ValueError("geometry domain must be finite and strictly ordered")
        nt = self.chart.tetrahedra.shape[0]
        if self.capacity.shape != (nt,) or self.conductivity.shape != (nt,):
            raise ValueError("thermal material arrays must contain one value per tetrahedron")
        if np.any(self.capacity <= 0.0) or np.any(self.conductivity <= 0.0):
            raise ValueError("thermal material coefficients must be positive")
        if self.thermal_basis.ndim != 2 or self.thermal_basis.shape[0] != len(self.free_nodes):
            raise ValueError("thermal basis/free-node dimensions do not match")
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        self._cache: OrderedDict[tuple[float, ...], ReducedThermalOperator] = OrderedDict()

    @classmethod
    def from_geometry_research_model(cls, model, *, cache_size: int = 64):
        return cls(
            reference_vertices=model.chart.reference_vertices,
            tetrahedra=model.chart.tetrahedra,
            vertex_directions=model.chart.vertex_directions,
            parameter_names=model.chart.parameter_names,
            geometry_reference=model.geometry_reference,
            geometry_lower=model.lower,
            geometry_upper=model.upper,
            volumetric_heat_capacity=model.capacity,
            thermal_conductivity=model.conductivity,
            thermal_basis=model.thermal_model.Phi,
            free_nodes=model.reference.core.thermal_assembly.free_nodes,
            cache_size=cache_size,
        )

    @property
    def geometry_dimension(self) -> int:
        return self.chart.n_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.chart.parameter_names

    @property
    def thermal_rank(self) -> int:
        return self.thermal_basis.shape[1]

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        z = np.asarray(normalized, dtype=float).reshape(-1)
        if z.shape != (self.geometry_dimension,) or np.any(~np.isfinite(z)):
            raise ValueError("normalized geometry dimension mismatch")
        if np.any(z < -1.0) or np.any(z > 1.0):
            raise ValueError("normalized geometry is outside [-1,1]")
        return 0.5 * (self.lower + self.upper + z * (self.upper - self.lower))

    def operator(self, geometry: np.ndarray) -> ReducedThermalOperator:
        z = np.asarray(geometry, dtype=float).reshape(-1)
        if z.shape != (self.geometry_dimension,) or np.any(~np.isfinite(z)):
            raise ValueError("geometry dimension mismatch")
        key = tuple(float(v) for v in z)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        physical = self.denormalize(z)
        mesh = self.chart.mesh(physical - self.geometry_reference)
        assembly = mesh.assemble_p1_thermal(
            rho_cp_tetra=self.capacity,
            conductivity_tetra=self.conductivity,
            homogeneous_dirichlet_boundary=True,
        )
        if not np.array_equal(assembly.free_nodes, self.free_nodes):
            raise RuntimeError("geometry changed the shared thermal coordinate topology")
        phi = self.thermal_basis
        mass = phi.T @ (assembly.M @ phi)
        stiffness = phi.T @ (assembly.K @ phi)
        result = ReducedThermalOperator(mass, stiffness)
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def persistence_arrays(self) -> dict[str, np.ndarray]:
        return {
            "geometry_reference_vertices": self.chart.reference_vertices,
            "geometry_tetrahedra": self.chart.tetrahedra,
            "geometry_vertex_directions": self.chart.vertex_directions,
            "geometry_reference": self.geometry_reference,
            "geometry_lower": self.lower,
            "geometry_upper": self.upper,
            "geometry_capacity": self.capacity,
            "geometry_conductivity": self.conductivity,
            "geometry_thermal_basis": self.thermal_basis,
            "geometry_free_nodes": self.free_nodes,
            "geometry_parameter_names": np.asarray(self.parameter_names, dtype="U"),
            "geometry_cache_size": np.array(self.cache_size, dtype=np.int64),
        }

    @classmethod
    def from_persistence(cls, data):
        return cls(
            reference_vertices=data["geometry_reference_vertices"],
            tetrahedra=data["geometry_tetrahedra"],
            vertex_directions=data["geometry_vertex_directions"],
            parameter_names=tuple(str(v) for v in data["geometry_parameter_names"].tolist()),
            geometry_reference=data["geometry_reference"],
            geometry_lower=data["geometry_lower"],
            geometry_upper=data["geometry_upper"],
            volumetric_heat_capacity=data["geometry_capacity"],
            thermal_conductivity=data["geometry_conductivity"],
            thermal_basis=data["geometry_thermal_basis"],
            free_nodes=data["geometry_free_nodes"],
            cache_size=int(data["geometry_cache_size"]),
        )


__all__ = ["AffineGeometryThermalOperatorFamily"]
