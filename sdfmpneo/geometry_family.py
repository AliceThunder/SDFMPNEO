from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrothermal import ElectroThermalFieldEvaluation
from .thermal.atlas import canonicalize_thermal_subspaces


@dataclass(frozen=True)
class GeometryFieldChart:
    geometry: np.ndarray
    local_to_canonical: np.ndarray
    canonical_to_local: np.ndarray
    spectral_clusters: tuple
    field: object


class CanonicalGeometryElectroThermalField:
    """Exact coordinate transform of one local electrothermal reduced model.

    If the target local thermal basis is ``Phi`` and the atlas returns
    ``Phi Q`` as the canonical aligned basis, then

        a_local = Q a_canonical,
        F_canonical = Q^T F_local,
        J_canonical = Q^T J_local Q.

    No dynamical approximation is introduced by this wrapper.  Eigenvalue
    crossings are handled at the certified spectral-subspace level by the atlas;
    the network never tracks a fragile individual eigenvector across a cluster.
    """

    def __init__(self, local_field, canonical_to_local: np.ndarray, chart_step) -> None:
        Q = np.asarray(canonical_to_local, dtype=float)
        n = int(local_field.n_modes)
        if Q.shape != (n, n):
            raise ValueError("canonical_to_local must be square in thermal rank")
        defect = float(np.linalg.norm(Q.T @ Q - np.eye(n), ord=2))
        backward = 64.0 * max(1, n) * np.finfo(float).eps * max(1.0, float(np.linalg.norm(Q, ord=2)) ** 2)
        if defect > backward:
            raise np.linalg.LinAlgError("thermal atlas coordinate transform lost orthogonality")
        self.local_field = local_field
        self.thermal_model = local_field.thermal_model
        self.electromagnetic_model = local_field.electromagnetic_model
        self.em_model = local_field.em_model
        self.canonical_to_local = Q
        self.local_to_canonical = Q.T
        self.chart_step = chart_step

    @property
    def n_modes(self) -> int:
        return self.canonical_to_local.shape[0]

    @property
    def n_operating(self) -> int:
        return int(self.local_field.n_operating)

    def rhs(self, operating: np.ndarray | None = None) -> np.ndarray:
        return self.local_field.rhs(operating)

    def evaluate(self, state: np.ndarray, operating: np.ndarray | None = None) -> ElectroThermalFieldEvaluation:
        a = np.asarray(state, dtype=float)
        if a.shape != (self.n_modes,):
            raise ValueError("canonical state dimension mismatch")
        local_a = self.canonical_to_local @ a
        local = self.local_field.evaluate(local_a, operating)
        QT = self.local_to_canonical
        Q = self.canonical_to_local
        q = QT @ np.asarray(local.heat_source, dtype=float)
        Jq = QT @ np.asarray(local.heat_source_jacobian, dtype=float) @ Q
        F = QT @ np.asarray(local.vector_field, dtype=float)
        JF = QT @ np.asarray(local.vector_field_jacobian, dtype=float) @ Q
        u = np.asarray(local.operating, dtype=float)
        return ElectroThermalFieldEvaluation(
            state=a.copy(),
            operating=u.copy(),
            rhs=np.asarray(local.rhs, dtype=complex),
            heat_source=q,
            heat_source_jacobian=Jq,
            vector_field=F,
            vector_field_jacobian=JF,
        )

    def residual(self, state: np.ndarray, derivative: np.ndarray, operating: np.ndarray | None = None) -> np.ndarray:
        da = np.asarray(derivative, dtype=float)
        evaluation = self.evaluate(state, operating)
        if da.shape != evaluation.vector_field.shape:
            raise ValueError("derivative dimension mismatch")
        return da - evaluation.vector_field

    def contraction_margin(self, state: np.ndarray, operating: np.ndarray | None = None) -> float:
        J = self.evaluate(state, operating).vector_field_jacobian
        return float(-np.max(np.linalg.eigvalsh(0.5 * (J + J.T))))


class CertifiedGeometryElectroThermalFamily:
    """Deterministic family of physical fields on one canonical thermal atlas.

    ``field_factory(G)`` must build the production local
    ``CertifiedElectroThermalVectorField`` at the declared geometry.  Geometry
    dependence is therefore physical/operator dependence, never a learned weight
    generator.
    """

    def __init__(self, reference_thermal_model, geometry_names, field_factory) -> None:
        self.reference_thermal_model = reference_thermal_model
        self.geometry_names = tuple(str(v) for v in geometry_names)
        if not self.geometry_names or len(set(self.geometry_names)) != len(self.geometry_names):
            raise ValueError("geometry_names must be unique and non-empty")
        self.field_factory = field_factory

    @property
    def n_geometry(self) -> int:
        return len(self.geometry_names)

    @property
    def n_modes(self) -> int:
        return int(self.reference_thermal_model.rank)

    def chart(self, geometry: np.ndarray) -> GeometryFieldChart:
        g = np.asarray(geometry, dtype=float)
        if g.shape != (self.n_geometry,):
            raise ValueError("geometry parameter dimension mismatch")
        local_field = self.field_factory(g)
        target = local_field.thermal_model
        step = canonicalize_thermal_subspaces(self.reference_thermal_model, target)
        M = target.M
        Phi = np.asarray(target.Phi, dtype=float)
        aligned = np.asarray(step.aligned_basis, dtype=float)
        # Phi is M-orthonormal, so Q=Phi^T M (Phi Q).
        Q = np.asarray(Phi.T @ (M @ aligned), dtype=float)
        canonical = CanonicalGeometryElectroThermalField(local_field, Q, step)
        return GeometryFieldChart(
            geometry=g.copy(),
            local_to_canonical=Q.T.copy(),
            canonical_to_local=Q.copy(),
            spectral_clusters=tuple(step.clusters),
            field=canonical,
        )

    def field(self, geometry: np.ndarray) -> CanonicalGeometryElectroThermalField:
        return self.chart(geometry).field
