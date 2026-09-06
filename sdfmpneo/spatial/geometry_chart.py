from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tetra3d import TetrahedralComplex3D


def _tet_jacobian(vertices: np.ndarray, tet: np.ndarray) -> np.ndarray:
    x = vertices[tet]
    return np.column_stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]])


@dataclass(frozen=True)
class GeometryBoxCertificate:
    lower: np.ndarray
    upper: np.ndarray
    maximum_relative_jacobian_perturbation: float
    directional_distortion_bounds: np.ndarray
    minimum_jacobian_singular_ratio: float
    maximum_jacobian_singular_ratio: float
    hcurl_mass_ratio: tuple[float, float]
    hcurl_curl_ratio: tuple[float, float]
    p1_mass_ratio: tuple[float, float]
    p1_stiffness_ratio: tuple[float, float]

    @property
    def certified_nondegenerate(self) -> bool:
        return self.maximum_relative_jacobian_perturbation < 1.0


@dataclass(frozen=True)
class AffineTetrahedralGeometryChart:
    """Reference-domain affine vertex chart with rigorous element distortion bounds.

    Vertices vary as ``X(mu)=X0+sum_k mu_k D_k`` while tetrahedral connectivity
    stays fixed.  On a parameter box, each element deformation relative to the
    box center is ``F=I+E`` with

        ||E||_2 <= sum_k h_k ||J_k J_c^{-1}||_2 = rho.

    ``rho<1`` proves orientation/non-degeneracy on the whole box.  The singular
    value enclosure ``1-rho <= sigma(F) <= 1+rho`` then gives exact quadratic-
    form ratios for covariant Nedelec mass/curl and P1 mass/stiffness forms.
    """

    reference_vertices: np.ndarray
    tetrahedra: np.ndarray
    vertex_directions: np.ndarray
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        X = np.asarray(self.reference_vertices, dtype=float)
        T = np.asarray(self.tetrahedra, dtype=int)
        D = np.asarray(self.vertex_directions, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("reference_vertices must have shape (n_nodes,3)")
        if T.ndim != 2 or T.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (n_tetra,4)")
        if D.ndim != 3 or D.shape[1:] != X.shape:
            raise ValueError("vertex_directions must have shape (n_parameter,n_nodes,3)")
        if len(self.parameter_names) != D.shape[0] or len(set(self.parameter_names)) != D.shape[0]:
            raise ValueError("parameter_names must be unique and match vertex_directions")
        # Build once to prove the reference connectivity itself is valid.
        TetrahedralComplex3D.build(X, T)
        object.__setattr__(self, "reference_vertices", X)
        object.__setattr__(self, "tetrahedra", T)
        object.__setattr__(self, "vertex_directions", D)
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))

    @property
    def n_parameters(self) -> int:
        return self.vertex_directions.shape[0]

    def vertices(self, parameters: np.ndarray) -> np.ndarray:
        p = np.asarray(parameters, dtype=float)
        if p.shape != (self.n_parameters,):
            raise ValueError("geometry parameter dimension mismatch")
        return self.reference_vertices + np.tensordot(p, self.vertex_directions, axes=(0, 0))

    def mesh(self, parameters: np.ndarray) -> TetrahedralComplex3D:
        return TetrahedralComplex3D.build(self.vertices(parameters), self.tetrahedra)

    def certify_box(self, lower: np.ndarray, upper: np.ndarray) -> GeometryBoxCertificate:
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if lo.shape != (self.n_parameters,) or hi.shape != lo.shape or np.any(hi < lo):
            raise ValueError("invalid geometry parameter box")
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        Xc = self.vertices(center)
        directional = np.zeros(self.n_parameters, dtype=float)
        rho = 0.0

        for tet in self.tetrahedra:
            Jc = _tet_jacobian(Xc, tet)
            det = float(np.linalg.det(Jc))
            if det <= 0.0:
                raise ValueError("geometry chart center contains an inverted/degenerate tetrahedron")
            invJ = np.linalg.inv(Jc)
            local = np.empty(self.n_parameters, dtype=float)
            for k in range(self.n_parameters):
                Jk = _tet_jacobian(self.vertex_directions[k], tet)
                local[k] = float(np.linalg.norm(Jk @ invJ, ord=2))
            directional = np.maximum(directional, local)
            rho = max(rho, float(np.dot(half, local)))

        if rho >= 1.0:
            return GeometryBoxCertificate(
                lo, hi, rho, directional,
                0.0, float("inf"),
                (0.0, float("inf")), (0.0, float("inf")),
                (0.0, float("inf")), (0.0, float("inf")),
            )

        smin = 1.0 - rho
        smax = 1.0 + rho
        det_lo = smin**3
        det_hi = smax**3
        covariant = (det_lo / smax**2, det_hi / smin**2)
        curl = (smin**2 / det_hi, smax**2 / det_lo)
        return GeometryBoxCertificate(
            lower=lo,
            upper=hi,
            maximum_relative_jacobian_perturbation=float(rho),
            directional_distortion_bounds=directional,
            minimum_jacobian_singular_ratio=smin,
            maximum_jacobian_singular_ratio=smax,
            hcurl_mass_ratio=(float(covariant[0]), float(covariant[1])),
            hcurl_curl_ratio=(float(curl[0]), float(curl[1])),
            p1_mass_ratio=(float(det_lo), float(det_hi)),
            p1_stiffness_ratio=(float(covariant[0]), float(covariant[1])),
        )
