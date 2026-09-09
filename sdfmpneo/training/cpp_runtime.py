from __future__ import annotations

from threading import RLock

import numpy as np
import scipy.sparse as sp

from ..cpp_training_backend import (
    nedelec_local,
    p1_thermal_local,
    reduced_assemble_many,
    reduced_assemble_products,
)

_INSTALLED = False
_CACHE_LOCK = RLock()
_GAUGE_CACHE = {}
_CONDUCTIVE_CACHE = {}
_ACTIVE_COMPONENT_CACHE = {}


def install_cpp_training_backend() -> None:
    """Install exact-equivalent C++/vectorized hot paths with Python fallback."""
    global _INSTALLED
    if _INSTALLED:
        return

    from ..spatial.tetra3d import TetrahedralComplex3D, TetrahedralThermalAssembly
    from ..spatial.geometry_chart import AffineTetrahedralGeometryChart
    from ..em import fast_reduced as fast
    from ..em import terminal_ports as terminal

    original_p1 = TetrahedralComplex3D.assemble_p1_thermal
    original_nedelec = TetrahedralComplex3D.assemble_nedelec_edge_matrices
    original_gauge = TetrahedralComplex3D.gauge_basis
    original_conductive = TetrahedralComplex3D.conductive_gradient
    original_chart_mesh = AffineTetrahedralGeometryChart.mesh
    original_active_components = terminal._active_components

    def chart_mesh(self, parameters):
        p = np.asarray(parameters, dtype=float)
        if p.shape != (self.n_parameters,):
            raise ValueError("geometry parameter dimension mismatch")
        vertices = self.reference_vertices + np.tensordot(
            p, self.vertex_directions, axes=(0, 0)
        )
        reference = getattr(self, "_sdfmpneo_topology_mesh", None)
        if reference is None:
            reference = TetrahedralComplex3D.build(self.reference_vertices, self.tetrahedra)
            object.__setattr__(self, "_sdfmpneo_topology_mesh", reference)
        tets = reference.tetrahedra
        x = vertices[tets]
        jacobian = np.stack(
            [x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]],
            axis=2,
        )
        det = np.linalg.det(jacobian)
        if np.any(~np.isfinite(det)) or np.any(det <= 0.0):
            return original_chart_mesh(self, p)
        return TetrahedralComplex3D(
            vertices=np.asarray(vertices, dtype=float),
            tetrahedra=tets,
            volumes=np.asarray(det / 6.0, dtype=float),
            edge_vertices=reference.edge_vertices,
            face_vertices=reference.face_vertices,
            tet_edge_indices=reference.tet_edge_indices,
            grad=reference.grad,
            curl=reference.curl,
            boundary_face_indices=reference.boundary_face_indices,
        )

    def gauge_basis(self):
        key = (id(self.grad), self.n_nodes, self.n_edges)
        with _CACHE_LOCK:
            cached = _GAUGE_CACHE.get(key)
        if cached is not None:
            return cached
        value = original_gauge(self)
        with _CACHE_LOCK:
            cached = _GAUGE_CACHE.setdefault(key, value)
        return cached

    def conductive_gradient(self, active_tetrahedra):
        active = np.asarray(active_tetrahedra, dtype=bool)
        if active.shape != (self.n_tetrahedra,):
            raise ValueError("active_tetrahedra must have shape (n_tetrahedra,)")
        key = (id(self.grad), active.tobytes())
        with _CACHE_LOCK:
            cached = _CONDUCTIVE_CACHE.get(key)
        if cached is not None:
            return cached
        value = original_conductive(self, active)
        with _CACHE_LOCK:
            cached = _CONDUCTIVE_CACHE.setdefault(key, value)
        return cached

    def assemble_p1_thermal(self, rho_cp_tetra, conductivity_tetra, *, homogeneous_dirichlet_boundary=True):
        rho_cp = np.asarray(rho_cp_tetra, dtype=float)
        kappa = np.asarray(conductivity_tetra, dtype=float)
        if rho_cp.shape != (self.n_tetrahedra,) or kappa.shape != (self.n_tetrahedra,):
            raise ValueError("thermal tetra arrays must have shape (n_tetrahedra,)")
        if np.any(rho_cp <= 0) or np.any(kappa <= 0):
            raise ValueError("rho_cp and thermal conductivity must be positive")
        local = p1_thermal_local(self.vertices, self.tetrahedra, rho_cp, kappa)
        if local is None:
            return original_p1(
                self, rho_cp, kappa,
                homogeneous_dirichlet_boundary=homogeneous_dirichlet_boundary,
            )
        local_M, local_K, volumes = local
        if not np.allclose(volumes, self.volumes, rtol=5e-13, atol=5e-15):
            raise FloatingPointError("C++ P1 kernel geometry volume mismatch")
        rows = np.repeat(self.tetrahedra, 4, axis=1).ravel()
        cols = np.tile(self.tetrahedra, (1, 4)).ravel()
        M_full = sp.coo_matrix(
            (local_M.ravel(), (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        K_full = sp.coo_matrix(
            (local_K.ravel(), (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        M_full.sum_duplicates(); K_full.sum_duplicates()
        M_full.eliminate_zeros(); K_full.eliminate_zeros()
        boundary = self.boundary_nodes()
        if homogeneous_dirichlet_boundary:
            mask = np.ones(self.n_nodes, dtype=bool); mask[boundary] = False
            free = np.nonzero(mask)[0]
            if free.size == 0:
                raise ValueError("homogeneous Dirichlet boundary leaves no thermal free nodes")
            M = M_full[free][:, free].tocsr(); K = K_full[free][:, free].tocsr()
        else:
            free = np.arange(self.n_nodes, dtype=int); M = M_full; K = K_full
        return TetrahedralThermalAssembly(
            M=M, K=K, free_nodes=free, boundary_nodes=boundary,
            n_full_nodes=self.n_nodes, tetrahedra=self.tetrahedra.copy(),
        )

    def assemble_nedelec_edge_matrices(self, reluctivity_tetra, conductivity_tetra):
        nu = np.asarray(reluctivity_tetra, dtype=float)
        sigma = np.asarray(conductivity_tetra, dtype=float)
        if nu.shape != (self.n_tetrahedra,) or sigma.shape != (self.n_tetrahedra,):
            raise ValueError("tetra material arrays must have shape (n_tetrahedra,)")
        if np.any(nu <= 0) or np.any(sigma < 0):
            raise ValueError("reluctivity must be positive and conductivity non-negative")
        local = nedelec_local(self.vertices, self.tetrahedra, nu, sigma)
        if local is None:
            return original_nedelec(self, nu, sigma)
        local_K, local_M = local
        edges = self.tet_edge_indices
        rows = np.repeat(edges, 6, axis=1).ravel()
        cols = np.tile(edges, (1, 6)).ravel()
        K = sp.coo_matrix(
            (local_K.ravel(), (rows, cols)), shape=(self.n_edges, self.n_edges)
        ).tocsr()
        M = sp.coo_matrix(
            (local_M.ravel(), (rows, cols)), shape=(self.n_edges, self.n_edges)
        ).tocsr()
        K.sum_duplicates(); M.sum_duplicates(); K.eliminate_zeros(); M.eliminate_zeros()
        return K, M

    def reduced_init(self, mesh, edge_fields):
        fields = np.asarray(edge_fields, dtype=complex)
        if fields.ndim != 2 or fields.shape[0] != mesh.n_edges:
            raise ValueError("edge_fields must have shape (n_edges,n_reduced)")
        self.mesh = mesh
        self.fields = fields
        tets = np.asarray(mesh.tetrahedra, dtype=int)
        x = np.asarray(mesh.vertices, dtype=float)[tets]
        B = np.concatenate([np.ones((tets.shape[0], 4, 1)), x], axis=2)
        inverse = np.linalg.inv(B)
        gradients = np.transpose(inverse[:, 1:, :], (0, 2, 1))
        pi = np.array([0, 0, 0, 1, 1, 2], dtype=int)
        pj = np.array([1, 2, 3, 2, 3, 3], dtype=int)
        vi = tets[:, pi]; vj = tets[:, pj]
        left = np.where(vi < vj, pi[np.newaxis, :], pj[np.newaxis, :])
        right = np.where(vi < vj, pj[np.newaxis, :], pi[np.newaxis, :])
        coefficients = np.zeros((tets.shape[0], 6, 4, 3), dtype=float)
        q = np.arange(tets.shape[0])[:, np.newaxis]
        pidx = np.arange(6)[np.newaxis, :]
        coefficients[q, pidx, left, :] = gradients[q, right, :]
        coefficients[q, pidx, right, :] = -gradients[q, left, :]
        self._cpp_coefficients = np.ascontiguousarray(coefficients)
        self._cpp_local_fields = np.ascontiguousarray(
            fields[mesh.tet_edge_indices], dtype=np.complex128
        )

    def reduced_assemble_many_method(self, tetra_polynomial_families):
        families = tuple(tuple(values) for values in tetra_polynomial_families)
        for polynomials in families:
            if len(polynomials) != self.mesh.n_tetrahedra:
                raise ValueError("one polynomial is required for every tetrahedron")
        value = reduced_assemble_many(
            self.mesh.volumes, self._cpp_coefficients, self._cpp_local_fields, families
        )
        if value is not None:
            return value
        n = self.fields.shape[1]
        reduced = np.zeros((len(families), n, n), dtype=complex)
        for q in range(self.mesh.n_tetrahedra):
            coeff = self._cpp_coefficients[q]; fields = self._cpp_local_fields[q]
            volume = float(self.mesh.volumes[q])
            for family, polynomials in enumerate(families):
                poly = polynomials[q]
                if not poly:
                    continue
                moments = fast._ReducedNedelecAssembler._moments(volume, poly)
                local = np.einsum("pik,ij,qjk->pq", coeff, moments, coeff)
                reduced[family] += fields.conj().T @ (local @ fields)
        return reduced

    def reduced_assemble_products_method(self, tetra_polynomial_families, multiplier_families):
        families = tuple(tuple(values) for values in tetra_polynomial_families)
        multipliers = tuple(tuple(values) for values in multiplier_families)
        for values in families + multipliers:
            if len(values) != self.mesh.n_tetrahedra:
                raise ValueError("one polynomial is required for every tetrahedron")
        value = reduced_assemble_products(
            self.mesh.volumes, self._cpp_coefficients, self._cpp_local_fields,
            families, multipliers,
        )
        if value is not None:
            return value
        from ..spatial.barycentric_polynomial import polynomial_multiply
        n = self.fields.shape[1]
        reduced = np.zeros((len(multipliers), len(families), n, n), dtype=complex)
        for q in range(self.mesh.n_tetrahedra):
            coeff = self._cpp_coefficients[q]; fields = self._cpp_local_fields[q]
            volume = float(self.mesh.volumes[q])
            for output, tests in enumerate(multipliers):
                test = tests[q]
                if not test:
                    continue
                for family, polynomials in enumerate(families):
                    poly = polynomials[q]
                    if not poly:
                        continue
                    weighted = polynomial_multiply(poly, test)
                    if not weighted:
                        continue
                    moments = fast._ReducedNedelecAssembler._moments(volume, weighted)
                    local = np.einsum("pik,ij,qjk->pq", coeff, moments, coeff)
                    reduced[output, family] += fields.conj().T @ (local @ fields)
        return reduced

    def surface_nodal_weights(tagged, physical_tag):
        triangles = tagged.boundary_triangles[tagged.boundary_mask(physical_tag)]
        if triangles.size == 0:
            raise ValueError(f"terminal physical tag {physical_tag} has no boundary triangles")
        xyz = tagged.mesh.vertices
        points = xyz[triangles]
        areas = 0.5 * np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
        )
        if np.any(areas <= 0.0):
            raise ValueError("terminal contains a degenerate boundary triangle")
        weights = np.zeros(tagged.mesh.n_nodes, dtype=float)
        np.add.at(weights, triangles.ravel(), np.repeat(areas / 3.0, 3))
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("terminal surface has zero area")
        return weights / total

    def active_components(problem):
        mesh = problem.mesh
        support = np.zeros(mesh.n_tetrahedra, dtype=bool)
        T0 = np.asarray(problem.temperature_reference_local, dtype=float)
        for region in problem.conductivity_regions:
            mask = np.asarray(region.mask, dtype=bool)
            if not np.any(mask):
                continue
            values = np.asarray(region.law.evaluate(T0[mask]), dtype=float)
            dynamic = np.any(np.asarray(region.law.derivative(T0[mask])) != 0.0, axis=1)
            nonzero = np.any(values > 0.0, axis=1)
            idx = np.flatnonzero(mask); support[idx[nonzero | dynamic]] = True
        key = (id(mesh.grad), support.tobytes())
        with _CACHE_LOCK:
            cached = _ACTIVE_COMPONENT_CACHE.get(key)
        if cached is not None:
            return cached
        value = original_active_components(problem)
        with _CACHE_LOCK:
            cached = _ACTIVE_COMPONENT_CACHE.setdefault(key, value)
        return cached

    AffineTetrahedralGeometryChart.mesh = chart_mesh
    TetrahedralComplex3D.gauge_basis = gauge_basis
    TetrahedralComplex3D.conductive_gradient = conductive_gradient
    TetrahedralComplex3D.assemble_p1_thermal = assemble_p1_thermal
    TetrahedralComplex3D.assemble_nedelec_edge_matrices = assemble_nedelec_edge_matrices
    fast._ReducedNedelecAssembler.__init__ = reduced_init
    fast._ReducedNedelecAssembler.assemble_many = reduced_assemble_many_method
    fast._ReducedNedelecAssembler.assemble_products = reduced_assemble_products_method
    terminal._surface_nodal_weights = surface_nodal_weights
    terminal._active_components = active_components
    _INSTALLED = True
