"""Open-boundary production Maxwell background with geometry-aware thermal ROM."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import scipy.sparse as sp

from .unified_background import BackgroundContext, EPS0, MU0, FixedMultiscaleBackground


class OpenBoundaryBackground(FixedMultiscaleBackground):
    """Fixed Cartesian EM background with physically regularized stranded sources."""

    boundary_model = "silver_muller_impedance"
    source_model = "stranded_rectangular_cross_section_gauss3"
    terminal_model = "impressed_port_path_with_endpoint_charge_balance"

    def __init__(self, *args, thermal_library=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thermal_library = None
        if thermal_library is not None:
            self.set_thermal_library(thermal_library)

    @property
    def thermal_rank(self):
        if self.thermal_library is not None:
            return int(self.thermal_library.rank)
        return super().thermal_rank

    def set_thermal_library(self, library):
        if library is None or int(getattr(library, "rank", 0)) < 1:
            raise ValueError("geometry-aware thermal library must have positive rank")
        if len(library.local_modes) != len(self.coil_materials):
            raise ValueError("thermal library port count differs from background")
        self.thermal_library = library
        self.thermal_basis = None
        return library

    @staticmethod
    def _cross_section_frame(coil, tangent):
        tangent = np.asarray(tangent, float)
        tangent /= max(np.linalg.norm(tangent), np.finfo(float).tiny)
        nominal_normal = np.asarray(coil.pose.rotation[:, 2], float)
        width_axis = np.cross(nominal_normal, tangent)
        if np.linalg.norm(width_axis) <= 1e-12:
            axes = np.eye(3)
            reference = axes[int(np.argmin(np.abs(axes @ tangent)))]
            width_axis = np.cross(reference, tangent)
        width_axis /= max(np.linalg.norm(width_axis), np.finfo(float).tiny)
        thickness_axis = np.cross(tangent, width_axis)
        thickness_axis /= max(np.linalg.norm(thickness_axis), np.finfo(float).tiny)
        if np.dot(thickness_axis, nominal_normal) < 0.0:
            thickness_axis = -thickness_axis
        return width_axis, thickness_axis

    def _deposit_stranded_coil(self, coil, points):
        """Deposit one ampere over the physical rectangular conductor cross section.

        Three-point Gauss quadrature in width/thickness creates a finite-support
        stranded-current model tied to the actual conductor dimensions. The total
        cross-section quadrature weight is one, so refinement changes resolution of
        the same physical source rather than changing its total ampere-turns.
        """
        self._require_inside(points, "coil centerline")
        nodes, weights = np.polynomial.legendre.leggauss(3)
        source = np.zeros(self.n_edges, float)
        heat = np.zeros(self.n_cells, float)
        for p0, p1 in zip(points[:-1], points[1:]):
            d = np.asarray(p1 - p0, float)
            length = float(np.linalg.norm(d))
            if length <= np.finfo(float).tiny:
                continue
            tangent = d / length
            width_axis, thickness_axis = self._cross_section_frame(coil, tangent)
            center = 0.5 * (p0 + p1)
            for u, wu in zip(nodes, weights):
                for v, wv in zip(nodes, weights):
                    qweight = float(wu * wv / 4.0)
                    point = (
                        center
                        + 0.5 * float(coil.conductor_width) * float(u) * width_axis
                        + 0.5 * float(coil.conductor_thickness) * float(v) * thickness_axis
                    )
                    for cell, weight in self._cell_stencil(point):
                        heat[cell] += length * qweight * weight
                    for axis, component in enumerate(d):
                        if abs(component) <= np.finfo(float).tiny:
                            continue
                        stencil = self._edge_stencil(axis, point)
                        if not stencil:
                            continue
                        for edge, weight in stencil:
                            source[edge] += qweight * component * weight / self.edge_lengths[edge]
        if heat.sum() <= 0.0 or np.linalg.norm(source) <= np.finfo(float).tiny:
            raise ValueError("finite-support coil deposition produced a zero physical source")
        heat /= heat.sum()
        return source, heat

    def _deposited_path_integral(self, source):
        """Recover the oriented vector integral represented by an edge source."""
        value = np.zeros(3, float)
        for edge, (axis, _i, _j, _k) in enumerate(self.edge_tuples):
            value[axis] += float(source[edge]) * float(self.edge_lengths[edge])
        return value

    def _spatial_context(self, geometry):
        """Assemble geometry/material/source data using production source regularization."""
        g = self.validate_geometry(geometry)
        spacing = 0.45 * min(np.min(self.dx), np.min(self.dy), np.min(self.dz))
        fractions = {
            name: np.zeros(self.n_cells)
            for name in set(self.coil_materials + self.package_materials + (self.seawater_material,))
        }
        sources, heat_weights = [], []
        source_audits = []
        for coil, material in zip(g.coils, self.coil_materials):
            points = coil.centerline(spacing)
            source, heat = self._deposit_stranded_coil(coil, points)
            sources.append(source)
            heat_weights.append(heat)
            area = float(coil.conductor_width * coil.conductor_thickness)
            length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            endpoint_vector = np.asarray(points[-1] - points[0], float)
            deposited_vector = self._deposited_path_integral(source)
            path_integral_error = float(
                np.linalg.norm(deposited_vector - endpoint_vector)
                / max(length, np.finfo(float).tiny)
            )
            fractions[material] += heat * (area * length) / self.cell_volumes
            source_audits.append({
                "model": self.source_model,
                "terminal_model": self.terminal_model,
                "conductor_width": float(coil.conductor_width),
                "conductor_thickness": float(coil.conductor_thickness),
                "path_length": length,
                "terminal_separation": float(np.linalg.norm(endpoint_vector)),
                "terminal_path_integral_relative_error": path_integral_error,
                "source_norm": float(np.linalg.norm(source)),
                "heat_weight_sum": float(np.sum(heat)),
            })

        copper = np.zeros(self.n_cells)
        for material in self.coil_materials:
            copper += fractions[material]
        scale = np.ones(self.n_cells)
        mask = copper > 1.0
        scale[mask] = 1.0 / copper[mask]
        for material in self.coil_materials:
            fractions[material] *= scale
        occupied = np.minimum(copper, 1.0)
        for package, material in zip(g.packages, self.package_materials):
            raw = self._package_fraction(package)
            add = raw * np.clip(1.0 - occupied, 0.0, 1.0)
            fractions[material] += add
            occupied += add
        fractions[self.seawater_material] = np.clip(1.0 - occupied, 0.0, 1.0)
        closure = float(np.max(np.abs(sum(fractions.values()) - 1.0)))
        if closure > 1e-10:
            raise FloatingPointError("material fractions do not close to unity")
        context = BackgroundContext(g, fractions, np.column_stack(sources), tuple(heat_weights))
        context.source_regularization = tuple(source_audits)
        context.material_fraction_closure_error = closure
        return context

    def geometry_context(self, geometry, *, assemble_thermal=True):
        context = self._spatial_context(geometry)
        if not assemble_thermal:
            return context
        if self.thermal_library is None:
            raise ValueError("geometry-aware thermal library has not been constructed")
        phi = self.thermal_library.basis_for_geometry(self, context.geometry)
        M, K = self.thermal_operator_full(context.fractions)
        Mr = phi.T @ (M @ phi)
        Kr = phi.T @ (K @ phi)
        limit = float(getattr(self.thermal_library, "conditioning_limit", 1e10))
        conditions = {}
        for name, matrix in (("M_r", Mr), ("K_r", Kr)):
            symmetric = 0.5 * (matrix + matrix.T)
            eig = np.linalg.eigvalsh(symmetric)
            if eig.size == 0 or not np.all(np.isfinite(eig)) or eig[0] <= 0.0:
                raise RuntimeError(f"{name}(g) is not positive definite")
            condition = float(eig[-1] / eig[0])
            if not np.isfinite(condition) or condition > limit:
                raise RuntimeError(
                    f"{name}(g) conditioning failed: cond={condition:.3e}, limit={limit:.3e}"
                )
            conditions[name] = condition
        context.thermal_basis = phi
        context.thermal_mass_full = M
        context.thermal_stiffness_full = K
        context.thermal_mass_reduced = Mr
        context.thermal_stiffness_reduced = Kr
        context.thermal_mass_condition = conditions["M_r"]
        context.thermal_stiffness_condition = conditions["K_r"]
        return context

    def _state_temperature(self, context, state):
        material_rise = None
        if state is None:
            rise = np.zeros(self.n_cells)
        elif isinstance(state, Mapping):
            material_rise = {str(name): float(value) for name, value in state.items()}
            if any(not np.isfinite(value) for value in material_rise.values()):
                raise ValueError("material temperature rises must be finite")
            rise = np.zeros(self.n_cells)
            for name, fraction in context.fractions.items():
                rise += fraction * material_rise.get(name, 0.0)
        else:
            value = np.asarray(state, float)
            if value.ndim == 0:
                rise = np.full(self.n_cells, float(value))
            else:
                value = value.reshape(-1)
                if value.shape == (self.n_cells,):
                    rise = value
                else:
                    phi = getattr(context, "thermal_basis", None)
                    if phi is None or value.shape != (phi.shape[1],):
                        raise ValueError("thermal state dimension mismatch")
                    rise = np.asarray(phi, float) @ value
            if np.any(~np.isfinite(rise)):
                raise ValueError("thermal state must be finite")
        return self.ambient_temperature + rise, material_rise

    def _build_edges(self):
        full, lengths, maps = [], [], []
        for axis in range(3):
            amap = {}
            if axis == 0:
                ranges = (range(self.nx), range(self.ny + 1), range(self.nz + 1))
            elif axis == 1:
                ranges = (range(self.nx + 1), range(self.ny), range(self.nz + 1))
            else:
                ranges = (range(self.nx + 1), range(self.ny + 1), range(self.nz))
            for i in ranges[0]:
                for j in ranges[1]:
                    for k in ranges[2]:
                        amap[(i, j, k)] = len(full)
                        full.append((axis, i, j, k))
                        if axis == 0:
                            lengths.append(self.dx[i])
                        elif axis == 1:
                            lengths.append(self.dy[j])
                        else:
                            lengths.append(self.dz[k])
            maps.append(amap)
        self.edge_tuples = tuple(full)
        self.edge_lengths = np.asarray(lengths, float)
        self.edge_maps = maps

        rows, cols, data = [], [], []
        for e, (axis, i, j, k) in enumerate(self.edge_tuples):
            cells = []
            if axis == 0:
                for jj in (j - 1, j):
                    for kk in (k - 1, k):
                        if 0 <= jj < self.ny and 0 <= kk < self.nz:
                            cells.append(self._cell_id(i, jj, kk))
            elif axis == 1:
                for ii in (i - 1, i):
                    for kk in (k - 1, k):
                        if 0 <= ii < self.nx and 0 <= kk < self.nz:
                            cells.append(self._cell_id(ii, j, kk))
            else:
                for ii in (i - 1, i):
                    for jj in (j - 1, j):
                        if 0 <= ii < self.nx and 0 <= jj < self.ny:
                            cells.append(self._cell_id(ii, jj, k))
            for c in cells:
                rows.append(e)
                cols.append(c)
                data.append(self.cell_volumes[c] / (4.0 * self.edge_lengths[e] ** 2))
        self.edge_cell_hodge = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_edges, self.n_cells)
        )
        self._build_boundary_edge_hodge()

    def _build_boundary_edge_hodge(self):
        weights = np.zeros(self.n_edges, float)

        def add(axis, key, area):
            e = self.edge_maps[axis].get(key)
            if e is not None:
                weights[e] += float(area) / (2.0 * self.edge_lengths[e] ** 2)

        for i_face in (0, self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    area = self.dy[j] * self.dz[k]
                    add(1, (i_face, j, k), area); add(1, (i_face, j, k + 1), area)
                    add(2, (i_face, j, k), area); add(2, (i_face, j + 1, k), area)
        for j_face in (0, self.ny):
            for i in range(self.nx):
                for k in range(self.nz):
                    area = self.dx[i] * self.dz[k]
                    add(0, (i, j_face, k), area); add(0, (i, j_face, k + 1), area)
                    add(2, (i, j_face, k), area); add(2, (i + 1, j_face, k), area)
        for k_face in (0, self.nz):
            for i in range(self.nx):
                for j in range(self.ny):
                    area = self.dx[i] * self.dy[j]
                    add(0, (i, j, k_face), area); add(0, (i, j + 1, k_face), area)
                    add(1, (i, j, k_face), area); add(1, (i + 1, j, k_face), area)
        if not np.any(weights > 0.0):
            raise RuntimeError("open boundary mass matrix is empty")
        self.boundary_edge_hodge = weights

    def boundary_admittance(self):
        material = self.materials[self.seawater_material]
        sigma = float(self._temperature_material(self.seawater_material, self.ambient_temperature))
        epsilon = EPS0 * float(material.get("relative_permittivity", 1.0))
        mu = MU0 * float(material.get("relative_permeability", 1.0))
        if mu <= 0.0 or epsilon <= 0.0 or sigma < 0.0:
            raise ValueError("open-boundary medium must be passive with positive epsilon/mu")
        admittance = np.sqrt((epsilon - 1j * sigma / self.omega) / mu)
        if admittance.real < 0.0 or (
            abs(admittance.real) <= np.finfo(float).eps and admittance.imag > 0.0
        ):
            admittance = -admittance
        if admittance.real < -1e-14:
            raise ValueError("open-boundary admittance selected a non-passive branch")
        return complex(admittance)

    def outward_loss_weights(self):
        return float(self.boundary_admittance().real) * self.boundary_edge_hodge

    def em_operator(self, context, state=None):
        volume = super().em_operator(context, state)
        boundary = 1j * self.omega * self.boundary_admittance() * sp.diags(
            self.boundary_edge_hodge
        )
        return (volume + boundary).tocsr()


__all__ = ["OpenBoundaryBackground"]