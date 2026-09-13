"""Fixed multiscale Cartesian Maxwell/thermal background.

Geometry changes material occupancy and impressed coil currents, never the
background topology.  The thermal basis is installed after construction by the
transient residual/energy driven builder.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import itertools

import numpy as np
import scipy.sparse as sp

from .unified_geometry import UnifiedUWPTGeometry

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


def stretched_axis(bounds, core_half, fine_step, growth, max_step, center=0.0):
    lo, hi = map(float, bounds)
    c = float(center)
    h = float(core_half)
    fine = float(fine_step)
    if not lo < c < hi or h <= 0 or fine <= 0 or growth < 1 or max_step < fine:
        raise ValueError("invalid background axis settings")
    a = max(lo, c - h)
    b = min(hi, c + h)
    n = max(1, int(np.ceil((b - a) / fine)))
    core = np.linspace(a, b, n + 1)
    left = []
    x = a
    step = fine
    while x > lo:
        step = min(max_step, step * growth)
        x2 = max(lo, x - step)
        left.append(x2)
        x = x2
    right = []
    x = b
    step = fine
    while x < hi:
        step = min(max_step, step * growth)
        x2 = min(hi, x + step)
        right.append(x2)
        x = x2
    return np.asarray(list(reversed(left)) + core.tolist() + right, float)


@dataclass
class BackgroundContext:
    geometry: UnifiedUWPTGeometry
    fractions: dict
    source_shape: np.ndarray
    line_heat_weights: tuple
    thermal_mass_full: object | None = None
    thermal_stiffness_full: object | None = None
    thermal_mass_reduced: np.ndarray | None = None
    thermal_stiffness_reduced: np.ndarray | None = None

    @property
    def has_thermal_operators(self) -> bool:
        return self.thermal_mass_reduced is not None and self.thermal_stiffness_reduced is not None


class FixedMultiscaleBackground:
    def __init__(self, x, y, z, *, frequency_hz, materials, coil_materials,
                 package_materials, seawater_material, thermal_basis=None,
                 ambient_temperature=293.15):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.z = np.asarray(z, float)
        for axis in (self.x, self.y, self.z):
            if axis.ndim != 1 or len(axis) < 3 or np.any(np.diff(axis) <= 0):
                raise ValueError("background axes must be strictly increasing")
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.nx, self.ny, self.nz = len(self.dx), len(self.dy), len(self.dz)
        self.frequency_hz = float(frequency_hz)
        self.omega = 2 * np.pi * self.frequency_hz
        self.ambient_temperature = float(ambient_temperature)
        self.materials = {str(k): dict(v) for k, v in materials.items()}
        self.coil_materials = tuple(map(str, coil_materials))
        self.package_materials = tuple(map(str, package_materials))
        self.seawater_material = str(seawater_material)
        if len(self.coil_materials) != len(self.package_materials):
            raise ValueError("coil/package material lists must match")
        required = set(self.coil_materials + self.package_materials + (self.seawater_material,))
        if required - set(self.materials):
            raise ValueError("background region materials are incomplete")
        self._build_cells()
        self._build_edges()
        self._build_curl()
        self._build_reconstruction()
        self.thermal_basis = None
        if thermal_basis is not None:
            self.set_thermal_basis(thermal_basis)

    @classmethod
    def from_config(cls, cfg, *, frequency_hz, materials, coil_materials,
                    package_materials, seawater_material, ambient_temperature):
        bounds = np.asarray(cfg["bounds"], float)
        core = np.asarray(cfg["core_half_extent"], float)
        center = np.asarray(cfg.get("core_center", [0, 0, 0]), float)
        if bounds.shape != (3, 2) or core.shape != (3,) or center.shape != (3,):
            raise ValueError("BACKGROUND bounds/core dimensions are invalid")
        axes = [
            stretched_axis(bounds[k], core[k], cfg["fine_step"], cfg.get("growth", 1.4),
                           cfg.get("max_step", 4 * cfg["fine_step"]), center[k])
            for k in range(3)
        ]
        return cls(*axes, frequency_hz=frequency_hz, materials=materials,
                   coil_materials=coil_materials, package_materials=package_materials,
                   seawater_material=seawater_material, ambient_temperature=ambient_temperature)

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def n_edges(self):
        return len(self.edge_lengths)

    @property
    def thermal_rank(self):
        return 0 if self.thermal_basis is None else int(self.thermal_basis.shape[1])

    def set_thermal_basis(self, basis):
        Phi = np.asarray(basis, float)
        if Phi.ndim != 2 or Phi.shape[0] != self.n_cells or Phi.shape[1] < 1:
            raise ValueError("thermal basis must have shape (n_cells, positive_rank)")
        if np.any(~np.isfinite(Phi)):
            raise ValueError("thermal basis must be finite")
        gram = Phi.T @ (self.cell_volumes[:, None] * Phi)
        gram = 0.5 * (gram + gram.T)
        if np.linalg.matrix_rank(gram) != Phi.shape[1]:
            raise ValueError("thermal basis is rank deficient")
        L = np.linalg.cholesky(gram)
        self.thermal_basis = Phi @ np.linalg.inv(L.T)
        return self.thermal_basis

    def project_temperature_rise(self, rise):
        """Legacy volume projection; production initial states use model M projection."""
        if self.thermal_basis is None:
            raise ValueError("thermal basis has not been constructed")
        value = np.asarray(rise, float)
        if value.ndim == 0:
            value = np.full(self.n_cells, float(value))
        value = value.reshape(-1)
        if value.shape != (self.n_cells,) or np.any(~np.isfinite(value)):
            raise ValueError("temperature rise must be scalar or one value per background cell")
        return self.thermal_basis.T @ (self.cell_volumes * value)

    def _cell_id(self, i, j, k):
        return (i * self.ny + j) * self.nz + k

    def _build_cells(self):
        xc = (self.x[:-1] + self.x[1:]) / 2
        yc = (self.y[:-1] + self.y[1:]) / 2
        zc = (self.z[:-1] + self.z[1:]) / 2
        self.cell_axes = (xc, yc, zc)
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        self.cell_centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        DX, DY, DZ = np.meshgrid(self.dx, self.dy, self.dz, indexing="ij")
        self.cell_widths = np.column_stack([DX.ravel(), DY.ravel(), DZ.ravel()])
        self.cell_volumes = np.prod(self.cell_widths, axis=1)

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
                        boundary = (
                            (axis == 0 and (j in (0, self.ny) or k in (0, self.nz)))
                            or (axis == 1 and (i in (0, self.nx) or k in (0, self.nz)))
                            or (axis == 2 and (i in (0, self.nx) or j in (0, self.ny)))
                        )
                        if boundary:
                            continue
                        amap[(i, j, k)] = len(full)
                        full.append((axis, i, j, k))
                        lengths.append((self.dx[i], self.dy[j], self.dz[k])[axis])
            maps.append(amap)
        self.edge_tuples = tuple(full)
        self.edge_lengths = np.asarray(lengths)
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
        self.edge_cell_hodge = sp.csr_matrix((data, (rows, cols)), shape=(self.n_edges, self.n_cells))

    def _build_curl(self):
        rows, cols, vals = [], [], []
        hrows, hcols, hdata = [], [], []
        face = 0

        def add(edge_map, key, sign):
            idx = edge_map.get(key)
            if idx is not None:
                rows.append(face); cols.append(idx); vals.append(sign)

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz + 1):
                    add(self.edge_maps[0], (i, j, k), 1); add(self.edge_maps[1], (i + 1, j, k), 1)
                    add(self.edge_maps[0], (i, j + 1, k), -1); add(self.edge_maps[1], (i, j, k), -1)
                    area = self.dx[i] * self.dy[j]
                    if k > 0:
                        hrows.append(face); hcols.append(self._cell_id(i, j, k - 1)); hdata.append(0.5 * self.dz[k - 1] / area)
                    if k < self.nz:
                        hrows.append(face); hcols.append(self._cell_id(i, j, k)); hdata.append(0.5 * self.dz[k] / area)
                    face += 1
        for i in range(self.nx):
            for j in range(self.ny + 1):
                for k in range(self.nz):
                    add(self.edge_maps[0], (i, j, k), 1); add(self.edge_maps[2], (i + 1, j, k), 1)
                    add(self.edge_maps[0], (i, j, k + 1), -1); add(self.edge_maps[2], (i, j, k), -1)
                    area = self.dx[i] * self.dz[k]
                    if j > 0:
                        hrows.append(face); hcols.append(self._cell_id(i, j - 1, k)); hdata.append(0.5 * self.dy[j - 1] / area)
                    if j < self.ny:
                        hrows.append(face); hcols.append(self._cell_id(i, j, k)); hdata.append(0.5 * self.dy[j] / area)
                    face += 1
        for i in range(self.nx + 1):
            for j in range(self.ny):
                for k in range(self.nz):
                    add(self.edge_maps[1], (i, j, k), 1); add(self.edge_maps[2], (i, j + 1, k), 1)
                    add(self.edge_maps[1], (i, j, k + 1), -1); add(self.edge_maps[2], (i, j, k), -1)
                    area = self.dy[j] * self.dz[k]
                    if i > 0:
                        hrows.append(face); hcols.append(self._cell_id(i - 1, j, k)); hdata.append(0.5 * self.dx[i - 1] / area)
                    if i < self.nx:
                        hrows.append(face); hcols.append(self._cell_id(i, j, k)); hdata.append(0.5 * self.dx[i] / area)
                    face += 1
        self.curl = sp.csr_matrix((vals, (rows, cols)), shape=(face, self.n_edges))
        self.face_cell_hodge = sp.csr_matrix((hdata, (hrows, hcols)), shape=(face, self.n_cells))

    def _build_reconstruction(self):
        mats = []
        for axis in range(3):
            rows, cols, data = [], [], []
            emap = self.edge_maps[axis]
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        c = self._cell_id(i, j, k)
                        if axis == 0:
                            keys = ((i, j, k), (i, j + 1, k), (i, j, k + 1), (i, j + 1, k + 1))
                        elif axis == 1:
                            keys = ((i, j, k), (i + 1, j, k), (i, j, k + 1), (i + 1, j, k + 1))
                        else:
                            keys = ((i, j, k), (i + 1, j, k), (i, j + 1, k), (i + 1, j + 1, k))
                        found = [emap[q] for q in keys if q in emap]
                        if found:
                            length = (self.dx[i], self.dy[j], self.dz[k])[axis]
                            for e in found:
                                rows.append(c); cols.append(e); data.append(1.0 / (len(found) * length))
            mats.append(sp.csr_matrix((data, (rows, cols)), shape=(self.n_cells, self.n_edges)))
        self.reconstruct = tuple(mats)

    def _require_inside(self, points, label):
        p = np.asarray(points, float)
        lo = np.array([self.x[0], self.y[0], self.z[0]])
        hi = np.array([self.x[-1], self.y[-1], self.z[-1]])
        if p.ndim != 2 or p.shape[1] != 3 or np.any(~np.isfinite(p)) or np.any(p <= lo) or np.any(p >= hi):
            raise ValueError(f"{label} lies outside the fixed physical background; enlarge BACKGROUND['bounds']")

    def _geometry(self, geometry):
        g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
        if g.n_ports != len(self.coil_materials):
            raise ValueError("geometry port count differs from configured materials")
        return g

    @staticmethod
    def _packages_overlap(a, b):
        """Exact OBB overlap test using the standard 15 separating axes."""
        A = np.asarray(a.pose.rotation, float)
        B = np.asarray(b.pose.rotation, float)
        R = A.T @ B
        t = A.T @ (np.asarray(b.pose.translation) - np.asarray(a.pose.translation))
        absR = np.abs(R) + 1e-14
        ea = np.asarray(a.half_extent, float)
        eb = np.asarray(b.half_extent, float)
        for i in range(3):
            if abs(t[i]) > ea[i] + np.dot(eb, absR[i, :]):
                return False
        for j in range(3):
            if abs(np.dot(t, R[:, j])) > eb[j] + np.dot(ea, absR[:, j]):
                return False
        for i in range(3):
            for j in range(3):
                ra = ea[(i + 1) % 3] * absR[(i + 2) % 3, j] + ea[(i + 2) % 3] * absR[(i + 1) % 3, j]
                rb = eb[(j + 1) % 3] * absR[i, (j + 2) % 3] + eb[(j + 2) % 3] * absR[i, (j + 1) % 3]
                value = abs(t[(i + 2) % 3] * R[(i + 1) % 3, j] - t[(i + 1) % 3] * R[(i + 2) % 3, j])
                if value > ra + rb:
                    return False
        return True

    def validate_geometry(self, geometry):
        g = self._geometry(geometry)
        spacing = 0.45 * min(np.min(self.dx), np.min(self.dy), np.min(self.dz))
        for index, (coil, package) in enumerate(zip(g.coils, g.packages)):
            points = coil.centerline(spacing)
            self._require_inside(points, f"coil {index} centerline")
            # Conservative cross-section margin around the centerline.  This is
            # deliberately simple and prevents sampled coils from leaving their
            # package without introducing a CAD boolean dependency.
            radius = 0.5 * np.hypot(coil.conductor_width, coil.conductor_thickness)
            local = np.abs(package.pose.inverse(points))
            if np.any(local + radius > package.half_extent + 1e-12):
                raise ValueError(f"coil {index} is not fully contained in its package")
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
        for index, package in enumerate(g.packages):
            self._require_inside(package.pose.apply(signs * package.half_extent), f"package {index}")
        for i in range(len(g.packages)):
            for j in range(i + 1, len(g.packages)):
                if self._packages_overlap(g.packages[i], g.packages[j]):
                    raise ValueError(f"packages {i} and {j} overlap")
        return g

    def _package_fraction(self, package):
        """Three-point tensor Gauss sub-cell quadrature for package occupancy."""
        nodes, weights = np.polynomial.legendre.leggauss(3)
        fraction = np.zeros(self.n_cells)
        normalization = 8.0  # tensor-product weights integrate 1 over [-1,1]^3
        for ix, wx in zip(nodes, weights):
            for iy, wy in zip(nodes, weights):
                for iz, wz in zip(nodes, weights):
                    offset = 0.5 * self.cell_widths * np.array([ix, iy, iz])
                    fraction += wx * wy * wz * package.contains(self.cell_centers + offset)
        return np.clip(fraction / normalization, 0.0, 1.0)

    @staticmethod
    def _linear_stencil(grid, value):
        grid = np.asarray(grid, float)
        x = float(value)
        if x <= grid[0]:
            return ((0, 1.0),)
        if x >= grid[-1]:
            return ((len(grid) - 1, 1.0),)
        hi = int(np.searchsorted(grid, x))
        lo = hi - 1
        t = (x - grid[lo]) / (grid[hi] - grid[lo])
        return ((lo, 1.0 - t), (hi, t))

    def _cell_stencil(self, point):
        stencils = [self._linear_stencil(grid, point[k]) for k, grid in enumerate(self.cell_axes)]
        out = []
        for (i, wi), (j, wj), (k, wk) in itertools.product(*stencils):
            out.append((self._cell_id(i, j, k), wi * wj * wk))
        return out

    def _edge_stencil(self, axis, point):
        xc, yc, zc = self.cell_axes
        grids = (
            (xc, self.y, self.z),
            (self.x, yc, self.z),
            (self.x, self.y, zc),
        )[axis]
        stencils = [self._linear_stencil(grid, point[k]) for k, grid in enumerate(grids)]
        weighted = []
        for a, b, c in itertools.product(*stencils):
            key = (a[0], b[0], c[0])
            e = self.edge_maps[axis].get(key)
            if e is not None:
                weighted.append((e, a[1] * b[1] * c[1]))
        total = sum(w for _, w in weighted)
        if total <= np.finfo(float).tiny:
            return []
        return [(e, w / total) for e, w in weighted]

    def _deposit_line(self, points):
        """Conservative piecewise-linear cloud-in-cell line deposition."""
        self._require_inside(points, "coil centerline")
        source = np.zeros(self.n_edges, float)
        heat = np.zeros(self.n_cells, float)
        for p0, p1 in zip(points[:-1], points[1:]):
            d = p1 - p0
            length = float(np.linalg.norm(d))
            if length == 0:
                continue
            mid = 0.5 * (p0 + p1)
            for cell, weight in self._cell_stencil(mid):
                heat[cell] += length * weight
            for axis, component in enumerate(d):
                if component == 0:
                    continue
                stencil = self._edge_stencil(axis, mid)
                if not stencil:
                    continue
                for e, weight in stencil:
                    source[e] += component * weight / self.edge_lengths[e]
        if heat.sum() <= 0 or np.linalg.norm(source) == 0:
            raise ValueError("coil deposition produced a zero physical source")
        heat /= heat.sum()
        return source, heat

    def geometry_context(self, geometry, *, assemble_thermal=True):
        g = self.validate_geometry(geometry)
        spacing = 0.45 * min(np.min(self.dx), np.min(self.dy), np.min(self.dz))
        fractions = {
            name: np.zeros(self.n_cells)
            for name in set(self.coil_materials + self.package_materials + (self.seawater_material,))
        }
        sources, heat_weights = [], []
        for coil, material in zip(g.coils, self.coil_materials):
            points = coil.centerline(spacing)
            source, heat = self._deposit_line(points)
            sources.append(source)
            heat_weights.append(heat)
            area = coil.conductor_width * coil.conductor_thickness
            length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            fractions[material] += heat * (area * length) / self.cell_volumes
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
        if np.max(np.abs(sum(fractions.values()) - 1.0)) > 1e-10:
            raise FloatingPointError("material fractions do not close to unity")
        source_shape = np.column_stack(sources)
        if not assemble_thermal:
            return BackgroundContext(g, fractions, source_shape, tuple(heat_weights))
        if self.thermal_basis is None:
            raise ValueError("thermal basis has not been constructed")
        M, K = self.thermal_operator_full(fractions)
        Phi = self.thermal_basis
        Mr = Phi.T @ (M @ Phi)
        Kr = Phi.T @ (K @ Phi)
        return BackgroundContext(g, fractions, source_shape, tuple(heat_weights), M, K, Mr, Kr)

    def _temperature_material(self, material, temperature):
        m = self.materials[material]
        sigma = float(m.get("electrical_conductivity", 0))
        alpha = float(m.get("resistivity_temperature_coefficient", 0))
        tref = float(m.get("reference_temperature", self.ambient_temperature))
        denominator = 1 + alpha * (np.asarray(temperature, float) - tref)
        if np.any(denominator <= 0):
            raise ValueError("temperature-dependent conductivity left its physical constitutive range")
        return sigma / denominator

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
                elif self.thermal_basis is not None and value.shape == (self.thermal_rank,):
                    rise = self.thermal_basis @ value
                else:
                    raise ValueError("thermal state dimension mismatch")
            if np.any(~np.isfinite(rise)):
                raise ValueError("thermal state must be finite")
        return self.ambient_temperature + rise, material_rise

    def cell_properties(self, context, state=None, *, em=False):
        T, material_rise = self._state_temperature(context, state)
        sigma = np.zeros(self.n_cells); eps = np.zeros(self.n_cells); mu_inv = np.zeros(self.n_cells)
        k = np.zeros(self.n_cells); cap = np.zeros(self.n_cells)
        for name, fraction in context.fractions.items():
            m = self.materials[name]
            local_temperature = (
                self.ambient_temperature + material_rise.get(name, 0.0)
                if material_rise is not None else T
            )
            local_sigma = self._temperature_material(name, local_temperature)
            if em and name in self.coil_materials:
                local_sigma = 0.0
            sigma += fraction * local_sigma
            eps += fraction * EPS0 * float(m.get("relative_permittivity", 1.0))
            mu_inv += fraction / (MU0 * float(m.get("relative_permeability", 1.0)))
            k += fraction * float(m.get("thermal_conductivity", 0))
            cap += fraction * float(m.get("volumetric_heat_capacity", 0))
        return sigma, eps, mu_inv, k, cap, T

    def em_operator(self, context, state=None):
        sigma, eps, mu_inv, _, _, _ = self.cell_properties(context, state, em=True)
        h2 = np.asarray(self.face_cell_hodge @ mu_inv).ravel()
        hs = np.asarray(self.edge_cell_hodge @ sigma).ravel()
        he = np.asarray(self.edge_cell_hodge @ eps).ravel()
        return (
            self.curl.T @ sp.diags(h2) @ self.curl
            - self.omega**2 * sp.diags(he)
            + 1j * self.omega * sp.diags(hs)
        ).tocsr()

    def rhs_matrix(self, context):
        return (-1j * self.omega) * np.asarray(context.source_shape, complex)

    def thermal_operator_full(self, fractions):
        k = np.zeros(self.n_cells); cap = np.zeros(self.n_cells)
        for name, fraction in fractions.items():
            m = self.materials[name]
            k += fraction * float(m.get("thermal_conductivity", 0))
            cap += fraction * float(m.get("volumetric_heat_capacity", 0))
        if np.any(k <= 0) or np.any(cap <= 0):
            raise ValueError("thermal material properties must be positive")
        M = sp.diags(cap * self.cell_volumes, format="csr")
        rows, cols, data = [], [], []
        diag = np.zeros(self.n_cells)

        def pair(c1, c2, area, distance):
            kf = 2 * k[c1] * k[c2] / (k[c1] + k[c2])
            conductance = kf * area / distance
            diag[c1] += conductance; diag[c2] += conductance
            rows.extend([c1, c2]); cols.extend([c2, c1]); data.extend([-conductance, -conductance])

        for i in range(self.nx):
            for j in range(self.ny):
                for kk in range(self.nz):
                    c = self._cell_id(i, j, kk)
                    if i + 1 < self.nx:
                        pair(c, self._cell_id(i + 1, j, kk), self.dy[j] * self.dz[kk], 0.5 * (self.dx[i] + self.dx[i + 1]))
                    if j + 1 < self.ny:
                        pair(c, self._cell_id(i, j + 1, kk), self.dx[i] * self.dz[kk], 0.5 * (self.dy[j] + self.dy[j + 1]))
                    if kk + 1 < self.nz:
                        pair(c, self._cell_id(i, j, kk + 1), self.dx[i] * self.dy[j], 0.5 * (self.dz[kk] + self.dz[kk + 1]))
                    if i in (0, self.nx - 1):
                        diag[c] += k[c] * self.dy[j] * self.dz[kk] / (0.5 * self.dx[i])
                    if j in (0, self.ny - 1):
                        diag[c] += k[c] * self.dx[i] * self.dz[kk] / (0.5 * self.dy[j])
                    if kk in (0, self.nz - 1):
                        diag[c] += k[c] * self.dx[i] * self.dy[j] / (0.5 * self.dz[kk])
        K = sp.csr_matrix((data, (rows, cols)), shape=(self.n_cells, self.n_cells)) + sp.diags(diag)
        return M, K

    def field_components(self, X):
        return tuple(R @ X for R in self.reconstruct)

    def material_joule_cells(self, context, state, X):
        """Legacy visualization reconstruction; production Joule uses edge Hodge tensors."""
        sigma, _, _, _, _, _ = self.cell_properties(context, state, em=True)
        ex, ey, ez = self.field_components(X)
        return ex, ey, ez, 0.5 * sigma * self.cell_volumes

    def wire_resistances(self, context, state=None):
        _, material_rise = self._state_temperature(context, state)
        _, _, _, _, _, T = self.cell_properties(context, state)
        out = []
        spacing = 0.45 * min(np.min(self.dx), np.min(self.dy), np.min(self.dz))
        for coil, material, weights in zip(
            context.geometry.coils, self.coil_materials, context.line_heat_weights
        ):
            temp = (
                self.ambient_temperature + material_rise.get(material, 0.0)
                if material_rise is not None else float(np.dot(weights, T))
            )
            sigma = float(self._temperature_material(material, np.array(temp)))
            length = coil.length(spacing)
            area = coil.conductor_width * coil.conductor_thickness
            mu = MU0 * float(self.materials[material].get("relative_permeability", 1))
            delta = np.sqrt(2 / (self.omega * mu * max(sigma, np.finfo(float).tiny)))
            effective = min(area, 2 * (coil.conductor_width + coil.conductor_thickness) * delta)
            out.append(length / (max(sigma, np.finfo(float).tiny) * max(effective, np.finfo(float).tiny)))
        return np.asarray(out)

    def save_arrays(self):
        if self.thermal_basis is None:
            raise ValueError("thermal basis has not been constructed")
        return {
            "background_x": self.x,
            "background_y": self.y,
            "background_z": self.z,
            "thermal_basis": self.thermal_basis,
        }


__all__ = ["BackgroundContext", "FixedMultiscaleBackground", "stretched_axis"]
