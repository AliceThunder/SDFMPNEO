from __future__ import annotations

import numpy as np

from .hybrid_domain import package_domain_topology
from .scene import Scene, TensorElectricMaterial


def _material_tensor(
    material,
    frequency_hz: float,
    *,
    conduction_dc: bool,
    rotation=None,
) -> np.ndarray:
    if isinstance(material, TensorElectricMaterial):
        coefficient = material.electric_coefficient_tensor(
            frequency_hz,
            conduction_dc=conduction_dc,
        )
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            coefficient = rotation @ coefficient @ rotation.T
        return np.asarray(coefficient, dtype=complex)

    if conduction_dc:
        scalar = float(material.loss_conductivity(0.0))
    else:
        scalar = complex(material.complex_permittivity(frequency_hz))
    return scalar * np.eye(3, dtype=complex)


def _conductivity_tensor(material, *, rotation=None) -> np.ndarray:
    if isinstance(material, TensorElectricMaterial):
        tensor = np.asarray(material.conductivity_tensor, dtype=float)
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            tensor = rotation @ tensor @ rotation.T
        return tensor
    return float(material.loss_conductivity(0.0)) * np.eye(3, dtype=float)


def _tensor_close(first, second) -> bool:
    first = np.asarray(first, dtype=complex)
    second = np.asarray(second, dtype=complex)
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        1e-30,
    )
    return bool(np.allclose(first, second, rtol=1e-11, atol=1e-13 * scale))


def _coefficient_metric(coefficient):
    coefficient = np.asarray(coefficient, dtype=complex)
    if (
        coefficient.shape != (3, 3)
        or np.any(~np.isfinite(coefficient))
        or not np.allclose(
            coefficient,
            coefficient.T,
            rtol=1e-11,
            atol=1e-13,
        )
    ):
        raise ValueError("electric coefficient tensor must be finite symmetric 3x3")

    real = np.real(coefficient)
    imaginary = np.imag(coefficient)
    eigenvalues_real, rotation = np.linalg.eigh(real)
    scale = max(float(np.max(np.abs(eigenvalues_real))), 1e-30)

    # Resolve arbitrary eigenvectors inside repeated-real-eigenvalue subspaces
    # with the imaginary part. TensorElectricMaterial requires the two parts to
    # be coaxial, so the resulting basis diagonalizes the full coefficient.
    start = 0
    while start < 3:
        stop = start + 1
        while (
            stop < 3
            and abs(eigenvalues_real[stop] - eigenvalues_real[start])
            <= 1e-10 * scale
        ):
            stop += 1
        if stop - start > 1:
            block = rotation[:, start:stop]
            imaginary_block = block.T @ imaginary @ block
            _, local_rotation = np.linalg.eigh(imaginary_block)
            rotation[:, start:stop] = block @ local_rotation
        start = stop

    real_local = rotation.T @ real @ rotation
    imaginary_local = rotation.T @ imaginary @ rotation
    off_diagonal = imaginary_local - np.diag(np.diag(imaginary_local))
    if np.linalg.norm(off_diagonal) > 1e-10 * max(
        np.linalg.norm(imaginary_local),
        1.0,
    ):
        raise ValueError(
            "complex electric coefficient must have coaxial real and imaginary tensor parts"
        )

    eigenvalues = np.diag(real_local).astype(complex) + 1j * np.diag(imaginary_local)
    if np.any(np.abs(eigenvalues) <= 1e-30):
        raise ValueError("electric coefficient tensor must be nonsingular")

    roots = np.sqrt(eigenvalues)
    roots = np.where(np.real(roots) < 0.0, -roots, roots)
    sqrt_det = np.prod(roots)
    inverse = rotation @ np.diag(1.0 / eigenvalues) @ rotation.T
    geometric_mean = np.exp((2.0 / 3.0) * np.sum(np.log(roots)))
    return inverse, sqrt_det, geometric_mean


def _laplace_kernel(
    targets,
    sources,
    coefficient,
    *,
    source_radius=None,
):
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)
    difference = targets[:, None, :] - sources[None, :, :]
    inverse, sqrt_det, geometric_mean = _coefficient_metric(coefficient)
    metric_squared = np.einsum(
        "qsi,ij,qsj->qs",
        difference,
        inverse,
        difference,
    )
    if source_radius is not None:
        radius = np.asarray(source_radius, dtype=float)
        if radius.shape != (len(sources),):
            raise ValueError("source_radius has incompatible shape")
        metric_squared = metric_squared + radius[None, :] ** 2 / geometric_mean

    distance = np.sqrt(metric_squared)
    distance = np.where(np.real(distance) < 0.0, -distance, distance)
    if np.any(np.abs(distance) <= 1e-20):
        raise ValueError(
            "electric Green kernel encountered a coincident unregularized source/target pair"
        )
    value = 1.0 / (4.0 * np.pi * sqrt_det * distance)
    return value, difference, distance, inverse


def _node_potential_kernel(
    targets,
    target_radii,
    sources,
    source_radii,
    coefficient,
):
    """Galerkin-style node potential with symmetric self regularization.

    Conductor charge nodes are both source and testing functions.  Their finite
    radius therefore belongs only to coincident/self interactions in the nodal
    potential matrix.  Applying a source-only Plummer radius to every off-
    diagonal interaction makes ``G_ij`` depend on radius ``j`` while ``G_ji``
    depends on radius ``i`` and destroys reciprocity whenever node radii differ.
    Interface incident fields still use ``_laplace_kernel(..., source_radius=)``
    because those targets are physical interface points rather than charge-node
    testing functions.
    """
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)
    target_radii = np.asarray(target_radii, dtype=float)
    source_radii = np.asarray(source_radii, dtype=float)
    if target_radii.shape != (len(targets),):
        raise ValueError("target_radii has incompatible shape")
    if source_radii.shape != (len(sources),):
        raise ValueError("source_radii has incompatible shape")
    if np.any(target_radii <= 0.0) or np.any(source_radii <= 0.0):
        raise ValueError("node radii must be positive")

    difference = targets[:, None, :] - sources[None, :, :]
    inverse, sqrt_det, geometric_mean = _coefficient_metric(coefficient)
    metric_squared = np.einsum(
        "qsi,ij,qsj->qs",
        difference,
        inverse,
        difference,
    )
    coincident = np.all(difference == 0.0, axis=2)
    if np.any(coincident):
        pair_radius = np.sqrt(
            target_radii[:, None] * source_radii[None, :]
        )
        metric_squared = np.where(
            coincident,
            pair_radius**2 / geometric_mean,
            metric_squared,
        )

    distance = np.sqrt(metric_squared)
    distance = np.where(np.real(distance) < 0.0, -distance, distance)
    if np.any(np.abs(distance) <= 1e-20):
        raise ValueError(
            "electric Green kernel encountered a coincident unregularized source/target pair"
        )
    return 1.0 / (4.0 * np.pi * sqrt_det * distance)


def _normal_flux_kernel(
    targets,
    normals,
    sources,
    coefficient,
    *,
    source_radius=None,
):
    value, difference, distance, _ = _laplace_kernel(
        targets,
        sources,
        coefficient,
        source_radius=source_radius,
    )
    normals = np.asarray(normals, dtype=float)
    projection = np.einsum("qsd,qd->qs", difference, normals)
    return -value * projection / (distance**2)


def _electric_field_kernel(
    targets,
    sources,
    coefficient,
    *,
    source_radius=None,
):
    """Return Green electric-field columns as ``(query, xyz, source)``.

    The scalar kernels naturally build ``(query, source, xyz)`` vectors.  All
    transmission operators, however, contract the last axis with source-state
    coefficients and therefore use ``(query, xyz, source)``.  Keep that layout
    explicit here so every caller shares one contract.
    """
    value, difference, distance, inverse = _laplace_kernel(
        targets,
        sources,
        coefficient,
        source_radius=source_radius,
    )
    metric_vector = np.einsum(
        "ij,qsj->qis",
        inverse,
        difference,
    )
    return (
        value[:, None, :]
        * metric_vector
        / (distance[:, None, :] ** 2)
    )


def _equilibrated_lstsq(matrix, rhs, *, rcond: float):
    matrix = np.asarray(matrix, dtype=complex)
    rhs = np.asarray(rhs, dtype=complex)
    row_scale = np.maximum(np.max(np.abs(matrix), axis=1), 1e-30)
    scaled_matrix = matrix / row_scale[:, None]
    scaled_rhs = rhs / row_scale[:, None]
    column_scale = np.maximum(np.max(np.abs(scaled_matrix), axis=0), 1e-30)
    equilibrated = scaled_matrix / column_scale[None, :]
    solution_scaled, _, rank, singular = np.linalg.lstsq(
        equilibrated,
        scaled_rhs,
        rcond=rcond,
    )
    solution = solution_scaled / column_scale[:, None]
    residual_matrix = matrix @ solution - rhs
    residual = float(
        np.linalg.norm(residual_matrix)
        / max(np.linalg.norm(rhs), 1.0)
    )
    condition = (
        float(singular[0] / max(singular[-1], 1e-30))
        if len(singular)
        else 1.0
    )
    return solution, residual, int(rank), condition


class PreparedTensorElectricTransmission:
    """Tensor electric scalar-potential transmission for package regions."""

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        source_positions,
        source_radii,
        *,
        conduction_dc: bool,
        surface_vertical_order: int = 12,
        surface_azimuthal_order: int = 24,
        mfs_offset_fraction: float = 0.12,
        interface_residual_tolerance: float = 2e-5,
        svd_rcond: float = 1e-11,
        maximum_raw_reciprocity_defect: float = 0.10,
    ):
        self.scene = scene
        self.frequency_hz = float(frequency_hz)
        self.conduction_dc = bool(conduction_dc)
        self.source_positions = np.asarray(source_positions, dtype=float)
        self.source_radii = np.asarray(source_radii, dtype=float)
        self.interface_residual_tolerance = float(interface_residual_tolerance)
        self.svd_rcond = float(svd_rcond)
        self.maximum_raw_reciprocity_defect = float(
            maximum_raw_reciprocity_defect
        )
        if (
            self.source_positions.ndim != 2
            or self.source_positions.shape[1] != 3
            or self.source_radii.shape != (len(self.source_positions),)
            or np.any(self.source_radii <= 0.0)
        ):
            raise ValueError("invalid tensor-electric source node geometry")
        if (
            not 0.0 < mfs_offset_fraction < 0.5
            or self.interface_residual_tolerance <= 0.0
            or self.svd_rcond <= 0.0
            or self.maximum_raw_reciprocity_defect <= 0.0
        ):
            raise ValueError("invalid tensor-electric MFS configuration")

        self.background_coefficient = _material_tensor(
            scene.medium,
            self.frequency_hz,
            conduction_dc=self.conduction_dc,
        )
        all_topology = package_domain_topology(scene.packages)
        resolved = []
        active = []
        order = sorted(
            range(len(scene.packages)),
            key=lambda index: all_topology.depth[index],
        )
        for index in order:
            package = scene.packages[index]
            parent = all_topology.parent[index]
            outside = (
                self.background_coefficient
                if parent is None
                else resolved[parent]
            )
            coefficient = _material_tensor(
                package.material,
                self.frequency_hz,
                conduction_dc=self.conduction_dc,
                rotation=package.geometry.pose.rotation,
            )
            while len(resolved) <= index:
                resolved.append(None)
            resolved[index] = coefficient
            if not _tensor_close(coefficient, outside):
                active.append((index, package.geometry, coefficient))

        if self.conduction_dc:
            tensors = [self.background_coefficient] + [item[2] for item in active]
            for coefficient in tensors:
                eigenvalues = np.linalg.eigvalsh(
                    np.real_if_close(coefficient).astype(float)
                )
                if np.min(eigenvalues) <= 1e-14:
                    raise NotImplementedError(
                        "tensor-electric exact DC conduction currently requires "
                        "positive-definite conductivity in every active material "
                        "region; mixed perfect-insulator tensor regions still use "
                        "the scalar partial-DC path"
                    )

        regions = []
        start = 0
        for package_index, geometry, coefficient in active:
            surface = geometry.surface_quadrature(
                vertical_order=surface_vertical_order,
                azimuthal_order=surface_azimuthal_order,
            )
            offset = float(mfs_offset_fraction) * float(
                np.min(geometry.half_extents)
            )
            for _ in range(12):
                exterior_mfs = surface.positions - offset * surface.normals
                interior_mfs = surface.positions + offset * surface.normals
                if (
                    np.all(
                        geometry.contains(exterior_mfs, tolerance=1e-12)
                    )
                    and not np.any(
                        geometry.contains(interior_mfs, tolerance=1e-12)
                    )
                ):
                    break
                offset *= 0.5
            else:
                raise RuntimeError("failed to place tensor-electric MFS sources")

            count = len(surface.positions)
            regions.append(
                {
                    "package_index": int(package_index),
                    "geometry": geometry,
                    "coefficient": coefficient,
                    "positions": surface.positions,
                    "normals": surface.normals,
                    "exterior_mfs": exterior_mfs,
                    "interior_mfs": interior_mfs,
                    "slice": slice(start, start + count),
                    "offset": float(offset),
                }
            )
            start += count

        geometries = tuple(region["geometry"] for region in regions)
        self.topology = package_domain_topology(geometries)
        children = [[] for _ in regions]
        for index, parent in enumerate(self.topology.parent):
            if parent is not None:
                children[parent].append(index)

        for index, region in enumerate(regions):
            parent = self.topology.parent[index]
            region["parent"] = parent
            region["children"] = tuple(children[index])
            if parent is None:
                continue

            parent_geometry = regions[parent]["geometry"]
            offset = float(region["offset"])
            for _ in range(12):
                exterior_mfs = region["positions"] - offset * region["normals"]
                interior_mfs = region["positions"] + offset * region["normals"]
                if (
                    np.all(
                        region["geometry"].contains(
                            exterior_mfs,
                            tolerance=1e-12,
                        )
                    )
                    and not np.any(
                        region["geometry"].contains(
                            interior_mfs,
                            tolerance=1e-12,
                        )
                    )
                    and np.all(
                        parent_geometry.contains(
                            interior_mfs,
                            tolerance=1e-12,
                        )
                    )
                ):
                    region["exterior_mfs"] = exterior_mfs
                    region["interior_mfs"] = interior_mfs
                    break
                offset *= 0.5
            else:
                raise RuntimeError(
                    "failed to place nested tensor-electric MFS sources inside "
                    "the parent material shell"
                )

        self.regions = tuple(regions)
        self.active_geometries = tuple(
            region["geometry"] for region in self.regions
        )
        self.source_region = self._active_region(self.source_positions)
        self.source_package_region = np.full(
            len(self.source_positions),
            -1,
            dtype=int,
        )
        for active_index, region in enumerate(self.regions):
            self.source_package_region[
                self.source_region == active_index
            ] = int(region["package_index"])

        self._coefficients = None
        self.interface_residual = 0.0
        self.interface_condition = 1.0
        self.raw_reciprocity_defect = 0.0
        self.raw_reciprocity_target_exceeded = False
        self._solve()

    def _active_region(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if not self.regions:
            return np.full(
                len(np.atleast_2d(points)),
                -1,
                dtype=int,
            )
        return np.asarray(
            self.topology.deepest_containing(
                self.active_geometries,
                points,
                tolerance=1e-12,
            ),
            dtype=int,
        )

    def _coefficient(self, region_index: int):
        return (
            self.background_coefficient
            if region_index < 0
            else self.regions[region_index]["coefficient"]
        )

    def _particular(
        self,
        points,
        *,
        region_index: int,
        normals=None,
        field: bool = False,
    ):
        mask = self.source_region == region_index
        count = len(self.source_positions)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if field:
            out = np.zeros((len(points), 3, count), dtype=complex)
        else:
            out = np.zeros((len(points), count), dtype=complex)
        flux = (
            None
            if normals is None
            else np.zeros((len(points), count), dtype=complex)
        )
        if not np.any(mask):
            return out, flux

        coefficient = self._coefficient(region_index)
        if field:
            out[:, :, mask] = _electric_field_kernel(
                points,
                self.source_positions[mask],
                coefficient,
                source_radius=self.source_radii[mask],
            )
        else:
            out[:, mask] = _laplace_kernel(
                points,
                self.source_positions[mask],
                coefficient,
                source_radius=self.source_radii[mask],
            )[0]
        if normals is not None:
            flux[:, mask] = _normal_flux_kernel(
                points,
                normals,
                self.source_positions[mask],
                coefficient,
                source_radius=self.source_radii[mask],
            )
        return out, flux

    def _region_basis(
        self,
        points,
        *,
        region_index: int,
        normals=None,
        field: bool = False,
    ):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n = sum(len(region["positions"]) for region in self.regions)
        if field:
            value = np.zeros((len(points), 3, 2 * n), dtype=complex)
        else:
            value = np.zeros((len(points), 2 * n), dtype=complex)
        flux = (
            None
            if normals is None
            else np.zeros((len(points), 2 * n), dtype=complex)
        )
        coefficient = self._coefficient(region_index)
        blocks = []
        if region_index < 0:
            for region in self.regions:
                if region["parent"] is None:
                    blocks.append((region["slice"], region["exterior_mfs"]))
        else:
            region = self.regions[region_index]
            sl = region["slice"]
            blocks.append(
                (
                    slice(n + sl.start, n + sl.stop),
                    region["interior_mfs"],
                )
            )
            for child_index in region["children"]:
                child = self.regions[child_index]
                blocks.append((child["slice"], child["exterior_mfs"]))

        for columns, sources in blocks:
            if field:
                value[:, :, columns] = _electric_field_kernel(
                    points,
                    sources,
                    coefficient,
                )
            else:
                value[:, columns] = _laplace_kernel(
                    points,
                    sources,
                    coefficient,
                )[0]
            if normals is not None:
                flux[:, columns] = _normal_flux_kernel(
                    points,
                    normals,
                    sources,
                    coefficient,
                )
        return value, flux

    def _solve(self):
        if not self.regions:
            self._coefficients = np.zeros(
                (0, len(self.source_positions)),
                dtype=complex,
            )
            return

        surface_positions = np.concatenate(
            [region["positions"] for region in self.regions],
            axis=0,
        )
        n = len(surface_positions)
        matrix = np.zeros((2 * n, 2 * n), dtype=complex)
        rhs_value = np.zeros((n, len(self.source_positions)), dtype=complex)
        rhs_flux = np.zeros_like(rhs_value)

        for region_index, region in enumerate(self.regions):
            sl = region["slice"]
            points = region["positions"]
            normals = region["normals"]
            parent = region["parent"]
            outside_index = -1 if parent is None else parent
            outside_value, outside_flux = self._region_basis(
                points,
                region_index=outside_index,
                normals=normals,
            )
            inside_value, inside_flux = self._region_basis(
                points,
                region_index=region_index,
                normals=normals,
            )
            matrix[sl, :] = outside_value - inside_value
            matrix[n + sl.start : n + sl.stop, :] = outside_flux - inside_flux

            outside_particular, outside_particular_flux = self._particular(
                points,
                region_index=outside_index,
                normals=normals,
            )
            inside_particular, inside_particular_flux = self._particular(
                points,
                region_index=region_index,
                normals=normals,
            )
            rhs_value[sl] = inside_particular - outside_particular
            rhs_flux[sl] = inside_particular_flux - outside_particular_flux

        rhs = np.vstack((rhs_value, rhs_flux))
        coefficients, residual, rank, condition = _equilibrated_lstsq(
            matrix,
            rhs,
            rcond=self.svd_rcond,
        )
        if residual > self.interface_residual_tolerance:
            raise RuntimeError(
                "tensor-electric MFS solve did not meet the declared residual "
                f"tolerance: {residual:.3e} > "
                f"{self.interface_residual_tolerance:.3e}"
            )
        if rank < min(matrix.shape):
            raise RuntimeError("tensor-electric MFS system is rank deficient")
        self._coefficients = coefficients
        self.interface_residual = residual
        self.interface_condition = condition

    def potential_matrix(self) -> np.ndarray:
        count = len(self.source_positions)
        raw = np.zeros((count, count), dtype=complex)
        query_region = self._active_region(self.source_positions)
        for region_index in np.unique(query_region):
            mask = query_region == region_index
            source_mask = self.source_region == region_index
            points = self.source_positions[mask]
            particular = np.zeros((len(points), count), dtype=complex)
            if np.any(source_mask):
                particular[:, source_mask] = _node_potential_kernel(
                    points,
                    self.source_radii[mask],
                    self.source_positions[source_mask],
                    self.source_radii[source_mask],
                    self._coefficient(int(region_index)),
                )
            if self.regions:
                basis, _ = self._region_basis(
                    points,
                    region_index=int(region_index),
                )
                particular = particular + basis @ self._coefficients
            raw[mask, :] = particular

        denominator = max(float(np.linalg.norm(raw)), 1e-30)
        defect = float(np.linalg.norm(raw - raw.T) / denominator)
        if not np.isfinite(defect):
            raise RuntimeError(
                "tensor-electric effective potential produced a non-finite raw "
                "reciprocity diagnostic"
            )
        self.raw_reciprocity_defect = defect
        self.raw_reciprocity_target_exceeded = bool(
            defect > self.maximum_raw_reciprocity_defect
        )

        # Point-collocation MFS is not self-adjoint at finite discretization, so
        # its raw node response need not be reciprocal even when the interface
        # residual and rank diagnostics are healthy.  Reciprocity is a property
        # of the physical electrostatic operator, therefore project the discrete
        # response onto the reciprocal subspace before it enters the KKT system.
        # Keep the raw defect above as a convergence/quality diagnostic instead
        # of rejecting otherwise valid random training scenes.
        return 0.5 * (raw + raw.T)

    def electric_field_transfer(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (3,) or (n,3)")

        query_region = self._active_region(points)
        out = np.zeros(
            (len(points), 3, len(self.source_positions)),
            dtype=complex,
        )
        for region_index in np.unique(query_region):
            mask = query_region == region_index
            local_points = points[mask]
            particular, _ = self._particular(
                local_points,
                region_index=int(region_index),
                field=True,
            )
            if self.regions:
                basis, _ = self._region_basis(
                    local_points,
                    region_index=int(region_index),
                    field=True,
                )
                particular = particular + np.einsum(
                    "qdk,ks->qds",
                    basis,
                    self._coefficients,
                )
            out[mask, :, :] = particular
        return out[0] if scalar else out

    def conductivity_tensor_at(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        all_topology = package_domain_topology(self.scene.packages)
        region = np.asarray(
            all_topology.deepest_containing(
                self.scene.packages,
                points,
                tolerance=1e-12,
            ),
            dtype=int,
        )
        out = np.zeros((len(points), 3, 3), dtype=float)
        background = _conductivity_tensor(self.scene.medium)
        out[:, :, :] = background[None, :, :]
        for index, package in enumerate(self.scene.packages):
            mask = region == index
            if np.any(mask):
                out[mask] = _conductivity_tensor(
                    package.material,
                    rotation=package.geometry.pose.rotation,
                )
        return out[0] if scalar else out
