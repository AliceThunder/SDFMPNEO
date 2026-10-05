"""Geometry-only tensor surrogates, including production cellwise Joule truth."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .unified_geometry import UnifiedUWPTGeometry

_SCHEMA_VERSION = 3
_SUPPORTED_SHAPES = ("circle", "rounded_square")


def _hermitian(value):
    a = np.asarray(value, complex)
    return 0.5 * (a + a.conj().T)


def _psd_clip(value, floor=0.0):
    a = _hermitian(value)
    w, v = np.linalg.eigh(a)
    return _hermitian((v * np.maximum(w.real, float(floor))) @ v.conj().T)


def _upper_pairs(n):
    return [(i, j) for i in range(int(n)) for j in range(i, int(n))]


def pack_complex_symmetric(matrix):
    z = np.asarray(matrix, complex)
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ValueError("complex symmetric matrix must be square")
    z = 0.5 * (z + z.T)
    pairs = _upper_pairs(z.shape[0])
    return np.asarray(
        [z[i, j].real for i, j in pairs] + [z[i, j].imag for i, j in pairs], float
    )


def unpack_complex_symmetric(packed, n):
    n = int(n)
    p = np.asarray(packed, float).reshape(-1)
    pairs = _upper_pairs(n)
    m = len(pairs)
    if p.size != 2 * m:
        raise ValueError("complex symmetric packed size mismatch")
    z = np.zeros((n, n), complex)
    for k, (i, j) in enumerate(pairs):
        value = p[k] + 1j * p[m + k]
        z[i, j] = value
        z[j, i] = value
    return z


def pack_hermitian(matrix):
    h = _hermitian(matrix)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hermitian matrix must be square")
    n = h.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    values = [float(h[i, i].real) for i in range(n)]
    values.extend(float(h[i, j].real) for i, j in pairs)
    values.extend(float(h[i, j].imag) for i, j in pairs)
    return np.asarray(values, float)


def unpack_hermitian(packed, n):
    n = int(n)
    p = np.asarray(packed, float).reshape(-1)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if p.size != n + 2 * len(pairs):
        raise ValueError("Hermitian packed size mismatch")
    h = np.zeros((n, n), complex)
    h[np.diag_indices(n)] = p[:n]
    offset = n
    for k, (i, j) in enumerate(pairs):
        value = p[offset + k] + 1j * p[offset + len(pairs) + k]
        h[i, j] = value
        h[j, i] = value.conjugate()
    return h



def _pack_real_symmetric(matrix):
    a = np.asarray(matrix, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("real symmetric matrix must be square")
    a = 0.5 * (a + a.T)
    return np.asarray(
        [a[i, j] for i, j in _upper_pairs(a.shape[0])],
        float,
    )


def _unpack_real_symmetric(packed, n):
    n = int(n)
    values = np.asarray(packed, float).reshape(-1)
    pairs = _upper_pairs(n)
    if values.size != len(pairs):
        raise ValueError("real symmetric packed size mismatch")
    out = np.zeros((n, n), float)
    for value, (i, j) in zip(values, pairs):
        out[i, j] = value
        out[j, i] = value
    return out


def pack_spatial_global_tensors(z_field, d_vol):
    """Pack the stable passive global representation.

    Instead of regressing Re(Z) and D independently and recovering the tiny
    outward matrix by catastrophic subtraction, regress D, Re(D_out) and Im(Z).
    Reciprocity fixes Im(Z) to a real-symmetric matrix, while
    Im(D_out)=-Im(D).  The packed width is exactly the legacy Z/D width.
    """
    z = 0.5 * (
        np.asarray(z_field, complex)
        + np.asarray(z_field, complex).T
    )
    d = _hermitian(d_vol)
    outward = _hermitian(z) - d
    return np.concatenate(
        [
            pack_hermitian(d),
            _pack_real_symmetric(outward.real),
            _pack_real_symmetric(z.imag),
        ]
    )


def decode_spatial_global_tensors(packed, n_ports):
    raw = np.asarray(packed, float).reshape(-1)
    n = int(n_ports)
    h_size = n * n
    s_size = n * (n + 1) // 2
    if raw.size != h_size + 2 * s_size:
        raise ValueError("spatial global packed size mismatch")

    d_raw = unpack_hermitian(raw[:h_size], n)
    outward_real_raw = _unpack_real_symmetric(
        raw[h_size:h_size + s_size],
        n,
    )
    reactance = _unpack_real_symmetric(
        raw[h_size + s_size:],
        n,
    )

    d = _psd_clip(d_raw)
    # Reciprocity requires Herm(Z) to be real symmetric.  Therefore the
    # imaginary part of physical D_out is exactly -Im(D).
    outward_candidate = _hermitian(
        outward_real_raw.astype(complex)
        - 1j * np.asarray(d.imag, float)
    )
    minimum = float(
        np.min(np.linalg.eigvalsh(outward_candidate)).real
    )
    shift = max(0.0, -minimum)
    outward = _hermitian(
        outward_candidate
        + shift * np.eye(n, dtype=complex)
    )
    resistance = np.asarray((d + outward).real, float)
    z = resistance + 1j * reactance
    implied = _hermitian(z) - d

    corrected = pack_spatial_global_tensors(z, d)
    correction = float(
        np.linalg.norm(corrected - raw)
        / max(
            float(np.linalg.norm(raw)),
            np.finfo(float).tiny,
        )
    )
    return DecodedTensors(
        np.asarray(z, complex),
        np.asarray(d, complex),
        np.empty((0, n, n), complex),
        np.asarray(implied, complex),
        correction,
        0.0,
    )


def tensor_output_dimension(n_ports, thermal_rank):
    n = int(n_ports)
    return n * (n + 1) + n * n * (1 + int(thermal_rank))


def tensor_block_sizes(n_ports, thermal_rank):
    n = int(n_ports)
    return n * (n + 1), n * n, int(thermal_rank)


def pack_tensors(z_field, d_vol, modal_h):
    n = int(np.asarray(d_vol).shape[0])
    modal = np.asarray(modal_h, complex)
    if modal.size == 0:
        modal = np.empty((0, n, n), complex)
    if modal.ndim != 3 or modal.shape[1:] != (n, n):
        raise ValueError(
            "modal Joule tensors must have shape (rank, ports, ports)"
        )
    return np.concatenate(
        [pack_complex_symmetric(z_field), pack_hermitian(d_vol)]
        + [pack_hermitian(h) for h in modal]
    )


def unpack_tensors(packed, n_ports, thermal_rank):
    p = np.asarray(packed, float).reshape(-1)
    n = int(n_ports)
    r = int(thermal_rank)
    z_size, h_size, _ = tensor_block_sizes(n, r)
    expected = z_size + h_size * (1 + r)
    if p.size != expected:
        raise ValueError(f"tensor packed size mismatch: expected {expected}, got {p.size}")
    z = unpack_complex_symmetric(p[:z_size], n)
    d = unpack_hermitian(p[z_size:z_size + h_size], n)
    start = z_size + h_size
    if r:
        modal = np.asarray(
            [
                unpack_hermitian(
                    p[
                        start + j * h_size:
                        start + (j + 1) * h_size
                    ],
                    n,
                )
                for j in range(r)
            ],
            complex,
        )
    else:
        modal = np.empty((0, n, n), complex)
    return z, d, modal


def _pose_features(pose):
    angles = np.asarray(pose.angles, float)
    return (
        np.asarray(pose.translation, float).tolist()
        + np.sin(angles).tolist()
        + np.cos(angles).tolist()
    )


def encode_geometry(geometry):
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    features = []
    for coil in g.coils:
        if coil.shape not in _SUPPORTED_SHAPES:
            raise ValueError(
                f"production tensor surrogate supports {_SUPPORTED_SHAPES}; got {coil.shape!r}"
            )
        features.extend(1.0 if coil.shape == name else 0.0 for name in _SUPPORTED_SHAPES)
        features.extend(
            [
                float(coil.turns),
                float(coil.outer_half_size),
                float(coil.pitch),
                float(coil.conductor_width),
                float(coil.conductor_thickness),
                0.0 if coil.corner_radius is None else float(coil.corner_radius),
            ]
        )
        features.extend(_pose_features(coil.pose))
    for package in g.packages:
        features.extend(np.asarray(package.half_extent, float).tolist())
        features.extend(_pose_features(package.pose))
    out = np.asarray(features, float)
    if np.any(~np.isfinite(out)):
        raise ValueError("geometry encoding contains non-finite values")
    return out




def encode_geometry_invariant(geometry):
    """Rigid-motion-invariant geometry descriptor for surrogate regression."""
    g = (
        geometry
        if isinstance(geometry, UnifiedUWPTGeometry)
        else UnifiedUWPTGeometry.from_mapping(geometry)
    )
    features = []
    for coil, package in zip(g.coils, g.packages):
        if coil.shape not in _SUPPORTED_SHAPES:
            raise ValueError(
                f"unsupported production coil shape {coil.shape!r}"
            )
        features.extend(
            1.0 if coil.shape == name else 0.0
            for name in _SUPPORTED_SHAPES
        )
        features.extend(
            [
                float(coil.turns),
                float(coil.outer_half_size),
                float(coil.pitch),
                float(coil.conductor_width),
                float(coil.conductor_thickness),
                0.0
                if coil.corner_radius is None
                else float(coil.corner_radius),
            ]
        )
        features.extend(
            np.asarray(package.half_extent, float).tolist()
        )
        # Package pose relative to its associated coil is physically relevant,
        # while their common absolute rigid motion is not.
        features.extend(
            np.asarray(
                coil.pose.inverse(package.pose.translation),
                float,
            ).tolist()
        )
        relative_package_rotation = (
            coil.pose.rotation.T @ package.pose.rotation
        )
        features.extend(
            np.asarray(
                relative_package_rotation,
                float,
            ).reshape(-1).tolist()
        )

        # Derived self-geometry scales dramatically reduce the burden on a
        # 96-sample global regressor: resistance/inductive trends are much
        # smoother in length, conductor area and normalized spiral dimensions
        # than in the raw independent parameters alone.
        length = float(coil.length(1.0e-3))
        conductor_area = float(
            coil.conductor_width
            * coil.conductor_thickness
        )
        inner_half = float(
            coil.outer_half_size
            - coil.pitch * coil.turns
        )
        features.extend(
            [
                length,
                conductor_area,
                length
                / max(
                    conductor_area,
                    np.finfo(float).tiny,
                ),
                inner_half,
                float(coil.conductor_width)
                / max(
                    float(coil.pitch),
                    np.finfo(float).tiny,
                ),
                float(coil.conductor_thickness)
                / max(
                    float(coil.conductor_width),
                    np.finfo(float).tiny,
                ),
                float(coil.pitch)
                / max(
                    float(coil.outer_half_size),
                    np.finfo(float).tiny,
                ),
            ]
        )

    for i in range(len(g.coils)):
        for j in range(i + 1, len(g.coils)):
            left = g.coils[i]
            right = g.coils[j]
            features.extend(
                np.asarray(
                    left.pose.inverse(right.pose.translation),
                    float,
                ).tolist()
            )
            relative_rotation = (
                left.pose.rotation.T @ right.pose.rotation
            )
            features.extend(
                np.asarray(
                    relative_rotation,
                    float,
                ).reshape(-1).tolist()
            )
            delta = np.asarray(
                left.pose.inverse(
                    right.pose.translation
                ),
                float,
            )
            distance = float(np.linalg.norm(delta))
            size = np.sqrt(
                max(
                    float(left.outer_half_size)
                    * float(right.outer_half_size),
                    np.finfo(float).tiny,
                )
            )
            lateral = float(
                np.linalg.norm(delta[:2])
            )
            left_normal = np.asarray(
                left.pose.rotation[:, 2],
                float,
            )
            right_normal = np.asarray(
                right.pose.rotation[:, 2],
                float,
            )
            normal_alignment = float(
                np.dot(left_normal, right_normal)
            )
            features.extend(
                [
                    distance,
                    distance / size,
                    lateral / size,
                    float(delta[2]) / size,
                    normal_alignment,
                    abs(normal_alignment),
                    (
                        float(left.outer_half_size)
                        * float(right.outer_half_size)
                    )
                    / max(
                        distance * distance,
                        1e-12,
                    ),
                ]
            )

    out = np.asarray(features, float)
    if np.any(~np.isfinite(out)):
        raise ValueError(
            "invariant geometry encoding contains non-finite values"
        )
    return out


def decode_geometry_encoding(encoded, n_ports):
    """Invert encode_geometry for cached training geometries.

    The encoding stores translations directly and rotations as sin/cos pairs,
    so this inversion is exact up to floating-point roundoff for the supported
    circle/rounded-square production family. It lets the v2 spatial neural
    field train from existing v56 truth caches without regenerating Maxwell truth.
    """
    values = np.asarray(encoded, float).reshape(-1)
    n_ports = int(n_ports)
    per_coil = len(_SUPPORTED_SHAPES) + 6 + 9
    per_package = 3 + 9
    expected = n_ports * (per_coil + per_package)
    if values.size != expected or np.any(~np.isfinite(values)):
        raise ValueError(
            f"geometry encoding width mismatch: expected {expected}, got {values.size}"
        )

    cursor = 0
    coils = []
    for port in range(n_ports):
        one_hot = values[cursor:cursor + len(_SUPPORTED_SHAPES)]
        cursor += len(_SUPPORTED_SHAPES)
        shape_index = int(np.argmax(one_hot))
        if one_hot[shape_index] < 0.5:
            raise ValueError("geometry encoding has no supported coil shape")
        shape = _SUPPORTED_SHAPES[shape_index]
        turns, outer, pitch, width, thickness, corner = values[cursor:cursor + 6]
        cursor += 6
        translation = values[cursor:cursor + 3]
        cursor += 3
        sine = values[cursor:cursor + 3]
        cursor += 3
        cosine = values[cursor:cursor + 3]
        cursor += 3
        angles = np.arctan2(sine, cosine)
        coils.append({
            "name": f"port_{port}",
            "shape": shape,
            "turns": float(turns),
            "outer_half_size": float(outer),
            "pitch": float(pitch),
            "conductor_width": float(width),
            "conductor_thickness": float(thickness),
            "corner_radius": float(corner),
            "translation": translation.tolist(),
            "angles": angles.tolist(),
        })

    packages = []
    for _port in range(n_ports):
        half_extent = values[cursor:cursor + 3]
        cursor += 3
        translation = values[cursor:cursor + 3]
        cursor += 3
        sine = values[cursor:cursor + 3]
        cursor += 3
        cosine = values[cursor:cursor + 3]
        cursor += 3
        angles = np.arctan2(sine, cosine)
        packages.append({
            "half_extent": half_extent.tolist(),
            "translation": translation.tolist(),
            "angles": angles.tolist(),
        })

    return UnifiedUWPTGeometry.from_mapping({
        "coils": coils,
        "packages": packages,
    })


def _modal_project_to_bounds(h, d, lower, upper):
    d = _psd_clip(d)
    wd, ud = np.linalg.eigh(d)
    scale = max(float(np.max(wd)), 1.0)
    keep = wd > 1e-12 * scale
    if not np.any(keep):
        return np.zeros_like(d)
    root = ud[:, keep] * np.sqrt(wd[keep])
    invroot = ud[:, keep] * (1.0 / np.sqrt(wd[keep]))
    reduced = _hermitian(invroot.conj().T @ _hermitian(h) @ invroot)
    wh, vh = np.linalg.eigh(reduced)
    wh = np.clip(wh.real, float(lower), float(upper))
    return _hermitian(root @ ((vh * wh) @ vh.conj().T) @ root.conj().T)


@dataclass(frozen=True)
class DecodedTensors:
    z_field: np.ndarray
    d_vol: np.ndarray
    modal_h: np.ndarray
    implied_d_out: np.ndarray
    zd_projection_correction: float
    h_projection_correction: float

    @property
    def projection_correction(self):
        return max(self.zd_projection_correction, self.h_projection_correction)

    def modal_heat(self, currents):
        c = np.asarray(currents, complex).reshape(-1)
        if c.shape != (self.z_field.shape[0],):
            raise ValueError("current vector has wrong port dimension")
        return 0.5 * np.real(
            np.einsum("p,rpq,q->r", c.conj(), self.modal_h, c, optimize=True)
        )

    def volume_power(self, currents):
        c = np.asarray(currents, complex).reshape(-1)
        return float(0.5 * np.real(c.conj() @ self.d_vol @ c))

    def implied_outward_power(self, currents):
        c = np.asarray(currents, complex).reshape(-1)
        return float(0.5 * np.real(c.conj() @ self.implied_d_out @ c))


def decode_physical_tensors(packed, n_ports, phi_min, phi_max):
    """Decode onto the reciprocal/passive/modal-feasible set.

    For reciprocal Z, Herm(Z)=Re(Z) is real symmetric. After projecting D to
    PSD we keep the imaginary part of D_out exactly equal to -Im(D) and add the
    minimum real diagonal shift needed to make D_out PSD. Therefore both
    reciprocity and Herm(Z)-D >= 0 hold simultaneously.
    """
    phi_min = np.asarray(phi_min, float).reshape(-1)
    phi_max = np.asarray(phi_max, float).reshape(-1)
    if phi_min.shape != phi_max.shape or np.any(phi_max < phi_min):
        raise ValueError("invalid geometry-dependent thermal-mode bounds")
    raw = np.asarray(packed, float).reshape(-1)
    z_raw, d_raw, h_raw = unpack_tensors(raw, n_ports, len(phi_min))

    d = _psd_clip(d_raw)
    r0 = 0.5 * (np.asarray(z_raw.real, float) + np.asarray(z_raw.real, float).T)
    d_out_candidate = _hermitian(r0.astype(complex) - d)
    minimum = float(np.min(np.linalg.eigvalsh(d_out_candidate)).real)
    shift = max(0.0, -minimum)
    d_out = _hermitian(d_out_candidate + shift * np.eye(int(n_ports)))
    r = np.asarray((d + d_out).real, float)
    x = 0.5 * (np.asarray(z_raw.imag, float) + np.asarray(z_raw.imag, float).T)
    z = r + 1j * x
    implied = _hermitian(z) - d

    if len(phi_min):
        modal = np.asarray(
            [
                _modal_project_to_bounds(
                    h_raw[j],
                    d,
                    phi_min[j],
                    phi_max[j],
                )
                for j in range(len(phi_min))
            ],
            complex,
        )
    else:
        modal = np.empty(
            (0, int(n_ports), int(n_ports)),
            complex,
        )
    n = int(n_ports)
    z_size, h_size, r_count = tensor_block_sizes(n, len(phi_min))
    corrected = pack_tensors(z, d, modal)
    zd_stop = z_size + h_size
    zd_correction = float(
        np.linalg.norm(corrected[:zd_stop] - raw[:zd_stop])
        / max(np.linalg.norm(raw[:zd_stop]), np.finfo(float).tiny)
    )
    h_correction = (
        float(
            np.linalg.norm(corrected[zd_stop:] - raw[zd_stop:])
            / max(np.linalg.norm(raw[zd_stop:]), np.finfo(float).tiny)
        )
        if r_count
        else 0.0
    )
    return DecodedTensors(z, d, modal, implied, zd_correction, h_correction)


def _edge_loss_weights(background, context):
    sigma, _, _, _, _, _ = background.cell_properties(context, None, em=True)
    sigma = np.asarray(sigma, float)
    return sigma, np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)


def _conductivity_support_bounds(phi, sigma):
    phi = np.asarray(phi, float)
    sigma = np.asarray(sigma, float).reshape(-1)
    threshold = max(float(np.max(sigma)), 1.0) * 1e-14
    support = sigma > threshold
    if not np.any(support):
        raise RuntimeError("physical conductivity-loss support is empty")
    return np.min(phi[support], axis=0), np.max(phi[support], axis=0)


def _audit_current_vectors(n_ports):
    n = int(n_ports)
    eye = np.eye(n, dtype=complex)
    vectors = [eye[:, p] for p in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            vectors.append(eye[:, i] + eye[:, j])
            vectors.append(eye[:, i] + 1j * eye[:, j])
    if len(vectors) != n * n:
        raise AssertionError("Hermitian current audit span dimension mismatch")
    return vectors


def _cell_volume_heat(background, sigma, field):
    edge_energy = np.abs(np.asarray(field, complex).reshape(-1)) ** 2
    cell_edge_energy = np.asarray(background.edge_cell_hodge.T @ edge_energy).reshape(-1)
    return np.asarray((0.5 * np.asarray(sigma, float) * cell_edge_energy).real, float)


def _require_static_field_materials(background):
    for name, material in background.materials.items():
        if name in background.coil_materials:
            continue
        if (
            float(material.get("electrical_conductivity", 0.0)) > 0.0
            and float(material.get("resistivity_temperature_coefficient", 0.0)) != 0.0
        ):
            raise ValueError(
                "geometry-only field tensors require temperature-independent non-wire EM "
                f"materials; {name!r} is temperature dependent"
            )


def _solve_port_fields(background, context):
    A = background.em_operator(context, None)
    B = background.rhs_matrix(context)
    try:
        lu = spla.splu(A.tocsc())
        X = np.column_stack([lu.solve(B[:, p]) for p in range(B.shape[1])])
    except RuntimeError:
        X = np.column_stack([spla.spsolve(A, B[:, p]) for p in range(B.shape[1])])
    if np.any(~np.isfinite(X)):
        raise FloatingPointError("Maxwell truth solve produced non-finite fields")
    bnorm = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
    residual = B - A @ X
    return X, float(np.max(np.linalg.norm(residual, axis=0) / bnorm))


def _port_truth_from_context(background, context):
    if not hasattr(background, "outward_loss_weights"):
        raise ValueError("production Maxwell truth requires an independent outward-power form")
    if getattr(background, "_sdfmpneo_thermal_maxwell_cache_path", None) is not None:
        # The cache lives in unified_thermal because it predates the spatial-Joule
        # representation.  Reuse it here as a generic certified port-field cache:
        # cache hits are rechecked against the current physical A/B and misses
        # still call this module's production _solve_port_fields implementation.
        from .unified_thermal import (
            _certify_cached_port_fields,
            _maxwell_port_fields,
        )

        X = _maxwell_port_fields(background, context)
        max_linear_residual = _certify_cached_port_fields(
            background,
            context,
            X,
        )
    else:
        X, max_linear_residual = _solve_port_fields(background, context)
    source = np.asarray(context.source_shape, float)
    reaction = -source.T @ X
    reciprocity = float(
        np.linalg.norm(reaction - reaction.T)
        / max(float(np.linalg.norm(reaction)), np.finfo(float).tiny)
    )
    sigma, edge_loss = _edge_loss_weights(background, context)
    d = _hermitian(X.conj().T @ (edge_loss[:, None] * X))
    outward_weights = np.asarray(background.outward_loss_weights(), float).reshape(-1)
    if outward_weights.shape != (background.n_edges,) or np.any(outward_weights < -1e-14):
        raise ValueError("invalid open-boundary outward-power weights")
    d_out = _hermitian(X.conj().T @ (outward_weights[:, None] * X))
    raw_herm_z = _hermitian(reaction)
    balance_scale = max(
        np.linalg.norm(raw_herm_z), np.linalg.norm(d + d_out), np.finfo(float).tiny
    )
    power_balance = float(np.linalg.norm(raw_herm_z - d - d_out) / balance_scale)
    z = 0.5 * (reaction + reaction.T)
    implied = _hermitian(z) - d
    source_regularization = getattr(context, "source_regularization", ())
    regularized = bool(
        len(source_regularization) == source.shape[1]
        and all(
            row.get("model") == getattr(background, "source_model", None)
            and float(row.get("conductor_width", 0.0)) > 0.0
            and float(row.get("conductor_thickness", 0.0)) > 0.0
            and abs(float(row.get("heat_weight_sum", 0.0)) - 1.0) <= 1e-12
            for row in source_regularization
        )
    )
    audit = {
        "max_linear_relative_residual": max_linear_residual,
        "reciprocity_relative_error": reciprocity,
        "minimum_d_vol_eigenvalue": float(np.min(np.linalg.eigvalsh(d)).real),
        "minimum_physical_outward_eigenvalue": float(np.min(np.linalg.eigvalsh(d_out)).real),
        "minimum_implied_outward_eigenvalue": float(np.min(np.linalg.eigvalsh(implied)).real),
        "open_boundary_power_balance_relative_error": power_balance,
        "source_regularization_available": 1.0 if regularized else 0.0,
        "material_fraction_closure_error": float(
            getattr(context, "material_fraction_closure_error", np.inf)
        ),
        "independent_outward_power_available": True,
        "boundary_model": str(getattr(background, "boundary_model", "open_impedance")),
    }
    return z, d, d_out, X, sigma, audit


def solve_port_truth_tensors(background, geometry):
    _require_static_field_materials(background)
    context = background.geometry_context(geometry, assemble_thermal=False)
    z, d, d_out, _, _, audit = _port_truth_from_context(background, context)
    return z, d, d_out, audit


def solve_truth_tensors(background, geometry):
    """Generate Z/D/H using the current geometry-specific Phi(g)."""
    _require_static_field_materials(background)
    context = background.geometry_context(geometry, assemble_thermal=True)
    z, d, _, X, sigma, audit = _port_truth_from_context(background, context)
    phi = np.asarray(context.thermal_basis, float)
    modal = []
    for j in range(phi.shape[1]):
        weighted_edge = np.asarray(
            background.edge_cell_hodge @ (sigma * phi[:, j])
        ).reshape(-1)
        modal.append(_hermitian(X.conj().T @ (weighted_edge[:, None] * X)))
    modal = np.asarray(modal, complex)
    phi_min, phi_max = _conductivity_support_bounds(phi, sigma)

    loewner_violation = 0.0
    for j, h in enumerate(modal):
        low = np.min(np.linalg.eigvalsh(_hermitian(h - phi_min[j] * d))).real
        high = np.min(np.linalg.eigvalsh(_hermitian(phi_max[j] * d - h))).real
        scale = max(np.linalg.norm(d), np.linalg.norm(h), np.finfo(float).tiny)
        loewner_violation = max(loewner_violation, float(max(-low, -high, 0.0) / scale))

    power_consistency = 0.0
    modal_consistency = 0.0
    for current in _audit_current_vectors(X.shape[1]):
        field = X @ current
        q_cells = _cell_volume_heat(background, sigma, field)
        direct_power = float(np.sum(q_cells))
        tensor_power = float(0.5 * np.real(current.conj() @ d @ current))
        pscale = max(abs(direct_power), abs(tensor_power), np.finfo(float).tiny)
        power_consistency = max(power_consistency, abs(direct_power - tensor_power) / pscale)

        direct_modal = phi.T @ q_cells
        tensor_modal = 0.5 * np.real(
            np.einsum("p,rpq,q->r", current.conj(), modal, current, optimize=True)
        )
        mscale = max(
            float(np.linalg.norm(direct_modal)),
            float(np.linalg.norm(tensor_modal)),
            np.finfo(float).tiny,
        )
        modal_consistency = max(
            modal_consistency,
            float(np.linalg.norm(direct_modal - tensor_modal) / mscale),
        )

    audit["maximum_relative_loewner_violation"] = loewner_violation
    audit["joule_total_power_relative_error"] = float(power_consistency)
    audit["joule_modal_contraction_relative_error"] = float(modal_consistency)
    return z, d, modal, phi_min, phi_max, audit


@dataclass
class TensorDataset:
    inputs: np.ndarray
    outputs: np.ndarray
    phi_min: np.ndarray
    phi_max: np.ndarray
    split: np.ndarray
    audit: dict
    n_ports: int
    thermal_rank: int

    def indices(self, name):
        return np.flatnonzero(self.split == name)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.audit)
        np.savez_compressed(
            path,
            inputs=np.asarray(self.inputs, float),
            outputs=np.asarray(self.outputs, float),
            phi_min=np.asarray(self.phi_min, float),
            phi_max=np.asarray(self.phi_max, float),
            split=np.asarray(self.split, "U16"),
            n_ports=np.asarray(self.n_ports),
            thermal_rank=np.asarray(self.thermal_rank),
            audit_keys=np.asarray(keys, "U64"),
            audit_values=np.asarray([float(self.audit[k]) for k in keys], float),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            keys = data["audit_keys"].astype(str).tolist()
            values = data["audit_values"].astype(float).tolist()
            return cls(
                np.asarray(data["inputs"], float),
                np.asarray(data["outputs"], float),
                np.asarray(data["phi_min"], float),
                np.asarray(data["phi_max"], float),
                data["split"].astype(str),
                dict(zip(keys, values)),
                int(data["n_ports"]),
                int(data["thermal_rank"]),
            )


def _split_labels(n, seed):
    n = int(n)
    if n < 6:
        raise ValueError("at least six tensor geometries are required")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_audit = max(1, int(round(0.1 * n)))
    n_test = max(1, int(round(0.1 * n)))
    n_val = max(1, int(round(0.1 * n)))
    while n - n_audit - n_test - n_val < 3:
        if n_audit > 1:
            n_audit -= 1
        elif n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    split = np.full(n, "train", dtype="U16")
    split[order[:n_audit]] = "audit"
    split[order[n_audit:n_audit + n_test]] = "test"
    split[order[n_audit + n_test:n_audit + n_test + n_val]] = "validation"
    return split


def generate_tensor_dataset(background, geometries, *, seed=0, monitor=None):
    geometries = list(geometries)
    if getattr(background, "thermal_library", None) is None:
        raise ValueError("geometry-aware thermal library must be frozen before tensor labels")
    inputs, outputs, mins, maxs, audits = [], [], [], [], []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        z, d, modal, phi_min, phi_max, audit = solve_truth_tensors(background, geometry)
        inputs.append(encode_geometry(geometry))
        outputs.append(pack_tensors(z, d, modal))
        mins.append(phi_min)
        maxs.append(phi_max)
        audits.append(audit)
        print(
            "生成 geometry-aware Z_field / D_vol / H_j truth……"
            f"{100.0 * (index + 1) / len(geometries):5.1f}%  ({index + 1}/{len(geometries)})",
            flush=True,
        )
    numeric_audit = {
        "maximum_linear_relative_residual": max(a["max_linear_relative_residual"] for a in audits),
        "maximum_reciprocity_relative_error": max(a["reciprocity_relative_error"] for a in audits),
        "minimum_d_vol_eigenvalue": min(a["minimum_d_vol_eigenvalue"] for a in audits),
        "minimum_physical_outward_eigenvalue": min(a["minimum_physical_outward_eigenvalue"] for a in audits),
        "minimum_implied_outward_eigenvalue": min(a["minimum_implied_outward_eigenvalue"] for a in audits),
        "maximum_open_boundary_power_balance_relative_error": max(
            a["open_boundary_power_balance_relative_error"] for a in audits
        ),
        "maximum_relative_loewner_violation": max(
            a["maximum_relative_loewner_violation"] for a in audits
        ),
        "maximum_joule_total_power_relative_error": max(
            a["joule_total_power_relative_error"] for a in audits
        ),
        "maximum_joule_modal_contraction_relative_error": max(
            a["joule_modal_contraction_relative_error"] for a in audits
        ),
        "maximum_material_fraction_closure_error": max(
            a["material_fraction_closure_error"] for a in audits
        ),
        "source_regularization_available": min(
            a["source_regularization_available"] for a in audits
        ),
        "independent_outward_power_available": 1.0,
    }
    return TensorDataset(
        np.asarray(inputs, float),
        np.asarray(outputs, float),
        np.asarray(mins, float),
        np.asarray(maxs, float),
        _split_labels(len(geometries), seed),
        numeric_audit,
        len(background.coil_materials),
        int(background.thermal_rank),
    )


class UnifiedTensorSurrogate:
    def __init__(self, network, output_mean, output_scale, pod_basis, n_ports, thermal_rank):
        self.network = network
        self.output_mean = np.asarray(output_mean, float).reshape(-1)
        self.output_scale = np.asarray(output_scale, float).reshape(-1)
        self.pod_basis = np.asarray(pod_basis, float)
        self.n_ports = int(n_ports)
        self._thermal_rank = int(thermal_rank)
        full = tensor_output_dimension(self.n_ports, self._thermal_rank)
        if self.output_mean.shape != (full,) or self.output_scale.shape != (full,):
            raise ValueError("output normalization dimensions differ from tensor schema")
        if (
            self.pod_basis.ndim != 2
            or self.pod_basis.shape[0] != full
            or self.pod_basis.shape[1] < 1
        ):
            raise ValueError("invalid tensor POD basis")
        if self.network.config.output_dimension != self.pod_basis.shape[1]:
            raise ValueError("network output dimension does not match tensor POD rank")

    @property
    def thermal_rank(self):
        return self._thermal_rank

    @property
    def pod_rank(self):
        return int(self.pod_basis.shape[1])

    def predict_from_encoded(self, encoded, phi_min, phi_max):
        import torch

        phi_min = np.asarray(phi_min, float).reshape(-1)
        phi_max = np.asarray(phi_max, float).reshape(-1)
        if len(phi_min) != self.thermal_rank or phi_max.shape != phi_min.shape:
            raise ValueError(
                "geometry-dependent modal bounds differ from surrogate thermal rank"
            )
        parameter = next(self.network.parameters())
        x = torch.as_tensor(
            np.asarray(encoded, float), dtype=parameter.dtype, device=parameter.device
        )
        with torch.no_grad():
            beta = self.network(x).detach().cpu().numpy().astype(float)
        packed = self.output_mean + self.output_scale * (self.pod_basis @ beta)
        return decode_physical_tensors(packed, self.n_ports, phi_min, phi_max)

    def predict(self, geometry, phi_min, phi_max):
        return self.predict_from_encoded(encode_geometry(geometry), phi_min, phi_max)

    def checkpoint(self):
        parameter = next(self.network.parameters())
        normalizer = FeatureNormalizer(
            self.network.input_mean.detach().cpu().numpy(),
            self.network.input_scale.detach().cpu().numpy(),
        )
        return {
            "schema_version": _SCHEMA_VERSION,
            "network_config": self.network.config.to_dict(),
            "input_mean": normalizer.mean,
            "input_scale": normalizer.scale,
            "output_mean": self.output_mean,
            "output_scale": self.output_scale,
            "pod_basis": self.pod_basis,
            "n_ports": self.n_ports,
            "thermal_rank": self.thermal_rank,
            "network_state": {k: v.detach().cpu() for k, v in self.network.state_dict().items()},
            "dtype": str(parameter.dtype).replace("torch.", ""),
        }

    @classmethod
    def from_checkpoint(cls, payload, device="cpu"):
        import torch

        if int(payload.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise ValueError("unsupported tensor-surrogate artifact version")
        config = ResidualMLPConfig(**dict(payload["network_config"]))
        normalizer = FeatureNormalizer(
            np.asarray(payload["input_mean"], float),
            np.asarray(payload["input_scale"], float),
        )
        network = build_residual_mlp(config, normalizer)
        dtype = torch.float32 if payload.get("dtype") == "float32" else torch.float64
        network = network.to(device=device, dtype=dtype)
        network.load_state_dict(payload["network_state"])
        network.eval()
        return cls(
            network,
            payload["output_mean"],
            payload["output_scale"],
            payload["pod_basis"],
            int(payload["n_ports"]),
            int(payload["thermal_rank"]),
        )


_SPATIAL_SCHEMA_VERSION = 1


def spatial_tensor_output_dimension(n_ports, n_cells):
    n = int(n_ports)
    m = int(n_cells)
    if n < 1 or m < 1:
        raise ValueError("spatial tensor dimensions must be positive")
    return n * (n + 1) + n * n * (1 + m)


def pack_spatial_tensors(z_field, d_vol, cell_h):
    cells = np.asarray(cell_h, complex)
    if cells.ndim != 3 or cells.shape[1] != cells.shape[2]:
        raise ValueError("cell Joule tensors must have shape (n_cells, ports, ports)")
    return np.concatenate(
        [pack_complex_symmetric(z_field), pack_hermitian(d_vol)]
        + [pack_hermitian(h) for h in cells]
    )


def unpack_spatial_tensors(packed, n_ports, n_cells):
    p = np.asarray(packed, float).reshape(-1)
    n = int(n_ports)
    m = int(n_cells)
    z_size, h_size, _ = tensor_block_sizes(n, 0)
    expected = z_size + h_size * (1 + m)
    if p.size != expected:
        raise ValueError(
            f"spatial tensor packed size mismatch: expected {expected}, got {p.size}"
        )
    z = unpack_complex_symmetric(p[:z_size], n)
    d = unpack_hermitian(p[z_size:z_size + h_size], n)
    start = z_size + h_size
    cells = np.asarray(
        [
            unpack_hermitian(
                p[start + j * h_size:start + (j + 1) * h_size],
                n,
            )
            for j in range(m)
        ],
        complex,
    )
    return z, d, cells


def _batch_psd_clip(value, floor=0.0):
    a = np.asarray(value, complex)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("batch PSD projection expects (count,n,n)")
    a = 0.5 * (a + np.swapaxes(a.conj(), 1, 2))
    w, v = np.linalg.eigh(a)
    w = np.maximum(np.asarray(w.real, float), float(floor))
    projected = np.einsum(
        "...ik,...k,...jk->...ij",
        v,
        w,
        v.conj(),
        optimize=True,
    )
    return 0.5 * (
        projected + np.swapaxes(projected.conj(), 1, 2)
    )


def _psd_sqrt_and_inverse(value, *, inverse=False):
    a = _hermitian(value)
    w, v = np.linalg.eigh(a)
    scale = max(float(np.max(np.abs(w))), 1.0)
    floor = scale * 1e-12
    if inverse:
        coeff = 1.0 / np.sqrt(np.maximum(w.real, floor))
    else:
        coeff = np.sqrt(np.maximum(w.real, 0.0))
    return _hermitian((v * coeff) @ v.conj().T)


def normalize_cell_joule_tensors(cell_h, d_vol):
    """Project each cell PSD and enforce sum(cell_h)==D by congruence.

    Cell tensors represent the *D* convention, i.e. for a peak phasor current
    c the cell power is 0.5*c^H H_cell c.  PSD clipping guarantees nonnegative
    spatial Joule density for every current.  A single port-space congruence
    then restores the exact corrected total D tensor without changing PSD.
    """
    d = _psd_clip(d_vol)
    cells = _batch_psd_clip(cell_h)
    n_cells, n, _ = cells.shape
    if n_cells < 1:
        raise ValueError("cell Joule tensor field is empty")

    # Make the aggregate full-support before whitening.  The added amount is at
    # machine-scale relative to D and is removed by the exact congruence target.
    trace = max(float(np.trace(d).real), np.finfo(float).tiny)
    regularization = trace * 1e-14 / float(n_cells * n)
    if regularization > 0.0:
        cells = cells + regularization * np.eye(n, dtype=complex)[None, :, :]

    total = _hermitian(np.sum(cells, axis=0))
    d_half = _psd_sqrt_and_inverse(d, inverse=False)
    total_inv_half = _psd_sqrt_and_inverse(total, inverse=True)
    transform = d_half @ total_inv_half
    corrected = np.einsum(
        "ab,kbc,dc->kad",
        transform,
        cells,
        transform.conj(),
        optimize=True,
    )
    corrected = 0.5 * (corrected + np.swapaxes(corrected.conj(), 1, 2))

    # Remove only roundoff-level total mismatch with one final congruence.  This
    # also covers near-singular D without introducing signed cell corrections.
    total2 = _hermitian(np.sum(corrected, axis=0))
    err = float(
        np.linalg.norm(total2 - d)
        / max(float(np.linalg.norm(d)), np.finfo(float).tiny)
    )
    if err > 5e-11:
        total2_inv_half = _psd_sqrt_and_inverse(total2, inverse=True)
        transform2 = d_half @ total2_inv_half
        corrected = np.einsum(
            "ab,kbc,dc->kad",
            transform2,
            corrected,
            transform2.conj(),
            optimize=True,
        )
        corrected = 0.5 * (
            corrected + np.swapaxes(corrected.conj(), 1, 2)
        )

    final_total = _hermitian(np.sum(corrected, axis=0))
    final_scale = max(
        float(np.linalg.norm(d)),
        np.finfo(float).tiny,
    )
    final_mismatch = float(
        np.linalg.norm(final_total - d) / final_scale
    )
    final_minimum = float(
        np.min(np.linalg.eigvalsh(corrected).real)
    )
    eigen_scale = max(
        float(np.max(np.abs(np.linalg.eigvalsh(d).real))),
        1.0,
    )
    if (
        not np.isfinite(final_mismatch)
        or final_mismatch > 1e-10
    ):
        raise FloatingPointError(
            "cell Joule congruence failed exact sum-to-D invariant: "
            f"relative mismatch={final_mismatch:.3e}"
        )
    if (
        not np.isfinite(final_minimum)
        or final_minimum < -1e-10 * eigen_scale
    ):
        raise FloatingPointError(
            "cell Joule PSD projection lost nonnegativity: "
            f"minimum eigenvalue={final_minimum:.3e}"
        )

    return d, corrected


def cell_joule_tensors_from_port_fields(background, sigma, fields):
    """Exact coarse-grid cellwise Hermitian Joule tensors from port fields."""
    X = np.asarray(fields, complex)
    sigma = np.asarray(sigma, float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != background.n_edges:
        raise ValueError("port-field matrix has incompatible shape")
    if sigma.shape != (background.n_cells,):
        raise ValueError("cell conductivity has incompatible shape")
    n_ports = X.shape[1]
    cells = np.zeros(
        (background.n_cells, n_ports, n_ports),
        complex,
    )
    hodge_t = background.edge_cell_hodge.T
    for i in range(n_ports):
        for j in range(i, n_ports):
            edge_product = X[:, i].conj() * X[:, j]
            values = sigma * np.asarray(hodge_t @ edge_product).reshape(-1)
            cells[:, i, j] = values
            cells[:, j, i] = values.conj()
    return 0.5 * (cells + np.swapaxes(cells.conj(), 1, 2))



def _pack_hermitian_batch(matrices):
    a = np.asarray(matrices, complex)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("Hermitian batch must have shape (count,n,n)")
    a = 0.5 * (a + np.swapaxes(a.conj(), 1, 2))
    n = a.shape[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    out = [np.asarray(a[:, i, i].real, float) for i in range(n)]
    out.extend(np.asarray(a[:, i, j].real, float) for i, j in pairs)
    out.extend(np.asarray(a[:, i, j].imag, float) for i, j in pairs)
    return np.column_stack(out)


def _unpack_hermitian_batch(packed, n):
    p = np.asarray(packed, float)
    n = int(n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    expected = n + 2 * len(pairs)
    if p.ndim != 2 or p.shape[1] != expected:
        raise ValueError(
            f"Hermitian batch packed size mismatch: expected {expected}"
        )
    h = np.zeros((p.shape[0], n, n), complex)
    for i in range(n):
        h[:, i, i] = p[:, i]
    start = n
    for k, (i, j) in enumerate(pairs):
        value = p[:, start + k] + 1j * p[:, start + len(pairs) + k]
        h[:, i, j] = value
        h[:, j, i] = value.conj()
    return h




def spatial_joule_density_prior(
    background,
    spatial_context,
    cell_indices=None,
    *,
    strength=0.9,
):
    """Mean-one Maxwell-free prior centered on the deposited conductor paths."""
    s = float(strength)
    if not 0.0 <= s < 1.0:
        raise ValueError(
            "field density prior strength must lie in [0,1)"
        )
    if cell_indices is None:
        ids = np.arange(background.n_cells, dtype=int)
    else:
        ids = np.asarray(cell_indices, int).reshape(-1)
    n_ports = len(spatial_context.line_heat_weights)
    if n_ports < 1:
        raise ValueError("spatial context has no source heat weights")
    source_density = np.zeros(ids.size, float)
    for weights in spatial_context.line_heat_weights:
        source_density += np.asarray(weights, float)[ids]
    source_density *= (
        float(background.n_cells) / float(n_ports)
    )
    prior = (1.0 - s) + s * source_density
    if np.any(~np.isfinite(prior)) or np.any(prior <= 0.0):
        raise FloatingPointError(
            "spatial Joule density prior is not finite/positive"
        )
    return np.asarray(prior, float)


def pack_whitened_field_factors(
    whitened,
    *,
    total_cells,
    density_prior=None,
):
    """Encode PSD whitened cells as residual log-density + PSD square-root shape."""
    cells = _batch_psd_clip(np.asarray(whitened, complex))
    count, n, _ = cells.shape
    traces = np.maximum(
        np.trace(cells, axis1=1, axis2=2).real,
        0.0,
    )
    density = (
        traces * float(total_cells) / float(n)
    )
    if density_prior is None:
        prior = np.ones(count, float)
    else:
        prior = np.asarray(
            density_prior,
            float,
        ).reshape(-1)
        if (
            prior.shape != (count,)
            or np.any(~np.isfinite(prior))
            or np.any(prior <= 0.0)
        ):
            raise ValueError(
                "density_prior must be positive with one value per cell"
            )
    log_density = np.log(
        np.maximum(density, 1e-12) / prior
    )

    floor = max(
        float(np.max(traces)) * 1e-14,
        np.finfo(float).tiny,
    )
    shape = np.empty_like(cells)
    identity = np.eye(n, dtype=complex) / float(n)
    positive = traces > floor
    shape[~positive] = identity
    if np.any(positive):
        shape[positive] = (
            cells[positive]
            / traces[positive, None, None]
        )

    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    roots = np.einsum(
        "kij,kj,klj->kil",
        eigenvectors,
        np.sqrt(np.maximum(eigenvalues.real, 0.0)),
        eigenvectors.conj(),
        optimize=True,
    )
    roots = 0.5 * (
        roots + np.swapaxes(roots.conj(), 1, 2)
    )
    return np.column_stack(
        (
            log_density,
            _pack_hermitian_batch(roots),
        )
    )


def unpack_whitened_field_factors(
    packed,
    *,
    n_ports,
    total_cells,
    density_prior=None,
    log_density_bounds=None,
):
    """Decode residual log-density + Hermitian factor into an intrinsically PSD field."""
    values = np.asarray(packed, float)
    n = int(n_ports)
    if (
        values.ndim != 2
        or values.shape[1] != 1 + n * n
    ):
        raise ValueError("whitened field factor width mismatch")
    if log_density_bounds is None:
        lower, upper = -35.0, 35.0
    else:
        bounds = np.asarray(
            log_density_bounds,
            float,
        ).reshape(-1)
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(
                "log_density_bounds must be finite increasing pair"
            )
        lower, upper = float(bounds[0]), float(bounds[1])
    log_density = np.clip(values[:, 0], lower, upper)
    roots = _unpack_hermitian_batch(values[:, 1:], n)
    shapes = np.einsum(
        "kab,kcb->kac",
        roots,
        roots.conj(),
        optimize=True,
    )
    shapes = 0.5 * (
        shapes + np.swapaxes(shapes.conj(), 1, 2)
    )
    traces = np.trace(
        shapes,
        axis1=1,
        axis2=2,
    ).real
    bad = traces <= np.finfo(float).tiny
    if np.any(bad):
        shapes[bad] = (
            np.eye(n, dtype=complex)[None, :, :]
            / float(n)
        )
        traces[bad] = 1.0
    shapes = shapes / traces[:, None, None]
    if density_prior is None:
        prior = np.ones(values.shape[0], float)
    else:
        prior = np.asarray(
            density_prior,
            float,
        ).reshape(-1)
        if (
            prior.shape != (values.shape[0],)
            or np.any(~np.isfinite(prior))
            or np.any(prior <= 0.0)
        ):
            raise ValueError(
                "density_prior must be positive with one value per cell"
            )
    density = (
        prior
        * np.exp(log_density)
        * float(n)
        / float(total_cells)
    )
    return density[:, None, None] * shapes


def whiten_cell_joule_tensors(cell_h, d_vol):
    """Return dimensionless PSD cell tensors whose sum is identity."""
    d = _psd_clip(d_vol)
    inv = _psd_sqrt_and_inverse(d, inverse=True)
    cells = np.asarray(cell_h, complex)
    whitened = np.einsum(
        "ab,kbc,dc->kad",
        inv,
        cells,
        inv.conj(),
        optimize=True,
    )
    whitened = 0.5 * (
        whitened + np.swapaxes(whitened.conj(), 1, 2)
    )
    _, whitened = normalize_cell_joule_tensors(
        whitened,
        np.eye(d.shape[0], dtype=complex),
    )
    return whitened


def spatial_cell_features(
    background,
    geometry,
    cell_indices=None,
    *,
    spatial_context=None,
):
    """Geometry/material/source features for the local Joule neural field.

    The context terms are Maxwell-free analytic geometry data: deposited copper
    fractions, package/seawater fractions, and the normalized stranded-source
    line heat weights.  They explicitly expose the actual moving conductor
    support instead of forcing an MLP to reconstruct a thin spiral from raw
    center coordinates alone.
    """
    g = (
        geometry
        if isinstance(geometry, UnifiedUWPTGeometry)
        else UnifiedUWPTGeometry.from_mapping(geometry)
    )
    if len(g.coils) != len(g.packages):
        raise ValueError("spatial neural field requires one package per port")
    if cell_indices is None:
        centers = np.asarray(background.cell_centers, float)
    else:
        ids = np.asarray(cell_indices, int).reshape(-1)
        centers = np.asarray(background.cell_centers, float)[ids]

    geom = encode_geometry_invariant(g)
    repeated = np.broadcast_to(
        geom[None, :],
        (centers.shape[0], geom.size),
    )
    pieces = [np.asarray(repeated, float)]

    context = (
        spatial_context
        if spatial_context is not None
        else background._spatial_context(g)
    )
    ids = (
        np.arange(background.n_cells, dtype=int)
        if cell_indices is None
        else np.asarray(cell_indices, int).reshape(-1)
    )
    if context.geometry.n_ports != g.n_ports:
        raise ValueError("spatial context port count differs from geometry")

    # Exact analytic material occupancy.  The first channels locate each moving
    # copper path and package; seawater closes the partition.
    for material in background.coil_materials:
        fraction = np.asarray(
            context.fractions[material],
            float,
        )[ids]
        pieces.append(fraction[:, None])
    for material in background.package_materials:
        fraction = np.asarray(
            context.fractions[material],
            float,
        )[ids]
        pieces.append(fraction[:, None])
    seawater = np.asarray(
        context.fractions[background.seawater_material],
        float,
    )[ids]
    pieces.append(seawater[:, None])

    # Port source-support indicator.  Multiplying by n_cells makes the feature
    # O(1) on occupied cells while preserving exact zero/near-zero support.
    for weights in context.line_heat_weights:
        values = np.asarray(weights, float)[ids]
        scaled = values * float(background.n_cells)
        pieces.append(
            np.log1p(np.maximum(scaled, 0.0))[:, None]
        )

    def signed_log(value):
        a = np.asarray(value, float)
        return np.sign(a) * np.log1p(np.abs(a))

    for coil, package in zip(g.coils, g.packages):
        local = np.asarray(coil.pose.inverse(centers), float)
        z_scale = max(
            float(package.half_extent[2]),
            float(coil.conductor_thickness),
            float(coil.pitch),
            np.finfo(float).tiny,
        )
        scale = np.array(
            [
                max(float(coil.outer_half_size), np.finfo(float).tiny),
                max(float(coil.outer_half_size), np.finfo(float).tiny),
                z_scale,
            ],
            float,
        )
        normalized = local / scale[None, :]
        pieces.append(signed_log(normalized))
        radial = np.sqrt(
            normalized[:, 0] ** 2 + normalized[:, 1] ** 2
        )
        pieces.append(np.log1p(radial)[:, None])
        pieces.append(np.log1p(np.abs(normalized[:, 2]))[:, None])

        package_local = np.asarray(
            package.pose.inverse(centers),
            float,
        )
        package_scale = np.maximum(
            np.asarray(package.half_extent, float),
            np.finfo(float).tiny,
        )
        pieces.append(
            signed_log(package_local / package_scale[None, :])
        )

    out = np.column_stack(pieces)
    if np.any(~np.isfinite(out)):
        raise FloatingPointError(
            "spatial neural-field features contain non-finite values"
        )
    return np.asarray(out, float)


@dataclass(frozen=True)
class SpatialDecodedTensors:
    z_field: np.ndarray
    d_vol: np.ndarray
    cell_h: np.ndarray
    implied_d_out: np.ndarray
    zd_projection_correction: float
    spatial_projection_correction: float

    @property
    def projection_correction(self):
        return max(
            float(self.zd_projection_correction),
            float(self.spatial_projection_correction),
        )

    @property
    def h_projection_correction(self):
        # Compatibility name used by existing diagnostics/model containers.
        return float(self.spatial_projection_correction)

    def cell_heat(self, currents):
        c = np.asarray(currents, complex).reshape(-1)
        if c.shape != (self.z_field.shape[0],):
            raise ValueError("current vector has wrong port dimension")
        heat = 0.5 * np.real(
            np.einsum(
                "p,kpq,q->k",
                c.conj(),
                self.cell_h,
                c,
                optimize=True,
            )
        )
        # PSD projection should make this nonnegative; tolerate only roundoff.
        scale = max(float(np.max(np.abs(heat))), 1.0)
        if float(np.min(heat)) < -1e-11 * scale:
            raise RuntimeError("decoded cell Joule field lost nonnegativity")
        heat = np.maximum(np.asarray(heat, float), 0.0)
        target = max(
            float(0.5 * np.real(c.conj() @ self.d_vol @ c)),
            0.0,
        )
        total = float(np.sum(heat))
        if target <= np.finfo(float).tiny:
            return np.zeros_like(heat)
        if total <= np.finfo(float).tiny:
            raise RuntimeError(
                "decoded cell Joule field lost positive total-power support"
            )
        # Removing tiny negative roundoff must not weaken the exact D-volume
        # identity used by the circuit/thermal coupling.
        heat *= target / total
        return heat

    def volume_power(self, currents):
        c = np.asarray(currents, complex).reshape(-1)
        return float(0.5 * np.real(c.conj() @ self.d_vol @ c))

    def implied_outward_power(self, currents):
        c = np.asarray(currents, complex).reshape(-1)
        return float(0.5 * np.real(c.conj() @ self.implied_d_out @ c))


def decode_spatial_tensors(packed, n_ports, n_cells):
    raw = np.asarray(packed, float).reshape(-1)
    z_raw, d_raw, cells_raw = unpack_spatial_tensors(
        raw,
        n_ports,
        n_cells,
    )
    d, cells = normalize_cell_joule_tensors(cells_raw, d_raw)

    r0 = 0.5 * (
        np.asarray(z_raw.real, float)
        + np.asarray(z_raw.real, float).T
    )
    d_out_candidate = _hermitian(r0.astype(complex) - d)
    minimum = float(np.min(np.linalg.eigvalsh(d_out_candidate)).real)
    shift = max(0.0, -minimum)
    d_out = _hermitian(
        d_out_candidate + shift * np.eye(int(n_ports))
    )
    r = np.asarray((d + d_out).real, float)
    x = 0.5 * (
        np.asarray(z_raw.imag, float)
        + np.asarray(z_raw.imag, float).T
    )
    z = r + 1j * x
    implied = _hermitian(z) - d

    corrected = pack_spatial_tensors(z, d, cells)
    z_size, h_size, _ = tensor_block_sizes(int(n_ports), 0)
    zd_stop = z_size + h_size
    zd_correction = float(
        np.linalg.norm(corrected[:zd_stop] - raw[:zd_stop])
        / max(np.linalg.norm(raw[:zd_stop]), np.finfo(float).tiny)
    )
    spatial_correction = float(
        np.linalg.norm(corrected[zd_stop:] - raw[zd_stop:])
        / max(
            np.linalg.norm(raw[zd_stop:]),
            np.finfo(float).tiny,
        )
    )
    return SpatialDecodedTensors(
        z,
        d,
        cells,
        implied,
        zd_correction,
        spatial_correction,
    )


@dataclass
class SpatialTensorDataset:
    inputs: np.ndarray
    outputs: np.ndarray
    split: np.ndarray
    audit: dict
    n_ports: int
    n_cells: int

    representation: str = "cellwise_joule_tensor_v1"

    def indices(self, name):
        return np.flatnonzero(self.split == name)

    @property
    def thermal_rank(self):
        # Compatibility: production spatial source surrogate is thermal-rank free.
        return 0

    @property
    def phi_min(self):
        return np.empty((len(self.inputs), 0), float)

    @property
    def phi_max(self):
        return np.empty((len(self.inputs), 0), float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.audit)
        np.savez_compressed(
            path,
            representation=np.asarray(self.representation),
            inputs=np.asarray(self.inputs, float),
            outputs=np.asarray(self.outputs, float),
            split=np.asarray(self.split, "U16"),
            n_ports=np.asarray(self.n_ports),
            n_cells=np.asarray(self.n_cells),
            audit_keys=np.asarray(keys, "U96"),
            audit_values=np.asarray(
                [float(self.audit[k]) for k in keys],
                float,
            ),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            representation = (
                str(data["representation"])
                if "representation" in data
                else ""
            )
            if representation != "cellwise_joule_tensor_v1":
                raise ValueError(
                    "unsupported spatial tensor dataset representation"
                )
            keys = data["audit_keys"].astype(str).tolist()
            values = data["audit_values"].astype(float).tolist()
            return cls(
                np.asarray(data["inputs"], float),
                np.asarray(data["outputs"], float),
                data["split"].astype(str),
                dict(zip(keys, values)),
                int(data["n_ports"]),
                int(data["n_cells"]),
            )


def project_spatial_dataset_to_production_cone(dataset):
    """Upgrade cached spatial labels to the exact production decoder cone.

    This is intentionally Maxwell-free: old truth caches already contain Z, D
    and cellwise Joule tensors.  Re-decoding and re-packing makes those labels
    obey the same D/cell PSD, exact sum-to-D and implied-outward passivity
    constraints used at inference, without regenerating expensive EM truth.
    """
    if dataset.representation != "cellwise_joule_tensor_v1":
        raise ValueError("unsupported spatial tensor dataset representation")
    projected = []
    maximum_correction = 0.0
    minimum_d = float("inf")
    minimum_outward = float("inf")
    minimum_cell = float("inf")
    maximum_sum_mismatch = 0.0
    for packed in np.asarray(dataset.outputs, float):
        decoded = decode_spatial_tensors(
            packed,
            int(dataset.n_ports),
            int(dataset.n_cells),
        )
        projected.append(
            pack_spatial_tensors(
                decoded.z_field,
                decoded.d_vol,
                decoded.cell_h,
            )
        )
        maximum_correction = max(
            maximum_correction,
            float(decoded.projection_correction),
        )
        minimum_d = min(
            minimum_d,
            float(np.min(np.linalg.eigvalsh(decoded.d_vol).real)),
        )
        minimum_outward = min(
            minimum_outward,
            float(np.min(np.linalg.eigvalsh(decoded.implied_d_out).real)),
        )
        minimum_cell = min(
            minimum_cell,
            float(np.min(np.linalg.eigvalsh(decoded.cell_h).real)),
        )
        maximum_sum_mismatch = max(
            maximum_sum_mismatch,
            float(
                np.linalg.norm(np.sum(decoded.cell_h, axis=0) - decoded.d_vol)
                / max(float(np.linalg.norm(decoded.d_vol)), np.finfo(float).tiny)
            ),
        )
    dataset.outputs = np.asarray(projected, float)
    audit = dict(dataset.audit)
    audit["minimum_d_vol_eigenvalue"] = float(minimum_d)
    audit["minimum_implied_outward_eigenvalue"] = float(minimum_outward)
    audit["minimum_cell_joule_tensor_eigenvalue"] = float(minimum_cell)
    audit["maximum_spatial_joule_total_mismatch"] = float(maximum_sum_mismatch)
    audit["maximum_spatial_truth_projection_correction"] = float(
        max(
            float(audit.get("maximum_spatial_truth_projection_correction", 0.0)),
            maximum_correction,
        )
    )
    dataset.audit = audit
    return float(maximum_correction)


class UnifiedSpatialTensorSurrogate:
    """Two-head surrogate: global Z/D MLP + coordinate-conditioned Joule field."""

    representation = "cellwise_joule_neural_field_v4"

    def __init__(
        self,
        global_network,
        field_network,
        n_ports,
        n_cells,
        *,
        field_chunk_size=65536,
        global_output_mean=None,
        global_output_scale=None,
        field_output_mean=None,
        field_output_scale=None,
        field_density_prior_strength=0.9,
        field_log_density_bounds=(-35.0, 35.0),
    ):
        self.global_network = global_network
        self.field_network = field_network
        self.n_ports = int(n_ports)
        self.n_cells = int(n_cells)
        self.field_chunk_size = max(1, int(field_chunk_size))
        global_dim = tensor_output_dimension(self.n_ports, 0)
        field_dim = 1 + self.n_ports * self.n_ports
        self.global_output_mean = np.zeros(global_dim, float) if global_output_mean is None else np.asarray(global_output_mean, float).reshape(-1)
        self.global_output_scale = np.ones(global_dim, float) if global_output_scale is None else np.asarray(global_output_scale, float).reshape(-1)
        self.field_output_mean = np.zeros(field_dim, float) if field_output_mean is None else np.asarray(field_output_mean, float).reshape(-1)
        self.field_output_scale = np.ones(field_dim, float) if field_output_scale is None else np.asarray(field_output_scale, float).reshape(-1)
        self.field_density_prior_strength = float(
            field_density_prior_strength
        )
        self.field_log_density_bounds = np.asarray(
            field_log_density_bounds,
            float,
        ).reshape(-1)
        if (
            not 0.0 <= self.field_density_prior_strength < 1.0
            or self.field_log_density_bounds.shape != (2,)
            or np.any(~np.isfinite(self.field_log_density_bounds))
            or self.field_log_density_bounds[0]
            >= self.field_log_density_bounds[1]
        ):
            raise ValueError(
                "spatial field density prior/bounds are invalid"
            )
        if (
            self.global_output_mean.shape != (global_dim,)
            or self.global_output_scale.shape != (global_dim,)
            or self.field_output_mean.shape != (field_dim,)
            or self.field_output_scale.shape != (field_dim,)
            or np.any(~np.isfinite(self.global_output_mean))
            or np.any(~np.isfinite(self.global_output_scale))
            or np.any(~np.isfinite(self.field_output_mean))
            or np.any(~np.isfinite(self.field_output_scale))
            or np.any(self.global_output_scale <= 0.0)
            or np.any(self.field_output_scale <= 0.0)
        ):
            raise ValueError("spatial surrogate output normalization is invalid")
        if self.global_network.config.output_dimension != global_dim:
            raise ValueError(
                "global spatial surrogate output dimension differs from Z/D schema"
            )
        if self.field_network.config.output_dimension != field_dim:
            raise ValueError(
                "spatial neural-field output dimension differs from Hermitian schema"
            )

    @property
    def thermal_rank(self):
        return 0

    @property
    def pod_rank(self):
        # Compatibility field in reports/UI: production v2 has no spatial POD.
        return 0

    def _network_numpy(self, network, values):
        import torch

        parameter = next(network.parameters())
        x = torch.as_tensor(
            np.asarray(values, float),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.no_grad():
            return (
                network(x)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )

    def _global_tensors(self, geometry):
        encoded = encode_geometry_invariant(
            geometry
        )[None, :]
        normalized = self._network_numpy(
            self.global_network,
            encoded,
        )[0]
        packed = (
            self.global_output_mean
            + self.global_output_scale * normalized
        )
        return decode_spatial_global_tensors(
            packed,
            self.n_ports,
        )

    def predict(self, geometry, *, background):
        g = (
            geometry
            if isinstance(geometry, UnifiedUWPTGeometry)
            else UnifiedUWPTGeometry.from_mapping(geometry)
        )
        if int(background.n_cells) != self.n_cells:
            raise ValueError(
                "spatial surrogate background cell count differs from artifact"
            )

        global_tensors = self._global_tensors(g)
        spatial_context = background._spatial_context(g)
        raw_rows = []
        for start in range(0, self.n_cells, self.field_chunk_size):
            stop = min(self.n_cells, start + self.field_chunk_size)
            ids = np.arange(start, stop, dtype=int)
            features = spatial_cell_features(
                background,
                g,
                ids,
                spatial_context=spatial_context,
            )
            normalized = self._network_numpy(
                self.field_network,
                features,
            )
            raw_rows.append(
                self.field_output_mean[None, :]
                + self.field_output_scale[None, :] * normalized
            )
        raw_packed = np.vstack(raw_rows)
        density_prior = spatial_joule_density_prior(
            background,
            spatial_context,
            strength=self.field_density_prior_strength,
        )
        raw_shape = unpack_whitened_field_factors(
            raw_packed,
            n_ports=self.n_ports,
            total_cells=self.n_cells,
            density_prior=density_prior,
            log_density_bounds=self.field_log_density_bounds,
        )

        identity = np.eye(self.n_ports, dtype=complex)
        _, normalized_shape = normalize_cell_joule_tensors(
            raw_shape,
            identity,
        )
        d_half = _psd_sqrt_and_inverse(
            global_tensors.d_vol,
            inverse=False,
        )
        cells = np.einsum(
            "ab,kbc,dc->kad",
            d_half,
            normalized_shape,
            d_half.conj(),
            optimize=True,
        )
        cells = 0.5 * (
            cells + np.swapaxes(cells.conj(), 1, 2)
        )

        spatial_correction = float(
            np.linalg.norm(normalized_shape - raw_shape)
            / max(
                float(np.linalg.norm(raw_shape)),
                np.finfo(float).tiny,
            )
        )
        total_error = float(
            np.linalg.norm(
                np.sum(cells, axis=0)
                - global_tensors.d_vol
            )
            / max(
                float(np.linalg.norm(global_tensors.d_vol)),
                np.finfo(float).tiny,
            )
        )
        if total_error > 1e-10:
            raise FloatingPointError(
                "spatial neural field lost exact sum-to-D invariant: "
                f"{total_error:.3e}"
            )

        return SpatialDecodedTensors(
            np.asarray(global_tensors.z_field, complex),
            np.asarray(global_tensors.d_vol, complex),
            np.asarray(cells, complex),
            np.asarray(global_tensors.implied_d_out, complex),
            float(global_tensors.zd_projection_correction),
            spatial_correction,
        )

    def checkpoint(self):
        parameter = next(self.global_network.parameters())
        global_normalizer = FeatureNormalizer(
            self.global_network.input_mean.detach().cpu().numpy(),
            self.global_network.input_scale.detach().cpu().numpy(),
        )
        field_normalizer = FeatureNormalizer(
            self.field_network.input_mean.detach().cpu().numpy(),
            self.field_network.input_scale.detach().cpu().numpy(),
        )
        return {
            "schema_version": 5,
            "representation": self.representation,
            "global_network_config": self.global_network.config.to_dict(),
            "field_network_config": self.field_network.config.to_dict(),
            "global_input_mean": global_normalizer.mean,
            "global_input_scale": global_normalizer.scale,
            "field_input_mean": field_normalizer.mean,
            "field_input_scale": field_normalizer.scale,
            "n_ports": self.n_ports,
            "n_cells": self.n_cells,
            "field_chunk_size": self.field_chunk_size,
            "global_output_mean": self.global_output_mean,
            "global_output_scale": self.global_output_scale,
            "field_output_mean": self.field_output_mean,
            "field_output_scale": self.field_output_scale,
            "field_density_prior_strength":
                self.field_density_prior_strength,
            "field_log_density_bounds":
                self.field_log_density_bounds,
            "global_network_state": {
                k: v.detach().cpu()
                for k, v in self.global_network.state_dict().items()
            },
            "field_network_state": {
                k: v.detach().cpu()
                for k, v in self.field_network.state_dict().items()
            },
            "dtype": str(parameter.dtype).replace("torch.", ""),
        }

    @classmethod
    def from_checkpoint(cls, payload, device="cpu"):
        import torch

        if (
            int(payload.get("schema_version", -1)) != 5
            or payload.get("representation")
            != "cellwise_joule_neural_field_v4"
        ):
            raise ValueError(
                "unsupported spatial neural-field artifact version"
            )
        dtype = (
            torch.float32
            if payload.get("dtype") == "float32"
            else torch.float64
        )

        global_config = ResidualMLPConfig(
            **dict(payload["global_network_config"])
        )
        global_normalizer = FeatureNormalizer(
            np.asarray(payload["global_input_mean"], float),
            np.asarray(payload["global_input_scale"], float),
        )
        global_network = build_residual_mlp(
            global_config,
            global_normalizer,
        ).to(device=device, dtype=dtype)
        global_network.load_state_dict(
            payload["global_network_state"]
        )
        global_network.eval()

        field_config = ResidualMLPConfig(
            **dict(payload["field_network_config"])
        )
        field_normalizer = FeatureNormalizer(
            np.asarray(payload["field_input_mean"], float),
            np.asarray(payload["field_input_scale"], float),
        )
        field_network = build_residual_mlp(
            field_config,
            field_normalizer,
        ).to(device=device, dtype=dtype)
        field_network.load_state_dict(
            payload["field_network_state"]
        )
        field_network.eval()

        return cls(
            global_network,
            field_network,
            int(payload["n_ports"]),
            int(payload["n_cells"]),
            field_chunk_size=int(
                payload.get("field_chunk_size", 65536)
            ),
            global_output_mean=np.asarray(
                payload["global_output_mean"],
                float,
            ),
            global_output_scale=np.asarray(
                payload["global_output_scale"],
                float,
            ),
            field_output_mean=np.asarray(
                payload["field_output_mean"],
                float,
            ),
            field_output_scale=np.asarray(
                payload["field_output_scale"],
                float,
            ),
            field_density_prior_strength=float(
                payload.get(
                    "field_density_prior_strength",
                    0.9,
                )
            ),
            field_log_density_bounds=np.asarray(
                payload.get(
                    "field_log_density_bounds",
                    [-35.0, 35.0],
                ),
                float,
            ),
        )


__all__ = [
    "DecodedTensors",
    "SpatialDecodedTensors",
    "SpatialTensorDataset",
    "UnifiedSpatialTensorSurrogate",
    "TensorDataset",
    "UnifiedTensorSurrogate",
    "decode_physical_tensors",
    "decode_spatial_tensors",
    "decode_spatial_global_tensors",
    "decode_geometry_encoding",
    "encode_geometry",
    "encode_geometry_invariant",
    "pack_spatial_global_tensors",
    "pack_whitened_field_factors",
    "spatial_joule_density_prior",
    "unpack_whitened_field_factors",
    "generate_tensor_dataset",
    "pack_complex_symmetric",
    "pack_hermitian",
    "pack_tensors",
    "pack_spatial_tensors",
    "solve_port_truth_tensors",
    "solve_truth_tensors",
    "tensor_block_sizes",
    "tensor_output_dimension",
    "spatial_tensor_output_dimension",
    "normalize_cell_joule_tensors",
    "cell_joule_tensors_from_port_fields",
    "spatial_cell_features",
    "whiten_cell_joule_tensors",
    "unpack_complex_symmetric",
    "unpack_hermitian",
    "unpack_tensors",
    "unpack_spatial_tensors",
]
