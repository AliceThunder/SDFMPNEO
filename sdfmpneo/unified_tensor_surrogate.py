"""Geometry-only EM tensor surrogate with geometry-aware thermal modal labels."""
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


def tensor_output_dimension(n_ports, thermal_rank):
    n = int(n_ports)
    return n * (n + 1) + n * n * (1 + int(thermal_rank))


def tensor_block_sizes(n_ports, thermal_rank):
    n = int(n_ports)
    return n * (n + 1), n * n, int(thermal_rank)


def pack_tensors(z_field, d_vol, modal_h):
    modal = np.asarray(modal_h, complex)
    if modal.ndim != 3:
        raise ValueError("modal Joule tensors must have shape (rank, ports, ports)")
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
    modal = np.asarray(
        [
            unpack_hermitian(p[start + j * h_size:start + (j + 1) * h_size], n)
            for j in range(r)
        ],
        complex,
    )
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

    modal = np.asarray(
        [
            _modal_project_to_bounds(h_raw[j], d, phi_min[j], phi_max[j])
            for j in range(len(phi_min))
        ],
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
        or final_mismatch > 1e-9
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
        return np.maximum(np.asarray(heat, float), 0.0)

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


class UnifiedSpatialTensorSurrogate:
    def __init__(
        self,
        network,
        output_mean,
        output_scale,
        pod_basis,
        n_ports,
        n_cells,
    ):
        self.network = network
        self.output_mean = np.asarray(output_mean, float).reshape(-1)
        self.output_scale = np.asarray(output_scale, float).reshape(-1)
        self.pod_basis = np.asarray(pod_basis, float)
        self.n_ports = int(n_ports)
        self.n_cells = int(n_cells)
        full = spatial_tensor_output_dimension(
            self.n_ports,
            self.n_cells,
        )
        if (
            self.output_mean.shape != (full,)
            or self.output_scale.shape != (full,)
        ):
            raise ValueError(
                "spatial output normalization dimensions differ from schema"
            )
        if (
            self.pod_basis.ndim != 2
            or self.pod_basis.shape[0] != full
            or self.pod_basis.shape[1] < 1
        ):
            raise ValueError("invalid spatial tensor POD basis")
        if self.network.config.output_dimension != self.pod_basis.shape[1]:
            raise ValueError(
                "network output dimension does not match spatial POD rank"
            )

    @property
    def thermal_rank(self):
        return 0

    @property
    def pod_rank(self):
        return int(self.pod_basis.shape[1])

    def predict_from_encoded(self, encoded):
        import torch

        parameter = next(self.network.parameters())
        x = torch.as_tensor(
            np.asarray(encoded, float),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.no_grad():
            beta = (
                self.network(x)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
        packed = (
            self.output_mean
            + self.output_scale * (self.pod_basis @ beta)
        )
        return decode_spatial_tensors(
            packed,
            self.n_ports,
            self.n_cells,
        )

    def predict(self, geometry):
        return self.predict_from_encoded(encode_geometry(geometry))

    def checkpoint(self):
        parameter = next(self.network.parameters())
        normalizer = FeatureNormalizer(
            self.network.input_mean.detach().cpu().numpy(),
            self.network.input_scale.detach().cpu().numpy(),
        )
        return {
            "schema_version": _SPATIAL_SCHEMA_VERSION,
            "representation": "cellwise_joule_tensor_v1",
            "network_config": self.network.config.to_dict(),
            "input_mean": normalizer.mean,
            "input_scale": normalizer.scale,
            "output_mean": self.output_mean,
            "output_scale": self.output_scale,
            "pod_basis": self.pod_basis,
            "n_ports": self.n_ports,
            "n_cells": self.n_cells,
            "network_state": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            },
            "dtype": str(parameter.dtype).replace("torch.", ""),
        }

    @classmethod
    def from_checkpoint(cls, payload, device="cpu"):
        import torch

        if (
            int(payload.get("schema_version", -1))
            != _SPATIAL_SCHEMA_VERSION
            or payload.get("representation")
            != "cellwise_joule_tensor_v1"
        ):
            raise ValueError(
                "unsupported spatial tensor-surrogate artifact version"
            )
        config = ResidualMLPConfig(
            **dict(payload["network_config"])
        )
        normalizer = FeatureNormalizer(
            np.asarray(payload["input_mean"], float),
            np.asarray(payload["input_scale"], float),
        )
        network = build_residual_mlp(config, normalizer)
        dtype = (
            torch.float32
            if payload.get("dtype") == "float32"
            else torch.float64
        )
        network = network.to(device=device, dtype=dtype)
        network.load_state_dict(payload["network_state"])
        network.eval()
        return cls(
            network,
            payload["output_mean"],
            payload["output_scale"],
            payload["pod_basis"],
            int(payload["n_ports"]),
            int(payload["n_cells"]),
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
    "encode_geometry",
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
    "unpack_complex_symmetric",
    "unpack_hermitian",
    "unpack_tensors",
    "unpack_spatial_tensors",
]
