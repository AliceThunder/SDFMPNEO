"""Geometry-only electromagnetic tensor surrogate for the unified UWPT model.

The neural map is strictly static::

    geometry -> POD coefficients -> {Z_field, D_vol, H_1, ..., H_r}

Current phasors, wire resistance and thermal dynamics stay outside the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .unified_geometry import UnifiedUWPTGeometry

_SCHEMA_VERSION = 2
_SUPPORTED_SHAPES = ("circle", "rounded_square")


def _hermitian(value):
    a = np.asarray(value, complex)
    return 0.5 * (a + a.conj().T)


def _psd_clip(value, floor=0.0):
    a = _hermitian(value)
    w, v = np.linalg.eigh(a)
    w = np.maximum(w.real, float(floor))
    return _hermitian((v * w) @ v.conj().T)


def _upper_pairs(n):
    return [(i, j) for i in range(int(n)) for j in range(i, int(n))]


def pack_complex_symmetric(matrix):
    z = np.asarray(matrix, complex)
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ValueError("complex symmetric matrix must be square")
    z = 0.5 * (z + z.T)
    pairs = _upper_pairs(z.shape[0])
    return np.asarray(
        [z[i, j].real for i, j in pairs] + [z[i, j].imag for i, j in pairs],
        float,
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
    r = int(thermal_rank)
    return n * (n + 1) + n * n * (1 + r)


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
    modal = np.asarray([
        unpack_hermitian(p[start + j * h_size:start + (j + 1) * h_size], n)
        for j in range(r)
    ], complex)
    return z, d, modal


def _pose_features(pose):
    angles = np.asarray(pose.angles, float)
    return (
        np.asarray(pose.translation, float).tolist()
        + np.sin(angles).tolist()
        + np.cos(angles).tolist()
    )


def encode_geometry(geometry):
    """Fixed-width encoding for the supported production geometry family."""
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    features = []
    for coil in g.coils:
        if coil.shape not in _SUPPORTED_SHAPES:
            raise ValueError(
                f"production tensor surrogate supports {_SUPPORTED_SHAPES}; got {coil.shape!r}"
            )
        features.extend(1.0 if coil.shape == name else 0.0 for name in _SUPPORTED_SHAPES)
        features.extend([
            float(coil.turns), float(coil.outer_half_size), float(coil.pitch),
            float(coil.conductor_width), float(coil.conductor_thickness),
            0.0 if coil.corner_radius is None else float(coil.corner_radius),
        ])
        features.extend(_pose_features(coil.pose))
    for package in g.packages:
        features.extend(np.asarray(package.half_extent, float).tolist())
        # Keep package pose explicit.  It is redundant for today's default
        # family but prevents an independent package pose becoming a hidden
        # truth variable later.
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
    bounded = (vh * wh) @ vh.conj().T
    return _hermitian(root @ bounded @ root.conj().T)


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
    phi_min = np.asarray(phi_min, float).reshape(-1)
    phi_max = np.asarray(phi_max, float).reshape(-1)
    if phi_min.shape != phi_max.shape or np.any(phi_max < phi_min):
        raise ValueError("invalid thermal-mode bounds")
    raw = np.asarray(packed, float).reshape(-1)
    z_raw, d_raw, h_raw = unpack_tensors(raw, n_ports, len(phi_min))

    # Fixed safety layer: only dissipative blocks are corrected.  The reciprocal
    # reactive block is retained exactly from the network decode.
    d = _psd_clip(d_raw)
    x = np.asarray(z_raw.imag, float)
    d_out = _psd_clip(np.asarray(z_raw.real, float) - d)
    z = np.asarray((d + d_out).real, float) + 1j * x
    z = 0.5 * (z + z.T)
    modal = np.asarray([
        _modal_project_to_bounds(h_raw[j], d, phi_min[j], phi_max[j])
        for j in range(len(phi_min))
    ], complex)
    implied = _hermitian(z) - d

    n = int(n_ports)
    z_size, h_size, r = tensor_block_sizes(n, len(phi_min))
    corrected = pack_tensors(z, d, modal)
    zd_stop = z_size + h_size
    raw_zd = raw[:zd_stop]
    raw_h = raw[zd_stop:]
    corrected_zd = corrected[:zd_stop]
    corrected_h = corrected[zd_stop:]
    zd_correction = float(
        np.linalg.norm(corrected_zd - raw_zd)
        / max(np.linalg.norm(raw_zd), np.finfo(float).tiny)
    )
    h_correction = (
        float(
            np.linalg.norm(corrected_h - raw_h)
            / max(np.linalg.norm(raw_h), np.finfo(float).tiny)
        )
        if r else 0.0
    )
    return DecodedTensors(z, d, modal, implied, zd_correction, h_correction)


def _edge_loss_weights(background, context):
    sigma, _, _, _, _, _ = background.cell_properties(context, None, em=True)
    sigma = np.asarray(sigma, float)
    edge_loss = np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)
    return sigma, edge_loss


def _require_static_field_materials(background):
    for name, material in background.materials.items():
        if name in background.coil_materials:
            continue
        sigma = float(material.get("electrical_conductivity", 0.0))
        alpha = float(material.get("resistivity_temperature_coefficient", 0.0))
        if sigma > 0.0 and alpha != 0.0:
            raise ValueError(
                "geometry-only field tensors require temperature-independent non-wire EM materials; "
                f"{name!r} has nonzero resistivity_temperature_coefficient"
            )


def solve_truth_tensors(background, geometry):
    """Generate one offline tensor label from the full sparse Maxwell truth solve.

    Raw solver residual, reciprocity and PEC-domain power balance are measured
    before reciprocal projection.  Only labels that pass those audits should be
    admitted by the runtime Physics Gate.
    """
    _require_static_field_materials(background)
    context = background.geometry_context(geometry, assemble_thermal=True)
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
    max_linear_residual = float(np.max(np.linalg.norm(residual, axis=0) / bnorm))

    source = np.asarray(context.source_shape, float)
    reaction = -source.T @ X
    reaction_scale = max(float(np.linalg.norm(reaction)), np.finfo(float).tiny)
    reciprocity = float(np.linalg.norm(reaction - reaction.T) / reaction_scale)

    sigma, edge_loss = _edge_loss_weights(background, context)
    d = _hermitian(X.conj().T @ (edge_loss[:, None] * X))
    # The current background deletes tangential boundary DOFs, i.e. a finite PEC
    # truncation.  Hence outward Poynting flux is zero in this *discrete model*.
    # This check validates internal reaction/Joule consistency only; it is not
    # an open-domain certificate.
    raw_herm_z = _hermitian(reaction)
    closed_balance = float(
        np.linalg.norm(raw_herm_z - d)
        / max(np.linalg.norm(d), np.linalg.norm(raw_herm_z), np.finfo(float).tiny)
    )

    # Only after the raw audits do we create reciprocal training labels.
    z = 0.5 * (reaction + reaction.T)

    phi = np.asarray(background.thermal_basis, float)
    modal = []
    for j in range(phi.shape[1]):
        weighted_edge = np.asarray(
            background.edge_cell_hodge @ (sigma * phi[:, j])
        ).reshape(-1)
        modal.append(_hermitian(X.conj().T @ (weighted_edge[:, None] * X)))
    modal = np.asarray(modal, complex)

    min_d = float(np.min(np.linalg.eigvalsh(d)).real)
    implied = _hermitian(z) - d
    min_out = float(np.min(np.linalg.eigvalsh(implied)).real)
    phi_min = np.min(phi, axis=0)
    phi_max = np.max(phi, axis=0)
    loewner_violation = 0.0
    for j, h in enumerate(modal):
        low = np.min(np.linalg.eigvalsh(_hermitian(h - phi_min[j] * d))).real
        high = np.min(np.linalg.eigvalsh(_hermitian(phi_max[j] * d - h))).real
        scale = max(np.linalg.norm(d), np.linalg.norm(h), np.finfo(float).tiny)
        loewner_violation = max(
            loewner_violation,
            float(max(-low, -high, 0.0) / scale),
        )
    audit = {
        "max_linear_relative_residual": max_linear_residual,
        "reciprocity_relative_error": reciprocity,
        "minimum_d_vol_eigenvalue": min_d,
        "minimum_implied_outward_eigenvalue": min_out,
        "closed_boundary_power_balance_relative_error": closed_balance,
        "maximum_relative_loewner_violation": loewner_violation,
        "independent_outward_power_available": False,
        "boundary_model": "finite_pec_truncation_provisional",
    }
    return z, d, modal, audit


@dataclass
class TensorDataset:
    inputs: np.ndarray
    outputs: np.ndarray
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
    if background.thermal_basis is None:
        raise ValueError("thermal basis must be frozen before tensor labels are generated")
    inputs, outputs, audits = [], [], []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        z, d, modal, audit = solve_truth_tensors(background, geometry)
        inputs.append(encode_geometry(geometry))
        outputs.append(pack_tensors(z, d, modal))
        audits.append(audit)
        print(
            f"生成 Z_field / D_vol / H_j truth……{100.0 * (index + 1) / len(geometries):5.1f}%  "
            f"({index + 1}/{len(geometries)})",
            flush=True,
        )
    numeric_audit = {
        "maximum_linear_relative_residual": max(a["max_linear_relative_residual"] for a in audits),
        "maximum_reciprocity_relative_error": max(a["reciprocity_relative_error"] for a in audits),
        "minimum_d_vol_eigenvalue": min(a["minimum_d_vol_eigenvalue"] for a in audits),
        "minimum_implied_outward_eigenvalue": min(a["minimum_implied_outward_eigenvalue"] for a in audits),
        "maximum_closed_boundary_power_balance_relative_error": max(
            a["closed_boundary_power_balance_relative_error"] for a in audits
        ),
        "maximum_relative_loewner_violation": max(
            a["maximum_relative_loewner_violation"] for a in audits
        ),
        "independent_outward_power_available": 0.0,
    }
    return TensorDataset(
        np.asarray(inputs, float),
        np.asarray(outputs, float),
        _split_labels(len(geometries), seed),
        numeric_audit,
        len(background.coil_materials),
        background.thermal_rank,
    )


class UnifiedTensorSurrogate:
    def __init__(
        self,
        network,
        output_mean,
        output_scale,
        pod_basis,
        n_ports,
        phi_min,
        phi_max,
    ):
        self.network = network
        self.output_mean = np.asarray(output_mean, float).reshape(-1)
        self.output_scale = np.asarray(output_scale, float).reshape(-1)
        self.pod_basis = np.asarray(pod_basis, float)
        self.n_ports = int(n_ports)
        self.phi_min = np.asarray(phi_min, float).reshape(-1)
        self.phi_max = np.asarray(phi_max, float).reshape(-1)
        full = tensor_output_dimension(self.n_ports, len(self.phi_min))
        if self.output_mean.shape != (full,) or self.output_scale.shape != (full,):
            raise ValueError("output normalization dimensions differ from tensor schema")
        if self.pod_basis.ndim != 2 or self.pod_basis.shape[0] != full or self.pod_basis.shape[1] < 1:
            raise ValueError("invalid tensor POD basis")
        if self.network.config.output_dimension != self.pod_basis.shape[1]:
            raise ValueError("network output dimension does not match tensor POD rank")

    @property
    def thermal_rank(self):
        return len(self.phi_min)

    @property
    def pod_rank(self):
        return int(self.pod_basis.shape[1])

    def predict_from_encoded(self, encoded):
        import torch

        parameter = next(self.network.parameters())
        x = torch.as_tensor(
            np.asarray(encoded, float), dtype=parameter.dtype, device=parameter.device
        )
        with torch.no_grad():
            beta = self.network(x).detach().cpu().numpy().astype(float)
        normalized = self.pod_basis @ beta
        packed = self.output_mean + self.output_scale * normalized
        return decode_physical_tensors(packed, self.n_ports, self.phi_min, self.phi_max)

    def predict(self, geometry):
        return self.predict_from_encoded(encode_geometry(geometry))

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
            "phi_min": self.phi_min,
            "phi_max": self.phi_max,
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
            payload["phi_min"],
            payload["phi_max"],
        )


__all__ = [
    "DecodedTensors", "TensorDataset", "UnifiedTensorSurrogate",
    "decode_physical_tensors", "encode_geometry", "generate_tensor_dataset",
    "pack_complex_symmetric", "pack_hermitian", "pack_tensors",
    "solve_truth_tensors", "tensor_block_sizes", "tensor_output_dimension",
    "unpack_complex_symmetric", "unpack_hermitian", "unpack_tensors",
]
