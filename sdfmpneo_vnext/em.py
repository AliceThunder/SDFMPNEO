from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

from .basis import SectionBasis, polynomial_section_basis
from .scene import Scene


@dataclass(frozen=True)
class MQSConfig:
    segments_per_turn: int = 20
    min_segments: int = 16
    section_degree: int = 1
    radial_order: int = 4
    angular_order: int = 24
    line_order: int = 2
    self_softening_factor: float = 0.45

    def __post_init__(self):
        if self.segments_per_turn < 4 or self.min_segments < 4:
            raise ValueError("segment resolution is too small")
        if self.section_degree < 0 or self.line_order < 1:
            raise ValueError("invalid basis/quadrature order")
        if self.self_softening_factor <= 0:
            raise ValueError("self_softening_factor must be positive")


@dataclass
class _SegmentBlock:
    coil: int
    midpoint: np.ndarray
    tangent: np.ndarray
    length: float
    n1: np.ndarray
    n2: np.ndarray
    basis: SectionBasis
    mode_slice: slice


@dataclass(frozen=True)
class MQSResult:
    impedance: np.ndarray
    mode_coefficients: np.ndarray
    segment_multipliers: np.ndarray
    resistance_matrix: np.ndarray
    inductance_matrix: np.ndarray
    constraint_matrix: np.ndarray
    port_map: np.ndarray
    segment_coils: np.ndarray
    mode_segments: np.ndarray

    @property
    def n_ports(self) -> int:
        return self.impedance.shape[0]

    def port_voltage(self, currents) -> np.ndarray:
        i = np.asarray(currents, dtype=complex)
        if i.shape != (self.n_ports,):
            raise ValueError("currents has wrong shape")
        return self.impedance @ i

    def mode_current(self, currents) -> np.ndarray:
        i = np.asarray(currents, dtype=complex)
        if i.shape != (self.n_ports,):
            raise ValueError("currents has wrong shape")
        return self.mode_coefficients @ i

    def conductor_power(self, currents) -> float:
        c = self.mode_current(currents)
        return float(0.5 * np.real(np.vdot(c, self.resistance_matrix @ c)))

    def port_power(self, currents) -> float:
        i = np.asarray(currents, dtype=complex)
        return float(0.5 * np.real(np.vdot(i, self.impedance @ i)))

    def segment_power(self, currents) -> np.ndarray:
        c = self.mode_current(currents)
        modal = 0.5 * np.real(
            np.conj(c) * (self.resistance_matrix @ c)
        )
        out = np.zeros(len(self.segment_coils), dtype=float)
        np.add.at(out, self.mode_segments, modal)
        return out



class DenseMQSTeacher:
    """Dense finite-cross-section MQS Galerkin correctness backend.

    This is the first I1a backend: it enforces one work-conjugate current
    constraint per longitudinal segment while allowing zero-net-current
    cross-section modes to redistribute through magnetic coupling. It is
    intentionally independent of the legacy volume-grid solver.
    """

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        config: MQSConfig | None = None,
    ):
        if frequency_hz < 0 or not np.isfinite(frequency_hz):
            raise ValueError("frequency_hz must be finite and nonnegative")
        self.scene = scene
        self.frequency_hz = float(frequency_hz)
        self.omega = 2.0 * np.pi * self.frequency_hz
        self.config = config or MQSConfig()
        self._segments, self._n_modes = self._build_segments()

    def _build_segments(self):
        segments = []
        cursor = 0
        for ci, coil in enumerate(self.scene.coils):
            nseg = max(
                self.config.min_segments,
                int(np.ceil(self.config.segments_per_turn * coil.geometry.turns)),
            )
            poly = coil.geometry.polyline(nseg)
            basis = polynomial_section_basis(
                coil.geometry.conductor_width,
                coil.geometry.conductor_thickness,
                coil.geometry.cross_section_exponent,
                self.config.section_degree,
                self.config.radial_order,
                self.config.angular_order,
            )
            for s in range(nseg):
                sl = slice(cursor, cursor + basis.n_modes)
                segments.append(
                    _SegmentBlock(
                        ci,
                        poly.midpoints[s],
                        poly.tangents[s],
                        float(poly.lengths[s]),
                        poly.normal1[s],
                        poly.normal2[s],
                        basis,
                        sl,
                    )
                )
                cursor += basis.n_modes
        return segments, cursor

    @staticmethod
    def _line_rule(order: int):
        x, w = np.polynomial.legendre.leggauss(order)
        return 0.5 * x, 0.5 * w

    def _support_quadrature(self, seg: _SegmentBlock):
        z, wz = self._line_rule(self.config.line_order)
        q = seg.basis.quadrature
        pts = []
        weights = []
        values = []
        for zk, wk in zip(z, wz):
            center = seg.midpoint + zk * seg.length * seg.tangent
            p = (
                center
                + q.xy[:, 0, None] * seg.n1
                + q.xy[:, 1, None] * seg.n2
            )
            pts.append(p)
            weights.append(q.weights * (wk * seg.length))
            values.append(seg.basis.values)
        return (
            np.concatenate(pts),
            np.concatenate(weights),
            np.concatenate(values, axis=0),
        )

    def assemble(self):
        m = self._n_modes
        ns = len(self._segments)
        p = len(self.scene.coils)
        R = np.zeros((m, m), dtype=float)
        L = np.zeros((m, m), dtype=float)
        C = np.zeros((ns, m), dtype=float)
        B = np.zeros((ns, p), dtype=float)
        supports = []
        for si, seg in enumerate(self._segments):
            sl = seg.mode_slice
            sigma = self.scene.coils[seg.coil].material.conductivity
            R[sl, sl] = np.eye(sl.stop - sl.start) * (seg.length / sigma)
            C[si, sl] = seg.basis.moments
            B[si, seg.coil] = 1.0
            supports.append(self._support_quadrature(seg))

        mu = self.scene.medium.permeability
        pref = mu / (4.0 * np.pi)
        for i, seg_i in enumerate(self._segments):
            pi, wi, vi = supports[i]
            sli = seg_i.mode_slice
            for j in range(i + 1):
                seg_j = self._segments[j]
                pj, wj, vj = supports[j]
                slj = seg_j.mode_slice
                diff = pi[:, None, :] - pj[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                if i == j:
                    cell_scale_i = np.cbrt(np.maximum(wi, 1e-300))
                    cell_scale_j = np.cbrt(np.maximum(wj, 1e-300))
                    soft = self.config.self_softening_factor * (
                        cell_scale_i[:, None] + cell_scale_j[None, :]
                    )
                    dist = np.sqrt(dist2 + soft * soft)
                else:
                    dist = np.sqrt(np.maximum(dist2, 1e-30))
                K = (wi[:, None] * wj[None, :]) / dist
                block = (
                    pref
                    * float(np.dot(seg_i.tangent, seg_j.tangent))
                    * (vi.T @ K @ vj)
                )
                L[sli, slj] += block
                if i != j:
                    L[slj, sli] += block.T
        return R, L, C, B

    def solve(self) -> MQSResult:
        R, L, C, B = self.assemble()
        A = R.astype(complex) + 1j * self.omega * L
        m, ns = A.shape[0], C.shape[0]
        K = np.block(
            [
                [A, -C.T.astype(complex)],
                [C.astype(complex), np.zeros((ns, ns), dtype=complex)],
            ]
        )
        rhs = np.vstack(
            (
                np.zeros((m, B.shape[1]), dtype=complex),
                B.astype(complex),
            )
        )
        sol = solve(K, rhs, assume_a="gen", check_finite=True)
        coeff = sol[:m]
        lam = sol[m:]
        Z = B.T @ lam
        segment_coils = np.array(
            [s.coil for s in self._segments],
            dtype=int,
        )
        mode_segments = np.empty(
            self._n_modes,
            dtype=int,
        )
        for segment_index, segment in enumerate(
            self._segments
        ):
            mode_segments[
                segment.mode_slice
            ] = segment_index
        return MQSResult(
            Z,
            coeff,
            lam,
            R,
            L,
            C,
            B,
            segment_coils,
            mode_segments,
        )
