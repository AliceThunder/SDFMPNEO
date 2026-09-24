from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

from .em import DenseMQSTeacher, MQSConfig
from .scene import Scene, EPS0


def _equilibrated_dense_solve(matrix: np.ndarray, rhs: np.ndarray, iterations: int = 6):
    A = np.asarray(matrix, dtype=complex)
    b = np.asarray(rhs, dtype=complex)
    row = np.ones(A.shape[0], dtype=float)
    col = np.ones(A.shape[1], dtype=float)
    scaled = A.copy()
    for _ in range(iterations):
        rn = np.max(np.abs(scaled), axis=1)
        rs = 1.0 / np.sqrt(np.maximum(rn, 1e-300))
        row *= rs
        scaled = rs[:, None] * scaled
        cn = np.max(np.abs(scaled), axis=0)
        cs = 1.0 / np.sqrt(np.maximum(cn, 1e-300))
        col *= cs
        scaled = scaled * cs[None, :]
    y = solve(
        scaled,
        row[:, None] * b,
        assume_a="gen",
        check_finite=True,
    )
    return col[:, None] * y


@dataclass(frozen=True)
class MixedResult:
    impedance: np.ndarray
    current_coefficients: np.ndarray
    node_potential: np.ndarray
    node_charge: np.ndarray
    resistance_matrix: np.ndarray
    inductance_matrix: np.ndarray
    divergence_matrix: np.ndarray
    potential_matrix: np.ndarray
    port_injection: np.ndarray
    normalized_residual: float
    segment_coils: np.ndarray
    mode_segments: np.ndarray

    @property
    def n_ports(self) -> int:
        return self.impedance.shape[0]

    @property
    def mode_coefficients(self) -> np.ndarray:
        """Compatibility alias used by shared continuous-loss decoders."""
        return self.current_coefficients

    def port_power(self, currents) -> float:
        i = np.asarray(currents, dtype=complex)
        return float(
            0.5
            * np.real(
                np.vdot(i, self.impedance @ i)
            )
        )

    def conductor_power(self, currents) -> float:
        i = np.asarray(currents, dtype=complex)
        c = self.current_coefficients @ i
        return float(
            0.5
            * np.real(
                np.vdot(
                    c,
                    self.resistance_matrix @ c,
                )
            )
        )

    def coil_dissipation_matrices(self) -> np.ndarray:
        mode_coils = self.segment_coils[
            self.mode_segments
        ]
        n_coils = int(
            np.max(self.segment_coils)
        ) + 1
        out = np.zeros(
            (
                n_coils,
                self.n_ports,
                self.n_ports,
            ),
            dtype=complex,
        )
        transfer = self.current_coefficients
        for coil in range(n_coils):
            mask = (
                mode_coils == coil
            )
            if not np.any(mask):
                continue
            local_transfer = transfer[
                mask
            ]
            local_resistance = (
                self.resistance_matrix[
                    np.ix_(mask, mask)
                ]
            )
            matrix = (
                local_transfer.conj().T
                @ local_resistance
                @ local_transfer
            )
            out[coil] = 0.5 * (
                matrix
                + matrix.conj().T
            )
        return out

    def coil_power(self, currents) -> np.ndarray:
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.n_ports,
        ):
            raise ValueError(
                "currents has wrong shape"
            )
        channels = (
            self.coil_dissipation_matrices()
        )
        return np.asarray(
            [
                0.5
                * np.real(
                    np.vdot(
                        currents,
                        channel @ currents,
                    )
                )
                for channel in channels
            ],
            dtype=float,
        )

    def continuity_residual(
        self,
        currents,
        omega: float,
    ) -> float:
        i = np.asarray(currents, dtype=complex)
        c = self.current_coefficients @ i
        q = self.node_charge @ i
        rhs = self.port_injection @ i
        r = (
            self.divergence_matrix @ c
            + 1j * omega * q
            - rhs
        )
        scale = max(
            float(np.linalg.norm(rhs)),
            1.0,
        )
        return float(
            np.linalg.norm(r) / scale
        )


class DenseMixedConductorTeacher:
    """Dense current-potential-charge correctness backend.

    The formulation keeps charge as an independent state and is finite at DC;
    no 1/(j*omega) elimination is used. The magnetic current block is shared
    with DenseMQSTeacher while a nodal Coulomb potential block supplies scalar
    electric-potential and terminal-charging physics.
    """

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        config: MQSConfig | None = None,
        *,
        charge_self_radius_factor: float = 0.75,
    ):
        if charge_self_radius_factor <= 0:
            raise ValueError(
                "charge_self_radius_factor must be positive"
            )
        self.scene = scene
        self.frequency_hz = float(frequency_hz)
        if (
            not np.isfinite(self.frequency_hz)
            or self.frequency_hz < 0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        self.omega = (
            2.0
            * np.pi
            * self.frequency_hz
        )
        self.config = config or MQSConfig()
        self.charge_self_radius_factor = float(
            charge_self_radius_factor
        )
        self._mqs = DenseMQSTeacher(
            scene,
            frequency_hz,
            self.config,
        )

    def _topology(self, C: np.ndarray):
        segments = self._mqs._segments
        p = len(self.scene.coils)
        node_positions = []
        node_radii = []
        incidence_rows = []
        port_cols = []
        node_cursor = 0
        for ci, coil in enumerate(self.scene.coils):
            ids = [
                k
                for k, seg in enumerate(segments)
                if seg.coil == ci
            ]
            if not ids:
                raise RuntimeError(
                    "coil has no segments"
                )
            segs = [segments[k] for k in ids]
            pts = [
                segs[0].midpoint
                - 0.5
                * segs[0].length
                * segs[0].tangent
            ]
            pts.extend(
                seg.midpoint
                + 0.5
                * seg.length
                * seg.tangent
                for seg in segs
            )
            pts = np.asarray(
                pts,
                dtype=float,
            )
            nnode = len(pts)
            Bn = np.zeros(
                (nnode, len(segs)),
                dtype=float,
            )
            for local in range(len(segs)):
                Bn[local, local] += 1.0
                Bn[local + 1, local] -= 1.0
            incidence_rows.append(
                (node_cursor, Bn, ids)
            )
            node_positions.append(pts)
            node_radii.append(
                np.full(
                    nnode,
                    coil.geometry.equivalent_radius,
                )
            )
            bp = np.zeros(
                (nnode, p),
                dtype=float,
            )
            bp[0, ci] = 1.0
            bp[-1, ci] = -1.0
            port_cols.append(
                (node_cursor, bp)
            )
            node_cursor += nnode

        nnode_total = node_cursor
        D = np.zeros(
            (nnode_total, C.shape[1]),
            dtype=float,
        )
        B = np.zeros(
            (nnode_total, p),
            dtype=float,
        )
        for offset, Bn, ids in incidence_rows:
            rows = slice(
                offset,
                offset + Bn.shape[0],
            )
            D[rows] = (
                Bn
                @ C[np.asarray(ids)]
            )
        for offset, bp in port_cols:
            B[
                offset : offset + bp.shape[0]
            ] = bp

        # One orthonormal zero-mean potential/charge subspace per
        # electrically isolated conductor removes the common-potential gauge
        # without adding a diagonal regularizer.
        Q = np.zeros(
            (
                nnode_total,
                nnode_total - p,
            ),
            dtype=float,
        )
        row_cursor = 0
        col_cursor = 0
        for pts in node_positions:
            nloc = len(pts)
            raw = np.vstack(
                (
                    np.eye(nloc - 1),
                    -np.ones(
                        (1, nloc - 1)
                    ),
                )
            )
            qloc, _ = np.linalg.qr(
                raw,
                mode="reduced",
            )
            Q[
                row_cursor : row_cursor + nloc,
                col_cursor : col_cursor + nloc - 1,
            ] = qloc
            row_cursor += nloc
            col_cursor += nloc - 1

        return (
            D,
            B,
            Q,
            np.concatenate(node_positions),
            np.concatenate(node_radii),
        )

    def _potential_matrix(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
    ):
        eps = (
            EPS0
            * self.scene.medium.relative_permittivity
        )
        pref = 1.0 / (
            4.0
            * np.pi
            * eps
        )
        diff = (
            positions[:, None, :]
            - positions[None, :, :]
        )
        dist = np.linalg.norm(
            diff,
            axis=2,
        )
        soft = (
            self.charge_self_radius_factor
            * np.sqrt(
                radii[:, None]
                * radii[None, :]
            )
        )
        offdiag = ~np.eye(
            len(dist),
            dtype=bool,
        )
        if np.any(
            dist[offdiag]
            < 0.1 * soft[offdiag]
        ):
            raise ValueError(
                "coincident/overlapping terminal or charge nodes "
                "require an explicit junction/contact model"
            )
        dist = dist.copy()
        np.fill_diagonal(
            dist,
            np.diag(soft),
        )
        Phi = pref / dist
        return 0.5 * (
            Phi
            + Phi.T
        )

    def local_current_transfer(
        self,
        result: MixedResult,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self._mqs.local_current_transfer(
            result,
            coil_index,
            arc_fraction,
            xy,
        )

    def local_dissipation_matrix(
        self,
        result: MixedResult,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self._mqs.local_dissipation_matrix(
            result,
            coil_index,
            arc_fraction,
            xy,
        )

    def assemble(self):
        R, L, C, _ = self._mqs.assemble()
        D, Bp, Q, pos, radii = (
            self._topology(C)
        )
        Phi = self._potential_matrix(
            pos,
            radii,
        )
        return (
            R,
            L,
            D,
            Phi,
            Bp,
            Q,
        )

    def solve(self) -> MixedResult:
        R, L, D, Phi, B, Q = self.assemble()
        A = (
            R.astype(complex)
            + 1j
            * self.omega
            * L
        )
        Dr = Q.T @ D
        Br = Q.T @ B
        Phir = (
            Q.T
            @ Phi
            @ Q
        )
        m = A.shape[0]
        nr = Dr.shape[0]

        # Gauge-free mixed unknowns: modal current c, zero-mean scalar
        # potential phi_r, and zero-net-charge q_r. Charge is never
        # eliminated by dividing through j*omega.
        K = np.block(
            [
                [
                    A,
                    -Dr.T.astype(complex),
                    np.zeros(
                        (m, nr),
                        dtype=complex,
                    ),
                ],
                [
                    Dr.astype(complex),
                    np.zeros(
                        (nr, nr),
                        dtype=complex,
                    ),
                    1j
                    * self.omega
                    * np.eye(
                        nr,
                        dtype=complex,
                    ),
                ],
                [
                    np.zeros(
                        (nr, m),
                        dtype=complex,
                    ),
                    np.eye(
                        nr,
                        dtype=complex,
                    ),
                    -Phir.astype(complex),
                ],
            ]
        )
        rhs = np.vstack(
            (
                np.zeros(
                    (
                        m,
                        B.shape[1],
                    ),
                    dtype=complex,
                ),
                Br.astype(complex),
                np.zeros(
                    (
                        nr,
                        B.shape[1],
                    ),
                    dtype=complex,
                ),
            )
        )
        sol = _equilibrated_dense_solve(
            K,
            rhs,
        )
        c = sol[:m]
        phi_r = sol[m : m + nr]
        q_r = sol[m + nr :]
        phi = Q @ phi_r
        q = Q @ q_r
        Z = B.T @ phi

        residual = (
            K @ sol
            - rhs
        )
        scale = max(
            float(np.linalg.norm(rhs)),
            1.0,
        )
        eta = float(
            np.linalg.norm(residual)
            / scale
        )
        segment_coils = np.asarray(
            [
                segment.coil
                for segment
                in self._mqs._segments
            ],
            dtype=int,
        )
        mode_segments = np.empty(
            self._mqs._n_modes,
            dtype=int,
        )
        for segment_index, segment in enumerate(
            self._mqs._segments
        ):
            mode_segments[
                segment.mode_slice
            ] = segment_index
        return MixedResult(
            Z,
            c,
            phi,
            q,
            R,
            L,
            D,
            Phi,
            B,
            eta,
            segment_coils,
            mode_segments,
        )
