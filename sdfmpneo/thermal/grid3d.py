from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class RectilinearThermalFV3D:
    """Cell-centred orthogonal finite-volume thermal operator.

    The operator uses exact cell volumes and two-half-cell thermal resistances
    across internal faces. Heterogeneous conductor/package/seawater interfaces
    are therefore coupled by flux continuity rather than by arithmetic
    averaging. Non-homogeneous physical boundary data should be handled through
    the boundary lifting used by the reduced thermal theory; this class assembles
    the homogeneous deviation operator.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def build(cls, x, y, z) -> "RectilinearThermalFV3D":
        x = np.asarray(tuple(x), dtype=float)
        y = np.asarray(tuple(y), dtype=float)
        z = np.asarray(tuple(z), dtype=float)
        if min(x.size, y.size, z.size) < 2:
            raise ValueError("Each coordinate axis requires at least two nodes")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0) or np.any(np.diff(z) <= 0):
            raise ValueError("Coordinates must be strictly increasing")
        return cls(x=x, y=y, z=z)

    @property
    def shape_cells(self) -> tuple[int, int, int]:
        return (self.x.size - 1, self.y.size - 1, self.z.size - 1)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape_cells))

    def assemble(
        self,
        rho_cp,
        conductivity,
        *,
        dirichlet_outer: bool = True,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Return thermal mass M and conduction K.

        `rho_cp` is volumetric heat capacity and `conductivity` is thermal
        conductivity, both cellwise. If `dirichlet_outer` is true, the six outer
        surfaces represent a homogeneous Dirichlet condition on the deviation
        temperature. The boundary contribution uses the exact half-cell thermal
        resistance. No artificial convection coefficient is introduced.
        """

        rho_cp = np.asarray(rho_cp, dtype=float)
        conductivity = np.asarray(conductivity, dtype=float)
        if rho_cp.shape != self.shape_cells or conductivity.shape != self.shape_cells:
            raise ValueError("cell field shape mismatch")
        if np.any(rho_cp <= 0) or np.any(conductivity <= 0):
            raise ValueError("rho_cp and conductivity must be positive")

        dx, dy, dz = np.diff(self.x), np.diff(self.y), np.diff(self.z)
        nx, ny, nz = self.shape_cells
        n = self.n_cells

        def idx(i: int, j: int, k: int) -> int:
            return (i * ny + j) * nz + k

        volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
        M = sp.diags((rho_cp * volume).ravel(), format="csr")

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        diagonal = np.zeros(n, dtype=float)

        def couple(p: int, q: int, conductance: float) -> None:
            diagonal[p] += conductance
            diagonal[q] += conductance
            rows.extend([p, q])
            cols.extend([q, p])
            data.extend([-conductance, -conductance])

        # x-normal internal interfaces
        for i in range(nx - 1):
            for j in range(ny):
                for k in range(nz):
                    area = dy[j] * dz[k]
                    resistance = (
                        0.5 * dx[i] / conductivity[i, j, k]
                        + 0.5 * dx[i + 1] / conductivity[i + 1, j, k]
                    )
                    couple(idx(i, j, k), idx(i + 1, j, k), area / resistance)

        # y-normal internal interfaces
        for i in range(nx):
            for j in range(ny - 1):
                for k in range(nz):
                    area = dx[i] * dz[k]
                    resistance = (
                        0.5 * dy[j] / conductivity[i, j, k]
                        + 0.5 * dy[j + 1] / conductivity[i, j + 1, k]
                    )
                    couple(idx(i, j, k), idx(i, j + 1, k), area / resistance)

        # z-normal internal interfaces
        for i in range(nx):
            for j in range(ny):
                for k in range(nz - 1):
                    area = dx[i] * dy[j]
                    resistance = (
                        0.5 * dz[k] / conductivity[i, j, k]
                        + 0.5 * dz[k + 1] / conductivity[i, j, k + 1]
                    )
                    couple(idx(i, j, k), idx(i, j, k + 1), area / resistance)

        if dirichlet_outer:
            # Six boundary surfaces: exact centre-to-boundary half-cell resistance.
            for j in range(ny):
                for k in range(nz):
                    diagonal[idx(0, j, k)] += (
                        2.0 * conductivity[0, j, k] * dy[j] * dz[k] / dx[0]
                    )
                    diagonal[idx(nx - 1, j, k)] += (
                        2.0 * conductivity[nx - 1, j, k] * dy[j] * dz[k] / dx[nx - 1]
                    )
            for i in range(nx):
                for k in range(nz):
                    diagonal[idx(i, 0, k)] += (
                        2.0 * conductivity[i, 0, k] * dx[i] * dz[k] / dy[0]
                    )
                    diagonal[idx(i, ny - 1, k)] += (
                        2.0 * conductivity[i, ny - 1, k] * dx[i] * dz[k] / dy[ny - 1]
                    )
            for i in range(nx):
                for j in range(ny):
                    diagonal[idx(i, j, 0)] += (
                        2.0 * conductivity[i, j, 0] * dx[i] * dy[j] / dz[0]
                    )
                    diagonal[idx(i, j, nz - 1)] += (
                        2.0 * conductivity[i, j, nz - 1] * dx[i] * dy[j] / dz[nz - 1]
                    )

        rows.extend(range(n))
        cols.extend(range(n))
        data.extend(diagonal.tolist())
        K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        return M, K
