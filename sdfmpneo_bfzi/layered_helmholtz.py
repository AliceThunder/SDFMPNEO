"""Coaxial circular-sheet Helmholtz reference; NOT a BFZI/SE(3) solver.

Spectral A_phi Green function of three laterally unbounded planar layers,
with source current sheets at z=0 and z=d. Time convention exp(+1j*omega*t).
See Shafiei et al., IEEE Access 13 (2025), Eq. (3),(10),(20)-(31).
All permeability values are mu0; copper and enclosure effects are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import j1

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class LayeredCoaxialCase:
    tx_radii_m: tuple[float, ...]
    rx_radii_m: tuple[float, ...]
    gap_m: float
    frequency_hz: float
    conductivity_s_m: tuple[float, float, float]
    eps_r: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        if not self.tx_radii_m or not self.rx_radii_m:
            raise ValueError("Both coaxial ports need at least one circular turn")
        if any(not np.isfinite(v) or v <= 0 for v in self.tx_radii_m + self.rx_radii_m):
            raise ValueError("Radii must be finite positive metres")
        if not np.isfinite(self.gap_m) or self.gap_m <= 0:
            raise ValueError("Gap must be finite and positive")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("Frequency must be finite and positive")
        if len(self.conductivity_s_m) != 3 or any(not np.isfinite(s) or s < 0 for s in self.conductivity_s_m):
            raise ValueError("Supply three finite nonnegative conductivities")
        if len(self.eps_r) != 3 or any(not np.isfinite(e) or e <= 0 for e in self.eps_r):
            raise ValueError("Supply three finite positive relative permittivities")


def propagation(k: float, omega: float, sigma: float, er: float, *, include_displacement: bool):
    """u_j = sqrt(k**2 + i*w*mu0*sigma - w**2*mu0*eps0*er).

    The outgoing (+time) branch has Re(u)>=0 and Im(u)>=0 when Re(u)=0.
    """
    dielectric = omega**2 * MU0 * EPS0 * er if include_displacement else 0.0
    return np.sqrt(complex(k * k - dielectric, omega * MU0 * sigma))


def interface_green(k: float, case: LayeredCoaxialCase, *, include_displacement: bool = False):
    """The 2x2 interface trace Green matrix in H/m units.

    From the jump -[mu^-1 d_z A] = K_phi at z=0,d. Reflection-
    transmission expression avoids exponentially growing transfer matrices.
    """
    omega = 2 * np.pi * case.frequency_hz
    u1, u2, u3 = (
        propagation(k, omega, sig, er, include_displacement=include_displacement)
        for sig, er in zip(case.conductivity_s_m, case.eps_r)
    )
    q1, q2, q3 = u1 / MU0, u2 / MU0, u3 / MU0
    t = np.exp(-u2 * case.gap_m)
    denominator = (q2 + q1) * (q2 + q3) - (q2 - q1) * (q2 - q3) * t * t
    a = ((q2 + q3) + (q2 - q3) * t * t) / denominator
    b = 2 * q2 * t / denominator
    c = ((q2 + q1) + (q2 - q1) * t * t) / denominator
    return a, b, c


def incremental_impedance(
    case: LayeredCoaxialCase,
    *,
    k_max: float = 2500.0,
    rtol: float = 3e-7,
    include_displacement: bool = False,
):
    """Complex 2-port delta Z (ohm), relative to identical loops in free space.

    Does not give total self Z: zero-radius filament self inductance diverges.
    For MQS the free-space baseline is purely reactive and Re(delta Z)
    is the water-induced dissipation; for full-wave it also subtracts vacuum
    radiation and is NOT, by itself, a seawater-only positive power budget.
    Truncation at k_max is explicit; compare k_max and 2*k_max before use.
    """
    if not np.isfinite(k_max) or k_max <= 0 or not (0 < rtol < 0.1):
        raise ValueError("k_max>0 and 0<rtol<0.1 required")
    omega = 2 * np.pi * case.frequency_hz
    rt = np.asarray(case.tx_radii_m, dtype=float)
    rr = np.asarray(case.rx_radii_m, dtype=float)

    def f(k: float):
        a, b, c = interface_green(k, case, include_displacement=include_displacement)
        u0 = propagation(k, omega, 0, 1.0, include_displacement=include_displacement)
        vacuum_self = MU0 / (2 * u0)
        vacuum_mutual = vacuum_self * np.exp(-u0 * case.gap_m)
        st = np.sum(rt * j1(k * rt))
        sr = np.sum(rr * j1(k * rr))
        prefactor = 2j * np.pi * omega * k
        return prefactor * np.array([
            st * st * (a - vacuum_self),
            st * sr * (b - vacuum_mutual),
            sr * sr * (c - vacuum_self),
        ])

    # The full-wave vacuum square-root branch is nonanalytic at k0.
    # Splitting there avoids passing an interface singularity to quad_vec.
    k0 = omega * np.sqrt(MU0 * EPS0) if include_displacement else 0.0
    intervals = [(0.0, k_max)]
    if 0.0 < k0 < k_max:
        intervals = [(0.0, k0), (k0, k_max)]
    vals = np.zeros(3, dtype=complex)
    for lo, hi in intervals:
        part, _error = quad_vec(f, lo, hi, epsabs=1e-12, epsrel=rtol, limit=600)
        vals += part
    return np.array([[vals[0], vals[1]], [vals[1], vals[2]]])


def paper_case(
    scenario: str, *, frequency_hz: float = 85000.0,
    gap_m: float = 0.10, sigma_seawater: float = 4.0,
    turns_tx: int = 15, turns_rx: int = 17,
    pitch_m: float = 0.0025,
):
    """Shafiei Table 1 seed: 1mm gap + 1.5mm wire => 2.5mm centre pitch.

    This is the paper's stack of ideal circular loops, NOT the continuous
    BFZI winding. Wire cross-section and proximity loss are excluded.
    """
    if not (1 <= turns_tx <= 1000 and 1 <= turns_rx <= 1000) or pitch_m <= 0:
        raise ValueError("Invalid turn count/pitch")
    rt = tuple(0.170 / 2 + pitch_m * n for n in range(turns_tx))
    rr = tuple(0.165 / 2 + pitch_m * n for n in range(turns_rx))
    s = sigma_seawater
    if scenario == "infinite":
        sigmas = (s, s, s)
        er = (81.0, 81.0, 81.0)
    elif scenario == "semifinite":
        sigmas = (0.0, s, s)
        er = (1.0, 81.0, 81.0)
    elif scenario == "finite":
        sigmas = (0.0, s, 0.0)
        er = (1.0, 81.0, 1.0)
    else:
        raise ValueError("scenario must be infinite/semifinite/finite")
    return LayeredCoaxialCase(rt, rr, gap_m, frequency_hz, sigmas, er)
