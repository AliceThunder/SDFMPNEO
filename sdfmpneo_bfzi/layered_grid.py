"""Fast spatial Hankel reconstruction for the published coaxial three-layer D1 benchmark.

This is not an arbitrary-pose BFZI solver.  The circular source sheets at
z=0 and z=gap are zero-radius idealizations and do not define finite copper
self impedance or an unregularized global electric-field maximum.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1
from .layered_helmholtz import LayeredCoaxialCase, MU0, EPS0


def field_grid(
    case: LayeredCoaxialCase,
    radii_m,
    heights_m,
    currents: tuple[complex, complex] = (1., 0.),
    *,
    k_max: float = 4000.,
    panel_width: float = 8.,
    panel_order: int = 8,
    include_displacement: bool = False,
):
    """Vectorized cylindrical (r,z) spectral E/B/Joule fields.

    Independent r and z axes; returned fields have shape (len(radii_m),
    len(heights_m)). The prescribed k_max is a *spatial resolution* limit
    near filamentary current sheets; it must be increased for convergence.
    Sampling at z=0,d is disallowed because B_r has a distributional jump.
    No finite wire radius is modeled: maxima at ring centres are undefined.
    """
    r = np.asarray(radii_m, dtype=float)
    z = np.asarray(heights_m, dtype=float)
    if r.ndim != 1 or z.ndim != 1 or not r.size or not z.size:
        raise ValueError("Expected nonempty 1D radial and axial axes")
    if not np.isfinite(r).all() or (r < 0).any() or not np.isfinite(z).all():
        raise ValueError("All r>=0 and z must be finite")
    if np.any((z == 0) | (z == case.gap_m)):
        raise ValueError("Source planes z=0,d require one-sided traces")
    if len(currents) != 2 or not all(np.isfinite(t) for t in currents):
        raise ValueError("Two finite complex peak currents are required")
    if not np.isfinite(k_max) or k_max <= 0 or not np.isfinite(panel_width) or panel_width <= 0:
        raise ValueError("Positive k_max and panel_width required")
    if not 2 <= panel_order <= 64:
        raise ValueError("2<=panel_order<=64 required")
    omega = 2 * np.pi * case.frequency_hz
    # Split dielectric light-line square root singularities for zero-sigma layers.
    breaks = [0.]
    if include_displacement:
        for sig, er in zip(case.conductivity_s_m, case.eps_r):
            if sig == 0:
                line = omega * np.sqrt(MU0 * EPS0 * er)
                if 0 < line < k_max:
                    breaks.append(float(line))
    breaks.append(float(k_max))
    breaks = sorted(set(breaks))
    xg, wg = np.polynomial.legendre.leggauss(panel_order)
    nodes = []
    weights = []
    for start, stop in zip(breaks[:-1], breaks[1:]):
        npan = int(np.ceil((stop-start)/panel_width))
        edges = np.linspace(start, stop, npan+1)
        centers = (edges[:-1] + edges[1:])*.5
        half = (edges[1:] - edges[:-1])*.5
        nodes.append((centers[:,None] + half[:,None]*xg[None,:]).ravel())
        weights.append((half[:,None]*wg[None,:]).ravel())
    k = np.concatenate(nodes)
    w = np.concatenate(weights)
    ux = []
    for sig, er in zip(case.conductivity_s_m, case.eps_r):
        diel = omega**2 * MU0 * EPS0 * er if include_displacement else 0.
        ux.append(np.sqrt(k*k - diel + 1j*omega*MU0*sig))
    u1, u2, u3 = ux
    q1, q2, q3 = (u/MU0 for u in ux)
    t = np.exp(-u2*case.gap_m)
    denom = (q2+q1)*(q2+q3)-(q2-q1)*(q2-q3)*t*t
    g00 = ((q2+q3)+(q2-q3)*t*t)/denom
    g0d = 2*q2*t/denom
    gdd = ((q2+q1)+(q2-q1)*t*t)/denom
    rt = np.asarray(case.tx_radii_m)
    rr = np.asarray(case.rx_radii_m)
    st = np.sum(rt[:,None]*j1(rt[:,None]*k[None,:]),axis=0)*currents[0]
    sr = np.sum(rr[:,None]*j1(rr[:,None]*k[None,:]),axis=0)*currents[1]
    a0 = g00*st+g0d*sr
    ad = g0d*st+gdd*sr
    amp = np.empty((len(z),len(k)),dtype=complex)
    deriv = np.empty_like(amp)
    below = z < 0
    middle = (z > 0) & (z < case.gap_m)
    above = z > case.gap_m
    if np.any(below):
        amp[below] = np.exp(z[below,None]*u1[None,:])*a0[None,:]
        deriv[below] = amp[below]*u1[None,:]
    if np.any(above):
        amp[above] = np.exp(-(z[above,None]-case.gap_m)*u3[None,:])*ad[None,:]
        deriv[above] = -amp[above]*u3[None,:]
    if np.any(middle):
        zz = z[middle,None]
        d = case.gap_m
        den = -np.expm1(-2*u2*d)
        low = np.exp(-zz*u2)*(-np.expm1(-2*u2*(d-zz)))/den
        high = np.exp(-(d-zz)*u2)*(-np.expm1(-2*u2*zz))/den
        dlow = -u2*np.exp(-zz*u2)*(1+np.exp(-2*u2*(d-zz)))/den
        dhigh = u2*np.exp(-(d-zz)*u2)*(1+np.exp(-2*u2*zz))/den
        amp[middle] = low*a0 + high*ad
        deriv[middle] = dlow*a0+dhigh*ad
    x = r[:,None]*k[None,:]
    j = j1(x)
    ak = w*k
    A = j @ (amp*ak[None,:]).T
    Br = j @ (-deriv*ak[None,:]).T
    Bz = j0(x) @ (amp*(w*k*k)[None,:]).T
    E = -1j*omega*A
    sigma = np.select([below,middle,above],case.conductivity_s_m)
    J = sigma[None,:]*E
    q = .5*sigma[None,:]*abs(E)**2
    return {"A_phi_T_m": A, "E_phi_V_m": E, "B_r_T": Br,
            "B_z_T": Bz, "J_phi_A_m2": J, "q_W_m3": q}


def secondary_field_grid(
    case: LayeredCoaxialCase,
    radii_m,
    heights_m,
    currents: tuple[complex, complex] = (1., 0.),
    **grid_options,
):
    """Seawater-induced *secondary* (reaction) field on a cylindrical grid.

    Subtracts the identical source in vacuum, including permittivity=eps0
    when displacement is enabled. The actual seawater Joule heating is
    0.5*sigma*abs(E_total)**2, NEVER 0.5*sigma*abs(E_secondary)**2.
    This diagnostic is intended for Shafiei Table 2 E-field comparisons.
    """
    vacuum = LayeredCoaxialCase(
        case.tx_radii_m, case.rx_radii_m, case.gap_m,
        case.frequency_hz, (0., 0., 0.), (1., 1., 1.)
    )
    total = field_grid(case, radii_m, heights_m, currents, **grid_options)
    background = field_grid(vacuum, radii_m, heights_m, currents, **grid_options)
    return {
        "A_phi_secondary_T_m": total["A_phi_T_m"] - background["A_phi_T_m"],
        "E_phi_secondary_V_m": total["E_phi_V_m"] - background["E_phi_V_m"],
        "B_r_secondary_T": total["B_r_T"] - background["B_r_T"],
        "B_z_secondary_T": total["B_z_T"] - background["B_z_T"],
    }
