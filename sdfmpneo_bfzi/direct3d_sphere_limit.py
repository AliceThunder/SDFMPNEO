"""Independent finite-frequency magnetic-diffusion benchmark: conducting sphere.

A uniform harmonic B field drives eddy currents in a nonmagnetic conducting
sphere. For exp(+i*omega*t), the solution is a spherical modified-Bessel
l=1 mode, with NO fitted paper data and NO BFZI geometry assumption.

We impose the exact tangential TOTAL A on an INSCRIBED tetrahedral polyhedron
and solve the full 3-D Nedelec A-phi PDE. The analytic field satisfies the
same bulk PDE on that polyhedron. Report volume L2 field error, Joule error
on the SAME polyhedron, and separate sphere-geometry volume truncation.
This is a deliberately independent D2 diffusion benchmark, NOT TEAM-7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import spherical_in

from .direct3d_aphi import MU0, _BARY4, TetraMesh, box_tetra_mesh, solve_aphi


@dataclass(frozen=True)
class HarmonicConductingSphere:
    radius_m: float
    conductivity_s_m: float
    frequency_hz: float
    imposed_B_T: tuple[complex, complex, complex] = (0., 0., 1.e-3)

    def __post_init__(self):
        params = (self.radius_m, self.conductivity_s_m, self.frequency_hz)
        b = np.asarray(self.imposed_B_T, complex)
        if (not all(np.isfinite(v) and v > 0 for v in params)
                or b.shape != (3,) or not np.isfinite(b).all()
                or not np.any(abs(b) > 0)):
            raise ValueError("Positive sphere radius, sigma, f and finite nonzero B required")
        object.__setattr__(self, "imposed_B_T", tuple(b))

    def _radial(self, r):
        """A(x)=h(r)*(B0 cross x), with h dimensionless, h' in 1/m."""
        r = np.asarray(r, float)
        if (np.any(r < 0) or not np.isfinite(r).all()):
            raise ValueError("Radius must be finite and nonnegative")
        omega = 2*np.pi*self.frequency_hz
        kappa = np.sqrt(1j*omega*MU0*self.conductivity_s_m)
        a = self.radius_m
        x = kappa*a
        i1a = spherical_in(1, x)
        i1pa = spherical_in(1, x, derivative=True)
        factor = 1.5*a/(2. + x*i1pa/i1a)
        dipole = a*a*(factor - a/2.)
        if not all(np.isfinite(v) for v in (i1a, factor, dipole)):
            raise ValueError("Spherical Bessel overflow: reduce the induction parameter")
        h = np.empty(r.shape, complex)
        dh = np.empty(r.shape, complex)
        inside = (r <= a) & (r > 0)
        zero = r == 0
        outside = r > a
        if np.any(inside):
            ri = r[inside]
            val = spherical_in(1, kappa*ri)
            derivative = spherical_in(1, kappa*ri, derivative=True)
            h[inside] = factor*val/(i1a*ri)
            dh[inside] = factor/i1a*(kappa*derivative/ri-val/ri**2)
        if np.any(zero):
            h[zero] = factor*kappa/(3*i1a)
            dh[zero] = 0.
        if np.any(outside):
            ro = r[outside]
            h[outside] = .5 + dipole/ro**3
            dh[outside] = -3*dipole/ro**4
        return h, dh

    def fields(self, points):
        """Exact complex A [T*m], B [T], E [V/m], J [A/m2], q [W/m3].

        All returned arrays have the same leading dimensions as points.
        Time convention: peak phasors exp(+i*omega*t).
        """
        xyz = np.asarray(points, float)
        if xyz.shape[-1:] != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Expected finite point coordinates (...,3)")
        r = np.linalg.norm(xyz, axis=-1)
        h, dh = self._radial(r)
        b0 = np.asarray(self.imposed_B_T, complex)
        a = h[..., None]*np.cross(b0, xyz)
        rr = np.where(r == 0, 1., r)
        b = ((2*h+r*dh)[..., None]*b0
             -(dh/rr)[..., None]*xyz*np.sum(xyz*b0, axis=-1)[..., None])
        e = -1j*(2*np.pi*self.frequency_hz)*a
        sigma = self.conductivity_s_m*(r <= self.radius_m)
        j = sigma[..., None]*e
        q = .5*sigma*np.sum(abs(e)**2, axis=-1)
        return {"A_T_m": a, "B_T": b, "E_V_m": e,
                "J_A_m2": j, "q_W_m3": q}

    def sphere_joule_power(self, radial_order: int = 96):
        """Exact untruncated spherical Joule integral (1D radial quadrature)."""
        if not 8 <= radial_order <= 512:
            raise ValueError("8 <= radial_order <= 512 required")
        nodes, weights = np.polynomial.legendre.leggauss(radial_order)
        radius = self.radius_m*(nodes+1)/2
        h, _ = self._radial(radius)
        omega = 2*np.pi*self.frequency_hz
        b2 = float(np.vdot(self.imposed_B_T, self.imposed_B_T).real)
        return float((4*np.pi/3)*self.conductivity_s_m*omega**2*b2
                     *(self.radius_m/2)*np.sum(weights*abs(h)**2*radius**4))


def inscribed_sphere_tetra_mesh(radius_m, conductivity_s_m, divisions=6):
    """Conforming inscribed polyhedron from a radial cube-to-ball map.

    Mesh-only approximation: face nodes lie on the exact sphere but each
    boundary triangle is planar. Its physical volume is LESS than 4*pi*a^3/3.
    """
    if type(divisions) is not int or divisions < 2:
        raise ValueError("divisions must be an integer >= 2")
    if not all(np.isfinite(v) and v > 0 for v in (radius_m, conductivity_s_m)):
        raise ValueError("Positive finite radius and conductivity required")
    cube = box_tetra_mesh([[-radius_m, radius_m]]*3,
                          [divisions]*3, lambda _: conductivity_s_m)
    nodes = cube.nodes
    norms = np.linalg.norm(nodes, axis=1)
    max_component = np.max(abs(nodes), axis=1)
    factor = np.divide(max_component, norms, out=np.ones_like(norms),
                       where=norms > 0)
    mapped = nodes*factor[:, None]
    return TetraMesh(mapped, cube.tetrahedra, cube.sigma)


def sphere_d2_benchmark(reference: HarmonicConductingSphere, divisions=6):
    """Solve 3D Maxwell-MQS and compare A-phi's actual E,B and volume Joule.

    Crucially E is the FEM physical total E, not a post-hoc exact-field
    substitution. The analytic total A is used ONLY on the boundary.
    """
    mesh = inscribed_sphere_tetra_mesh(
        reference.radius_m, reference.conductivity_s_m, divisions)
    def zero_source(x):
        return np.zeros_like(x)
    solved = solve_aphi(
        mesh, reference.frequency_hz, [zero_source],
        outer_bc="prescribed_total",
        boundary_total_potentials=[lambda p: reference.fields(p)["A_T_m"]])
    corners = mesh.nodes[mesh.tetrahedra]
    jacobian = (corners[:, 1:]-corners[:, 0, None, :]).transpose(0, 2, 1)
    volumes = abs(np.linalg.det(jacobian))/6
    gauss_pts = np.einsum("qi,nic->nqc", _BARY4, corners)
    e_exact = reference.fields(gauss_pts)["E_V_m"]
    e_fem = solved.electric_at_quadrature[0]
    e_exact_norm = np.sum(volumes[:, None]*abs(e_exact)**2)/4
    e_diff_norm = np.sum(volumes[:, None]*abs(e_fem-e_exact)**2)/4
    b_exact = reference.fields(corners.mean(axis=1))["B_T"]
    b_fem = solved.magnetic_secondary[0]  # total solved A, no source split
    b_exact_norm = np.sum(volumes[:, None]*abs(b_exact)**2)
    b_diff_norm = np.sum(volumes[:, None]*abs(b_fem-b_exact)**2)
    sigma = reference.conductivity_s_m
    exact_poly_joule = .5*sigma*e_exact_norm
    fem_joule = solved.volume_power([1.])
    sphere_joule = reference.sphere_joule_power()
    vol = float(volumes.sum())
    ball_vol = 4*np.pi/3*reference.radius_m**3
    return {
        "status": "Independent 3D finite-frequency physical diffusion check; D2 not PASS",
        "tetrahedra": int(len(mesh.tetrahedra)),
        "divisions": divisions,
        "conductivity_S_m": sigma,
        "frequency_hz": reference.frequency_hz,
        "radius_m": reference.radius_m,
        "skin_depth_m": float(np.sqrt(2/(2*np.pi*reference.frequency_hz*MU0*sigma))),
        "complex_E_volume_relative_L2": float(np.sqrt(e_diff_norm/e_exact_norm)),
        "complex_B_volume_relative_L2": float(np.sqrt(b_diff_norm/b_exact_norm)),
        "joule_fem_W": float(fem_joule),
        "joule_exact_same_polyhedron_W": float(exact_poly_joule),
        "joule_exact_whole_sphere_W": float(sphere_joule),
        "joule_same_polyhedron_relative_error":
            float((fem_joule-exact_poly_joule)/exact_poly_joule),
        "polyhedron_to_sphere_volume_ratio": float(vol/ball_vol),
        "note": ("The total A at the boundary comes from an exact independent "
                 "Bessel diffusion solution, not from the FEM. The mesh is an "
                 "inscribed polyhedron; polyhedron/whole-sphere Joule are distinct. "
                 "No TEAM7/BFZI reference certification."),
    }
