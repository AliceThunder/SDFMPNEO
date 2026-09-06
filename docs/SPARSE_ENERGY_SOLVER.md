# Sparse high-contrast A-psi solve and energy certificate

This document defines the current sparse electromagnetic solve contract for the tetrahedral SDF-MPNEO path. The objective is not merely to obtain a numerical field solution, but to make the algebraic solve error an explicit, deterministic contribution to the final `Z/R/L/M` and electromagnetic-to-thermal error budget.

## 1. Physical sparse operator

After the magnetic tree-cotree gauge and conducting scalar-potential gauge are applied, the reciprocal magnetoquasistatic system has the form

```text
A x = b,
A = K + i D,
```

where

```text
K >= 0  : gauge-reduced magnetic curl-curl contribution,
D >= 0  : omega times the conductive electric-field contribution.
```

For real reciprocal material laws, `K` and `D` are real symmetric positive-semidefinite matrices. The nonlinear tetrahedral implementation keeps the following objects sparse:

```text
M_sigma(a),
A(a),
dA/da_k,
H_loss,j(a),
dH_loss,j/da_k.
```

Dense conversion is retained only as a verification-scale compatibility wrapper for the older reduced-order implementation.

## 2. Contrast-independent physical energy norm

Define

```text
H = K + D,
||x||_H^2 = x^H H x.
```

For any complex state `x`, let

```text
k = x^H K x >= 0,
d = x^H D x >= 0.
```

Then

```text
|x^H A x|
= |k + i d|
= sqrt(k^2+d^2)
>= (k+d)/sqrt(2)
= ||x||_H^2/sqrt(2).
```

Therefore the physical energy coercivity constant obeys the structural bound

```text
beta_H >= 1/sqrt(2).
```

The important point is that this lower bound contains **no copper/seawater conductivity ratio**. Raising the copper conductivity relative to seawater changes `D`, but does not degrade the proven constant in the `H` norm.

For an approximate field `x_h`, residual

```text
r = b - A x_h,
```

and error `e=x-x_h`, the dual energy norm is

```text
||r||_(H^-1) = sqrt(r^H H^-1 r),
```

and

```text
||e||_H
<= ||r||_(H^-1)/beta_H
<= sqrt(2) ||r||_(H^-1).
```

This is the recommended algebraic field-solve certificate.

## 3. No independent iterative tolerance

The scientific interface does not expose an arbitrary Krylov tolerance.

If the requested electromagnetic state accuracy is

```text
epsilon_H,
```

the solve is accepted only when the explicitly recomputed true residual satisfies

```text
sqrt(2) ||b-A x_h||_(H^-1) <= epsilon_H.
```

Internal iterative status is not a certificate. A solver can report convergence and still fail the physical residual test; conversely, only the a-posteriori inequality decides acceptance.

## 4. Current sparse numerical path

The current implementation provides two levels.

### 4.1 Energy-preconditioned BiCGSTAB

`H^-1` is used as a left preconditioner. A callback evaluates the true physical dual residual and stops only when the requested energy-state error has been certified.

The iteration work bound is the electromagnetic coordinate dimension, not a fitted iteration count.

### 4.2 Deterministic complete sparse-LU fallback

If the short-recurrence iteration does not reach the requested certificate, the current correctness baseline falls back to a complete sparse LU factorization of `A`.

There are no ILU drop tolerances, empirical fill factors, or relaxed error thresholds. The final solution is again accepted only from the original-matrix residual certificate.

This fallback proves that the sparse/high-contrast physical chain is executable, but it is **not** claimed to be the final memory-scalable solver for million-degree-of-freedom meshes.

## 5. Multiport output certificate

For closed unit-current port cochains `b_i`,

```text
Z_ij = i omega b_i^T x_j.
```

The source cochains are real, so energy Cauchy-Schwarz gives

```text
|Delta Z_ij|
<= omega ||b_i||_(H^-1) ||Delta x_j||_H.
```

Hence a user-facing requested complex-impedance element error `epsilon_Z` determines the field solve requirement automatically:

```text
epsilon_H
= epsilon_Z / [omega max_i ||b_i||_(H^-1)].
```

No separate linear-solver tolerance or external minimum singular value is required.

The returned algebraic bounds are

```text
|Delta Z_ij| <= E_Z,ij,
|Delta R_ij| <= E_Z,ij,
|Delta L_ij| <= E_Z,ij/omega.
```

Off-diagonal inductance entries therefore obtain the same certified mutual-inductance error bound.

At a fixed thermal state, the energy factorization and any full sparse-LU fallback are reused across all port right-hand sides.

## 6. Projected heat-source certificate

For reduced thermal test mode `phi_j`, define

```text
m_j = ||phi_j||_infinity.
```

The projected Joule source has a quadratic field form. Since the weighted Joule operator is bounded by `m_j` times the conductive energy,

```text
|Delta q_j|
<= (omega m_j / 2)
   [2 ||x_h||_H epsilon_H + epsilon_H^2].
```

Thus the algebraic field-solve contribution to the reduced electromagnetic heat source is available directly as

```text
eta_EM,alg = ||Delta q_em,r||_2.
```

This bound also remains independent of the copper/seawater conductivity ratio when expressed in the physical energy norm.

## 7. Relation to other electromagnetic error sources

The algebraic solve error is distinct from:

```text
constitutive-series error,
electromagnetic reduced-basis error,
spatial mesh-discretization error,
outer-domain truncation error.
```

They must not be hidden inside one solver tolerance. Each requires its own deterministic certificate and can then contribute additively to a conservative electromagnetic source bound used by the coupled thermal estimate.

For example,

```text
eta_EM
<= eta_constitutive
 + eta_algebraic
 + eta_EM-ROM
 + eta_mesh
 + eta_outer.
```

Only terms with an implemented proof/certificate should be included as finite certified quantities.

## 8. High-contrast regression

The repository contains tetrahedral regression cases with

```text
sigma_Cu approximately 5.8e7 S/m,
sigma_seawater = 5 S/m,
ratio > 1e7,
f = 100 kHz.
```

The tests check:

```text
sparse/dense assembly equivalence on the verification mesh,
physical 1/sqrt(2) coercivity inequality,
energy-state a-posteriori error bounds,
Z/R/L/M algebraic output bounds,
projected heat-source algebraic bounds,
no eager dense Riesz allocation on the sparse online path.
```

Numerical reference solutions are not treated as exact truth: when a comparison reaches floating-point error scale, the reference solution is also residual-certified and the observed pair difference is checked against the sum of both certificates.

## 9. Remaining scalable-solver obligation

The next production step is to replace exact `H^-1` and the complete sparse-LU fallback with a scalable multilevel/auxiliary-space realization while preserving a verified bound for the inexact preconditioner and final field error.

That future solver must satisfy the same non-empirical rule:

```text
no fitted drop tolerance,
no unexplained iteration count,
no contrast-dependent empirical safety factor,
no acceptance criterion other than a proved/a-posteriori physical error bound.
```
