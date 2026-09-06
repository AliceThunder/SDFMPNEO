# Executable core status

Run:

```bash
python -m pip install -e '.[dev]'
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q
```

GitHub Actions runs the complete test suite for every push to `main` and pull request targeting `main`.

## Implemented executable chain

The repository now contains:

- orthogonal 3-D and unstructured tetrahedral electromagnetic–thermal correctness paths;
- exact compatible incidence with `C @ G = 0`;
- deterministic tree-cotree magnetic gauge and conducting-component scalar-potential gauge;
- first-order Nedelec tetrahedral electromagnetic assembly;
- P1 tetrahedral thermal mass/conduction assembly on the same mesh;
- reciprocal scaled `A-psi` magnetoquasistatic coordinates;
- native sparse nonlinear tetrahedral `A(a)`, `dA/da`, loss operators, port projection, and physical energy metric;
- physical electromagnetic energy metric `H(a)=K+D(a)` for `A(a)=K+iD(a)`;
- structural energy coercivity `beta_H >= 1/sqrt(2)`, independent of copper/seawater conductivity contrast;
- generic certified Riesz-action interface with proved dual-norm intervals;
- certified PCG Riesz action with no scientific `rtol`/`atol`;
- physical magnetic/scalar block Riesz theorem and replaceable certified block actions;
- normalized-Gershgorin `gamma` certificate with residual-certified generalized-trace fallback when the magnetic block is not diagonally dominant;
- high-contrast regression with `sigma_Cu/sigma_seawater > 1e7`;
- **fully sparse physical-energy snapshot-free single- and multi-RHS electromagnetic reduction for the certified nonlinear path**;
- certified local residual-Riesz enrichment without full-order solution snapshots;
- reference-energy `H0=H(0)` reduced-basis whitening using only a small dense Gram matrix, with no dense full-order Cholesky factor;
- finite-candidate reduction certificates in the local physical energy norm;
- one-call top-level joint-port reduction through `build_reduced_electromagnetics(..., port_set=...)`;
- certified reduced multiport `Z/R/L/M` from a reduced equilibrium solve plus sparse residual/Riesz certification, without a full-order field equilibrium solve;
- regression preventing the formal nonlinear core from silently falling back to the old global sparse-LU `H^-1` reference action;
- sparse reduced Joule source and exact reduced heat-source Jacobian;
- legacy dense Cholesky-Riesz reduction retained only for affine/orthogonal verification paths;
- thermal generalized eigenmodes with spectral-tail certified rank selection;
- exact barycentric polynomial integration against Nedelec fields;
- affine tetrahedral temperature-feedback correctness path;
- certified nonlinear tetrahedral material path with reciprocal copper conductivity induced by linear resistivity, without conductivity linearization;
- rigorously selected reciprocal-series order from pointwise remainder bounds for both `1/d` and `1/d^2`;
- exact/certified `dA/da` and projected Joule-source Jacobians;
- field-derived `Z/R/L/M` and copper/seawater regional powers;
- reciprocity, passivity, and port-power/Joule-power/material-region power closure checks;
- residual-to-port-output error certificates;
- constitutive-series remainder propagated to deterministic `Z/R/L/M` bounds;
- constitutive-series remainder propagated to projected heat-source error bounds usable as `eta_EM`;
- continuous affine thermal-state EM residual certification by branch-and-bound;
- intrinsic analytic neural DAGs with arbitrary-depth response chains;
- intrinsic parameter nodes for `(a0,U,t) -> a(t)` at a fixed spatial/operator family;
- exact operating-parameter residual sensitivities;
- residual-driven network growth over state/operating-parameter analytic dictionaries;
- fast polynomial-exponential compilation plus an independent exact state-space realization backend with no near-resonance threshold;
- fixed and parameter-conditioned arbitrary-time online query interfaces.

No FEM/Maxwell solution snapshots or transient solution labels are used by the production training/reduction method.

## Current unstructured physical chain

```text
tetrahedral geometry + material regions
        |
        +--> exact G,C + tree/cotree gauges
        |
        +--> P1 thermal M_T,K_T
        |       |
        |       +--> thermal spectrum
        |       +--> certified retained rank
        |       +--> local P1 thermal modes
        |
        +--> Nedelec magnetic operator K
        +--> certified conductivity operator D(a)
                |
                +--> sparse reciprocal A(a)=K+iD(a)
                |       |
                |       +--> local H(a)=K+D(a)
                |       +--> beta_H >= 1/sqrt(2)
                |       +--> physical block theorem
                |       |       |
                |       |       +--> certified gamma
                |       |       +--> magnetic block action
                |       |       +--> scalar block action
                |       |       +--> certified PCG Riesz action
                |       |
                |       +--> sparse residual-Riesz ROM growth
                |                 |
                |                 +--> joint unit-port RHS set
                |                 +--> certified local Riesz lift
                |                 +--> H0-orthonormal global basis
                |                 +--> reduced A_r(a)
                |                 +--> reduced q_em,r,dq_em,r/da
                |                 +--> certified reduced Z/R/L/M
                |       +--> algebraic/reduction output bounds
                |
                +--> P_Cu/P_sea
                +--> constitutive/reduction/solve certificates
                          |
                          +--> analytic residual-driven network
                          +--> direct arbitrary-time query
```

## Reciprocal tetrahedral electromagnetic system

With

```text
A_field = R_A alpha,
psi = phi/(j omega),
E = -j omega (R_A alpha + G_c psi),
```

the tetrahedral Nedelec field system is

```text
[ K_A + j*w R_A^T M_sigma R_A,  j*w R_A^T M_sigma G_c ] [alpha]   [R_A^T J_s]
[ j*w G_c^T M_sigma R_A,         j*w G_c^T M_sigma G_c ] [ psi ] = [    0      ].
```

For reciprocal real material matrices it is complex symmetric. The code does not symmetrize `Z` afterwards.

## Contrast-independent sparse field certificate

Write the gauge-reduced physical matrix as

```text
A = K + iD,
K >= 0,
D >= 0,
H = K + D.
```

For any complex state `x`,

```text
|x^H A x|
= sqrt[(x^H K x)^2 + (x^H D x)^2]
>= x^H H x / sqrt(2).
```

Therefore

```text
beta_H >= 1/sqrt(2)
```

and for the explicitly recomputed residual `r=b-Ax_h`,

```text
||x-x_h||_H
<= sqrt(2) ||r||_(H^-1).
```

The constant is structural and contains no empirical conductivity-ratio factor. The recommended sparse path does not need an external minimum singular-value estimate.

For closed port source `b_i`,

```text
|Delta Z_ij|
<= w ||b_i||_(H^-1) ||Delta x_j||_H.
```

A requested per-entry impedance accuracy determines the field-state accuracy directly. There is no separate scientific `solver_tol` parameter.

For reduced thermal test mode `phi_j`, with `m_j=||phi_j||_infinity`, the same energy-state bound gives

```text
|Delta q_j|
<= (w m_j / 2)
   [2 ||x_h||_H epsilon_H + epsilon_H^2].
```

This is the executable algebraic-solve contribution to `eta_EM`.

The formal nonlinear ROM does **not** require an exact global `H^-1`. The Riesz action is certified by its own residual/error enclosure. The separate full-order correctness solver still retains complete sparse LU on `A` as a verification fallback; that fallback is not the reduced equilibrium used by the ROM.

See `docs/SPARSE_ENERGY_SOLVER.md` and `docs/CERTIFIED_RIESZ_ACTION.md` for the derivations.

## Physical-block Riesz certificate

Partition the local energy metric as

```text
H = [[K_A + D_AA, D_Apsi],
     [D_psiA,       D_psipsi]].
```

After tree-cotree gauge elimination `K_A>0`; conductivity positivity gives `D>=0`. If

```text
D_AA <= gamma K_A,
```

then

```text
H >= m_phys(gamma) diag(K_A,D_psipsi),
```

with

```text
m_phys(gamma)
= 2/[2+gamma+sqrt(gamma^2+4 gamma)].
```

If the magnetic/scalar block actions prove

```text
K_A      >= m_K P_K,
D_psipsi >= m_E P_E,
```

then

```text
H >= m_phys(gamma) min(m_K,m_E) diag(P_K,P_E).
```

This is the formal preconditioner bound used by `CertifiedPCGRieszAction`. The default block actions use complete sparse LU only as correctness-scale implementations. Tests also inject non-LU certified block actions with `splu` disabled, proving that block LU is replaceable without changing the outer theorem.

`gamma` is first bounded by normalized Gershgorin. If the magnetic block is not diagonally dominant, the current correctness fallback uses explicitly residual-certified magnetic factorization to prove

```text
gamma <= trace(K_A^-1 D_AA).
```

That sparse factorization is certificate-construction work only. Eliminating it with a scalable factorization-free `gamma` proof remains a production task.

## Sparse snapshot-free electromagnetic reduction

For a reduced basis `V`,

```text
A_r(a) = V^H A(a) V,
b_r = V^H b,
x_r(a) = V A_r(a)^-1 b_r.
```

The sparse full-order residual is

```text
r(a) = b - A(a) x_r(a).
```

Reduction error is certified in the **state-dependent physical metric**

```text
eta(a) = sqrt(2) ||r(a)||_(H(a)^-1).
```

A `CertifiedRieszAction` provides a certified interval for the dual norm and an approximate lift `y_tilde ~= H(a)^-1 r(a)`. A candidate is accepted only after the dual-norm upper bound proves the requested state-error target. If unresolved, `y_tilde` enriches the span and the complete residual is recomputed.

No full-order solution `A(a)^-1 b` is used as a basis snapshot.

A single global basis is stored using

```text
H0 = H(0).
```

`H0` is only a coordinate metric. The error theorem remains local in `H(a)`. For candidate basis `W`, the small Gram matrix

```text
G = W^H H0 W = L L^H
```

gives

```text
V = W L^(-H),
V^H H0 V = I
```

up to floating-point backward error. No full-order dense Cholesky factor is created.

For a joint port source matrix `B`,

```text
A_r C = V^H B,
X_r = V C,
Z_r = j*w B^T X_r.
```

The top-level nonlinear core accepts `port_set=ports`, so every unit-port source column is included in one joint residual-greedy construction. Each residual column is independently certified, and the same physical-block Riesz factory is stored in the reduced model and reused by online output certification.

Thus the field equilibrium is reduced-order, while sparse full-order work is limited to deterministic residual/operator/Riesz certification. Regression tests disable the old global sparse-LU Riesz reference action and confirm that the one-call joint-port ROM and certified `Z/R/L/M` chain remains executable.

The finite candidate-state set certifies those states/excitations only. It is not a continuous-domain certificate.

See `docs/SPARSE_ENERGY_REDUCTION.md` for the complete reduction derivation.

## Exact barycentric loss projection

For barycentric monomials on a tetrahedron,

```text
int_T prod_i lambda_i^(alpha_i) dV
= 6 |T| prod_i alpha_i! / (3 + sum_i alpha_i)!.
```

This identity is used to assemble polynomial-weighted Nedelec mass matrices. P1 conductivity, P1 thermal tests, and P1×P1 derivative weights are integrated analytically rather than sampled at element centres.

## Certified reciprocal copper law

For

```text
rho(T)=rho_ref[1+alpha(T-T_ref)],
sigma(T)=sigma_ref/d(T),
d(T)=1+alpha(T-T_ref),
```

`d(x)` is P1 inside each tetrahedron. Defining

```text
d_bar=(d_max+d_min)/2,
z=d/d_bar-1,
q=(d_max-d_min)/(d_max+d_min)<1,
```

gives

```text
1/d = d_bar^-1 sum_{n=0}^N (-z)^n + R_N,
|R_N|/|1/d| <= q^(N+1).
```

For the derivative,

```text
1/d^2 = d_bar^-2 sum_{n=0}^N (n+1)(-z)^n + R_N^(2),
```

with

```text
|R_N^(2)|/|1/d^2|
<= q^(N+1)[(N+2)+(N+1)q].
```

The code selects the minimum order satisfying the declared constitutive error allocation. A too-coarse series is allowed to become **uncertified**; no stability condition is relaxed to force a result.

## Electromagnetic error decomposition

The code distinguishes

```text
eta_constitutive : certified nonlinear material-series error,
eta_algebraic    : certified sparse linear/Riesz action error where applicable,
eta_EM-ROM       : certified reduced-space residual error on the declared domain.
```

Future certified terms are

```text
eta_mesh,
eta_outer.
```

They must not be hidden inside one numerical tolerance. A conservative source-level budget can use

```text
eta_EM
<= eta_constitutive
 + eta_algebraic
 + eta_EM-ROM
 + eta_mesh
 + eta_outer.
```

Only terms with an implemented proof should be treated as finite certified contributions.

The current constitutive certificate propagates pointwise conductivity error through an inverse perturbation theorem. If the inverse perturbation condition fails, the result remains uncertified/infinite rather than weakening the theorem.

## Multiport and regional power closure

For closed impressed-current ports collected in `B`,

```text
A X = B,
Z = j*w B^T X.
```

The tetrahedral affine and certified nonlinear paths are regression-tested for

```text
Z^T = Z,
1/2 Re(I^H Z I)
= 1/2 E^H M_sigma E
= P_Cu + P_sea.
```

The nonlinear regional power evaluator uses the same constitutive series and exact barycentric integration as the coupled field operator.

## Thermal rank certificate

For first omitted thermal eigenvalue `lambda_(r+1)`, omitted initial M-norm `E0`, and certified source dual bound `Q`,

```text
||T_tail(t)||_M
<= exp(-lambda_(r+1)t) E0
 + (1-exp(-lambda_(r+1)t)) Q/lambda_(r+1).
```

The verification implementation currently computes the full discrete spectrum and chooses the smallest retained rank meeting the requested accuracy. A scalable production eigensolver must retain the same certificate without computing the full spectrum.

## Intrinsic parameter-conditioned analytic network

The analytic algebra has network-internal zero-dynamics nodes for initial state and declared static operating parameters. For a fixed spatial/operator family the network directly represents

```text
(a0,U,t) -> a(t)
```

without an external conditioning network that generates coefficients.

The same graph has two evaluators:

```text
fast polynomial-exponential compiler,
exact state-space realization h(t)=c^T exp(A t)b.
```

The state-space realization handles repeated and nearly repeated decay rates as ordinary confluent/Jordan structure, so no empirical resonance threshold is needed.

Parameter-domain residual growth can select products containing `a0`, `U`, and existing response nodes. Candidate weights come from the tangent physical residual and are accepted only after full nonlinear residual re-evaluation.

## Continuous-domain status

A continuous branch-and-bound certificate is implemented for strictly affine EM thermal-state dependence. It returns

```text
certified / violated / indeterminate.
```

The certified nonlinear tetrahedral material path and sparse reducer have pointwise/a-posteriori certificates, but a **continuous state/geometry/frequency/source-domain certificate for the nonlinear reduced operator family is still pending**. A finite candidate set is not treated as proof of the continuous domain.

## Remaining production obligations

1. CAD/mesh import and conforming meshing for actual round/rounded-square conductors, package, and seawater domains.
2. Replace the current exact magnetic/scalar block actions by memory-scalable certified multilevel/auxiliary-space actions while retaining their proved block spectral-equivalence constants.
3. Replace the residual-certified sparse-LU `gamma` fallback by a scalable factorization-free magnetic-domination certificate.
4. Continuous-domain certification for nonlinear tetrahedral material, reduced-space residual, geometry, frequency, and source parameters.
5. Mesh-discretization and certified outer-domain truncation terms in the unified output/state certificate.
6. Certified open/infinite seawater electromagnetic and thermal outer-domain treatment.
7. Scalable partial thermal eigensolution with a certified lower bound on the first omitted eigenvalue.
8. Solid-conductor terminal-current constrained ports when impressed closed-current sources are not the intended excitation.
9. Parameterization across geometry/operator families that change the thermal spectrum; current intrinsic `U` parameterization assumes a fixed operator family.
10. Certified compression/minimalization of large analytic state-space realizations and a global convergence certificate for residual-grown analytic-network construction.

These limits are explicit implementation obligations, not empirical safety factors.
