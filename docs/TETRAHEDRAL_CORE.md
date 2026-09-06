# Certified tetrahedral electromagnetic–thermal core

This document records the current unstructured spatial path of SDF-MPNEO. It is a real first-order tetrahedral field discretization, not a geometry-specific impedance formula and not a voxel approximation.

## 1. Compatible tetrahedral topology

`TetrahedralComplex3D` receives only vertex coordinates and tetrahedral connectivity. It constructs oriented nodes, edges, and faces and the incidence matrices

```text
G : nodes -> edges,
C : edges -> faces,
```

with

```text
C G = 0
```

exactly in sparse integer arithmetic.

A deterministic spanning tree fixes the magnetic-vector-potential gauge. Cotree edges are the independent magnetic coordinates. Each connected conducting component removes one scalar-potential reference node. No gauge penalty, numerical null-space tolerance, or empirical rank threshold is used.

## 2. Electromagnetic finite elements

The magnetic field uses first-order Nedelec edge elements. For tetrahedron `T`, local basis functions are

```text
N_ij = lambda_i grad(lambda_j) - lambda_j grad(lambda_i).
```

The code assembles the edge curl-curl matrix and conductivity mass matrix directly. With

```text
psi = phi/(j omega),
A = R_A alpha,
E = -j omega (R_A alpha + G_c psi),
```

the reciprocal magnetoquasistatic block is

```text
[ K_A + j*w R_A^T M_sigma R_A,  j*w R_A^T M_sigma G_c ]
[ j*w G_c^T M_sigma R_A,         j*w G_c^T M_sigma G_c ].
```

For reciprocal real material operators this matrix is complex symmetric. Reciprocity therefore comes from the field discretization itself.

## 3. Shared P1 thermal field

The same tetrahedral mesh assembles the thermal P1 mass and conduction operators. With homogeneous Dirichlet boundary treatment in the current correctness implementation,

```text
K_T phi_i = lambda_i M_T phi_i.
```

The retained thermal rank is not an arbitrary user integer. The code either keeps the full discrete spectrum or selects the smallest rank satisfying the spectral-tail certificate based on omitted initial energy, a source dual bound, and requested state accuracy.

Thermal eigenmodes are mapped back to tetrahedral local P1 nodal values and are used directly by the electromagnetic constitutive integration.

## 4. Exact barycentric Nedelec integration

For a barycentric polynomial weight

```text
p(lambda_0,...,lambda_3),
```
nthe code uses the exact simplex identity

```text
int_T prod_i lambda_i^(alpha_i) dV
= 6 |T| prod_i alpha_i! / (3 + sum_i alpha_i)!.
```

Consequently P1 conductivity weights, P1 thermal test functions, and their polynomial products are integrated analytically against the Nedelec basis. No tetrahedron-centre sampling or empirical quadrature order is required by this core.

This is used for

```text
M_sigma(a),
dM_sigma/da_k,
q_j = 1/2 x^H L_E^H M_(sigma phi_j) L_E x,
dq_j/da_k.
```

## 5. Certified nonlinear copper law

For linear copper resistivity,

```text
rho(T) = rho_ref [1 + alpha (T-T_ref)],
sigma(T) = sigma_ref / d(T),
d(T) = 1 + alpha (T-T_ref).
```

Inside a P1 tetrahedron `d(x)` is P1. Define

```text
d_bar = (d_max+d_min)/2,
z = d/d_bar - 1,
q = (d_max-d_min)/(d_max+d_min) < 1.
```

Then

```text
1/d = (1/d_bar) sum_{n=0}^N (-z)^n + R_N,
```

with exact relative remainder bound

```text
|R_N| / |1/d| <= q^(N+1).
```

The derivative requires `1/d^2`, for which

```text
1/d^2 = (1/d_bar^2) sum_{n=0}^N (n+1)(-z)^n + R_N^(2)
```

and

```text
|R_N^(2)| / |1/d^2|
<= q^(N+1)[(N+2)+(N+1)q].
```

The smallest `N` satisfying the declared constitutive error budget is selected deterministically. Each retained term is a barycentric polynomial and is then integrated exactly. The method therefore does not linearize copper conductivity and does not choose a polynomial order heuristically.

The present certified tetrahedral material library supports:

- constant conductivity;
- a physically specified affine conductivity law;
- reciprocal conductivity induced by linear resistivity.

Other nonlinear laws require their own rigorous approximation/remainder construction before they can enter this certified path.

## 6. Snapshot-free electromagnetic reduction

The tetrahedral field problem implements the same generic operator interface as the rectilinear correctness core. The existing residual-Riesz reducer therefore constructs the reduced electromagnetic space without full-order solution snapshots.

The Riesz metric is based on magnetic energy plus Joule energy per electrical radian. Basis enrichment uses Cholesky Riesz coordinates and the full operator residual.

A finite candidate state set can build an executable reduced basis, but it is not interpreted as a continuous-domain certificate. Continuous certification for the nonlinear tetrahedral parameter domain remains a separate obligation.

## 7. Multiport and regional powers

Closed impressed-current port cochains are mapped into the same field coordinates. The code evaluates

```text
Z = j*w B^T A^-1 B,
R = Re(Z),
L = Im(Z)/w.
```

For both affine and certified nonlinear tetrahedral material paths, regression tests verify the physical closure

```text
Z^T = Z,
1/2 Re(I^H Z I)
= 1/2 E^H M_sigma E
= P_Cu + P_sea
```

for disjoint copper/seawater material regions.

The nonlinear regional powers use the same certified reciprocal series and exact barycentric Nedelec integration as the coupled field operator; they do not fall back to an empirical resistance model.

## 8. Constitutive remainder -> output certificates

The reciprocal series supplies a pointwise relative conductivity bound

```text
|sigma_tilde-sigma| <= eps_sigma sigma.
```

Hence

```text
|sigma_tilde-sigma|
<= eps_sigma/(1-eps_sigma) sigma_tilde.
```

This produces a computable field-operator perturbation bound `||Delta A||`. If

```text
||A_tilde^-1|| ||Delta A|| < 1,
```

the inverse perturbation theorem yields finite deterministic bounds for

```text
||A^-1-A_tilde^-1||,
||Delta Z||_2,
|Delta R_ij|,
|Delta L_ij|,
```

and for the electromagnetic state error.

The same state perturbation plus a bound on the loss-operator remainder produces componentwise and vector bounds for

```text
Delta q_em,r.
```

This quantity can be used directly as the electromagnetic contribution `eta_EM` in the coupled state certificate

```text
||e_T|| <= (eta_NN + eta_ROM + eta_EM) / kappa.
```

If the inverse perturbation condition fails, the code returns an uncertified/infinite bound rather than weakening the theorem.

## 9. Current unstructured chain

```text
curved-geometry tetrahedral mesh
        |
        +--> exact G,C + tree/cotree gauges
        |
        +--> P1 thermal M_T,K_T
        |        |
        |        +--> certified thermal rank
        |        +--> P1 thermal modes
        |
        +--> Nedelec magnetic operator
        +--> certified nonlinear M_sigma(T)
                 |
                 +--> reciprocal A-psi solve
                 +--> snapshot-free EM reduction
                 +--> q_em,r, dq_em,r/da
                 +--> Z/R/L/M
                 +--> P_Cu/P_sea
                 +--> constitutive-to-output certificates
                          |
                          +--> analytic neural residual/evolution
```

## 10. Remaining production obligations

The unstructured mathematical core is now present, but the following are not yet claimed complete:

1. CAD/mesh import and automatic conforming meshing for actual round/rounded-square coils, package, and seawater volumes;
2. sparse end-to-end Nedelec assembly/solve and verified high-contrast preconditioning for realistic copper/seawater conductivity ratios;
3. continuous parameter-domain certification for the nonlinear tetrahedral constitutive problem, geometry, and frequency;
4. propagation of mesh discretization and linear-solver errors into the unified output/state certificate;
5. open/infinite seawater electromagnetic and thermal outer-domain certification;
6. scalable partial thermal eigensolution with a certified lower bound on the first omitted eigenvalue;
7. solid-conductor terminal-current constrained ports when the excitation is not represented as an impressed closed-current source.

These are explicit remaining tasks; they are not hidden by empirical safety factors.
