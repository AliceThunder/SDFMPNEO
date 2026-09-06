# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator for Underwater WPT**

SDF-MPNEO is a solution-data-free electromagnetic–thermal surrogate framework for underwater wireless power transfer. Conductor, package, seawater, and passive media remain explicit spatial physics; fluid velocity is outside the present scope.

The method has no geometry-specific impedance-solver dependency. Its current executable path is

```text
3-D geometry/materials + a0 + operating condition U + arbitrary time t
                                  |
                                  v
                 compatible electromagnetic + thermal physics
                                  |
                +-----------------+-----------------+
                |                                   |
                v                                   v
 sparse physical-energy reduced EM       certified thermal spectrum
                |                                   |
                +---------- quasi-static -----------+
                             EM elimination
                                  |
                                  v
                     q_em,r(a,U), dq_em,r/da
                                  |
                                  v
                 intrinsic parametric analytic network
                                  |
                                  v
               a(t), T(x,t), Z/R/L/M, P_Cu, P_sea
                                  |
                                  v
                     deterministic error certificates
```

## What is implemented

1. **Compatible 3-D electromagnetic physics.** Both an orthogonal correctness grid and an unstructured tetrahedral path are implemented. The tetrahedral path uses first-order Nedelec edge elements; the shared thermal path uses P1 finite elements.
2. **Exact topology and gauges.** `C @ G = 0` is constructed exactly. Tree-cotree magnetic gauge elimination and one scalar-potential reference per conducting component remove nullspaces without penalty parameters or numerical rank thresholds.
3. **Reciprocal `A-psi` coordinates.** With `psi=phi/(j omega)`, reciprocal real materials produce a complex-symmetric field matrix, so reciprocity is structural.
4. **Fully sparse snapshot-free electromagnetic reduction for the nonlinear production path.** At each thermal state, the local physical metric `H(a)=K+D(a)` certifies the reduced residual and generates the enrichment direction `H(a)^-1 r`. The global basis is stored in the reference physical metric `H0=H(0)` without ever forming a dense full-order Cholesky factor. No FEM/Maxwell solution snapshots are used.
5. **Certified thermal rank selection.** Thermal modes come from the generalized heat eigenproblem. The retained rank is either the complete discrete spectrum or the smallest rank satisfying the spectral-tail error certificate.
6. **Nonlinear copper temperature law without conductivity linearization.** On tetrahedra, the reciprocal conductivity induced by linear copper resistivity is represented by a barycentric geometric series whose minimal order is determined by a rigorous relative remainder bound. Retained terms are integrated exactly against Nedelec fields.
7. **Exact P1-weighted Joule projection.** `q_em,r` and its Jacobian use analytic barycentric integration; no tetrahedron-centre loss sampling is required.
8. **Copper/seawater loss separation from the same field state.** Both affine and certified nonlinear tetrahedral paths support `P_Cu` and `P_sea` without add-on resistance formulas.
9. **Certified reduced multiport outputs.** A joint multi-RHS reduced space returns reciprocal `Z`, `R`, `L`, and `M` by solving only the small reduced equilibrium. Sparse full-order residual/Riesz actions provide deterministic per-entry output bounds; no full-order field equilibrium solve is required for the reduced multiport query.
10. **Constitutive remainder propagated to outputs.** The certified conductivity-series error is propagated through the EM inverse to deterministic bounds on `Z/R/L/M`, electromagnetic state, and projected heat source `q_em,r`. If the inverse perturbation condition fails, the result is explicitly uncertified rather than weakened heuristically.
11. **Native sparse high-contrast tetrahedral field path.** Nonlinear tetrahedral `A(a)`, `dA/da`, loss operators, port projection, and energy metrics remain sparse. Regression cases use copper/seawater conductivity ratios above `1e7`.
12. **Contrast-independent physical energy certificate.** For the physical sparse operator `A=K+iD`, `K,D>=0`, the metric `H=K+D` gives the structural coercivity bound `beta_H >= 1/sqrt(2)`, independent of conductivity contrast. Hence `||e||_H <= sqrt(2)||r||_(H^-1)`.
13. **Output-driven sparse solve accuracy.** Requested `Z` accuracy is converted directly into the required field energy accuracy; the user does not choose an unrelated Krylov tolerance or minimum singular-value estimate. Linear-solve and reduced-space errors propagate to `Z/R/L/M` and projected heat-source bounds.
14. **Deterministic sparse correctness fallback.** Energy-preconditioned BiCGSTAB is followed, when necessary, by complete sparse LU without ILU drop parameters. This is a certified correctness/high-contrast baseline, not a claim of million-degree-of-freedom scalability.
15. **Intrinsic analytic neural evolution.** Initial coordinates and static operating parameters are network-internal zero-dynamics analytic nodes. One graph represents `(a0,U,t) -> a(t)` for a fixed spatial/thermal operator family; no external condition encoder generates network weights.
16. **Residual-grown analytic topology.** Candidate response neurons are selected by the unresolved physical residual and exact heat-source Jacobian, followed by full nonlinear residual re-evaluation.
17. **Resonance-stable analytic evaluation.** The fast polynomial-exponential compiler has an independent exact state-space realization backend, so exact/near resonance requires no closeness threshold.
18. **Continuous affine-domain certification.** For affine EM parameter dependence, branch-and-bound proves a residual bound over a continuous parameter box or returns `violated` / `indeterminate`.

## Core equations

The reduced electromagnetic equilibrium is

```text
A_em,r(a;U) c = b_em,r(U),
q_em,r(a;U) = q_Cu,r + q_sea,r.
```

The thermal dynamics are

```text
da/dt + Lambda_T a = g_em(a;U).
```

The analytic neural operator evaluates

```text
a(t) = N_analytic(a0,U,t)
```

directly at arbitrary `t`, without thermal time marching during inference. Its physical residual is

```text
R = da/dt + Lambda_T a - g_em(a;U).
```

For a contraction margin `kappa>0`, the current error architecture uses

```text
||e_T|| <= (eta_NN + eta_ROM + eta_EM) / kappa.
```

The nonlinear constitutive remainder, electromagnetic reduced-space residual, and algebraic sparse field-solve error provide separate executable contributions to `eta_EM`; mesh and outer-domain errors remain explicit pending terms rather than hidden tolerances.

## Sparse physical-energy field certificate

For the gauge-reduced reciprocal tetrahedral operator,

```text
A = K + i D,
K >= 0,
D >= 0,
H = K + D.
```

For any complex field state `x`,

```text
|x^H A x|
= sqrt[(x^H K x)^2 + (x^H D x)^2]
>= x^H H x / sqrt(2).
```

Therefore

```text
beta_H >= 1/sqrt(2),
||x-x_h||_H <= sqrt(2) ||b-Ax_h||_(H^-1).
```

This stability bound contains no fitted conductivity-ratio correction. For closed port sources `b_i`,

```text
|Delta Z_ij|
<= omega ||b_i||_(H^-1) ||Delta x_j||_H.
```

A requested per-entry impedance accuracy therefore determines the field-solve requirement automatically. The same energy-state certificate is propagated to the reduced Joule heat source.

See [`docs/SPARSE_ENERGY_SOLVER.md`](docs/SPARSE_ENERGY_SOLVER.md) for the full-order solve theorem.

## Sparse physical-energy reduction

For reduced basis `V`,

```text
A_r(a) = V^H A(a) V,
b_r = V^H b,
x_r = V A_r(a)^-1 b_r.
```

The unresolved residual is

```text
r(a) = b - A(a) x_r(a).
```

At every candidate state, reduction error is certified with the **local** metric

```text
eta(a) = sqrt(2) ||r(a)||_(H(a)^-1),
H(a)=K+D(a).
```

The worst unresolved state/excitation generates the next basis direction

```text
y = H(a)^-1 r(a).
```

A single global coordinate system is stored in the intrinsic reference metric `H0=H(0)`. After each enrichment, only the small Gram matrix

```text
G = W^H H0 W
```

is Cholesky-whitened so that `V^H H0 V=I` up to backward error. Neither `H0` nor `A(a)` is densified.

For multiple ports,

```text
A_r C = V^H B,
X_r = V C,
Z_r = j*omega*B^T X_r.
```

Each reduced column is independently certified from its sparse full-order residual, yielding deterministic bounds on every entry of `Z/R/L/M` without solving the full-order electromagnetic equilibrium.

See [`docs/SPARSE_ENERGY_REDUCTION.md`](docs/SPARSE_ENERGY_REDUCTION.md) for the derivation and implementation contract.

## Multiport field output

For closed one-ampere impressed-current port cochains collected in `B`,

```text
A X = B,
Z = j*omega*B^T X,
R = Re(Z),
L = Im(Z)/omega.
```

Regression tests enforce

```text
Z^T = Z,
R is passive within numerical error,
1/2 Re(I^H Z I) = 1/2 E^H M_sigma E = P_Cu + P_sea.
```

## Certified nonlinear copper representation

Inside each tetrahedron,

```text
d(x) = 1 + alpha [T(x)-T_ref]
```

is P1. With

```text
d_bar = (d_max+d_min)/2,
z = d/d_bar - 1,
q = (d_max-d_min)/(d_max+d_min) < 1,
```

the code uses

```text
1/d = d_bar^-1 sum_{n=0}^N (-z)^n + R_N,
|R_N|/|1/d| <= q^(N+1),
```

and the corresponding certified `1/d^2` series for derivatives. The smallest `N` satisfying the declared constitutive error allocation is selected; `N` is not tuned empirically.

The resulting pointwise conductivity error is propagated to field/output bounds via the inverse perturbation condition

```text
||A_tilde^-1|| ||Delta A|| < 1.
```

Failure of this condition produces an uncertified result, as verified by regression tests.

## Run

```bash
python -m pip install -e '.[dev]'
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q
```

GitHub Actions runs the repository-wide test suite on every push to `main` and pull request targeting `main`.

## Documentation

- [`docs/SDFMPNEO_theory.tex`](docs/SDFMPNEO_theory.tex): governing equations and complete theory.
- [`docs/SDFMPNEO_implementation.md`](docs/SDFMPNEO_implementation.md): software architecture and implementation contract.
- [`docs/MVP_CORE.md`](docs/MVP_CORE.md): exact executable status and remaining obligations.
- [`docs/TETRAHEDRAL_CORE.md`](docs/TETRAHEDRAL_CORE.md): unstructured Nedelec/P1 field chain and nonlinear material integration.
- [`docs/SPARSE_ENERGY_SOLVER.md`](docs/SPARSE_ENERGY_SOLVER.md): sparse high-contrast field solve, `1/sqrt(2)` energy coercivity, and algebraic output certificates.
- [`docs/SPARSE_ENERGY_REDUCTION.md`](docs/SPARSE_ENERGY_REDUCTION.md): fully sparse snapshot-free residual-Riesz reduction and certified reduced multiport outputs.
- [`docs/COMPATIBLE_EM.md`](docs/COMPATIBLE_EM.md): reciprocal `A-psi` electromagnetic formulation and multiport field outputs.
- [`docs/SPATIAL_3D.md`](docs/SPATIAL_3D.md): shared spatial-discretization foundations.
- [`docs/NONLINEAR_CONSTITUTIVE.md`](docs/NONLINEAR_CONSTITUTIVE.md): nonlinear temperature-dependent material laws.
- [`docs/PARAMETRIC_ANALYTIC.md`](docs/PARAMETRIC_ANALYTIC.md): intrinsic `(a0,U,t)` analytic network and parameter-domain residual growth.

## Main implementation layers

- `sdfmpneo/spatial/tetra3d.py`: unstructured tetrahedral topology, Nedelec geometry, and P1 thermal assembly
- `sdfmpneo/spatial/barycentric_polynomial.py`: exact barycentric polynomial algebra/integration
- `sdfmpneo/em/tetra.py`: affine tetrahedral reciprocal `A-psi` verification path
- `sdfmpneo/em/tetra_nonlinear.py`: certified nonlinear tetrahedral sparse material/field path
- `sdfmpneo/em/reciprocal_series.py`: rigorous reciprocal copper series and remainder bounds
- `sdfmpneo/em/sparse_solver.py`: reusable certified sparse field solvers and physical energy residual norms
- `sdfmpneo/em/energy_solver.py`: physical `H=K+D` construction and contrast-independent A-psi solve wrapper
- `sdfmpneo/em/sparse_reduced.py`: production sparse physical-energy snapshot-free single/multi-RHS EM reduction
- `sdfmpneo/em/reduced.py`: legacy dense Cholesky-Riesz reducer retained for affine/orthogonal verification only
- `sdfmpneo/em/ports.py`: sparse port projection, full-order certificates, and certified reduced multiport `Z/R/L/M`
- `sdfmpneo/em/tetra_nonlinear_diagnostics.py`: nonlinear material-region Joule powers
- `sdfmpneo/thermal/spectral.py`: thermal spectrum and tail-certified rank selection
- `sdfmpneo/analytic`: intrinsic analytic DAG, parameter algebra, fast compiler, and stable realization
- `sdfmpneo/training`: physical residual and residual-driven analytic-network growth
- `sdfmpneo/certification`: state, EM-domain, constitutive, algebraic-solve, multiport, and heat-source certificates
- `sdfmpneo/tetra_core.py`: one-call certified nonlinear sparse-energy core plus explicitly named affine verification path
- `sdfmpneo/model.py`: fixed and parameter-conditioned arbitrary-time online query interfaces

## Non-negotiable rules

- No labelled FEM/Maxwell/experimental solution data in training.
- No full-order solution snapshots are required for production EM basis construction.
- No geometry-specific impedance solver is a core dependency.
- No empirical near/far split, fixed neural width/depth, artificial thermal time constant, gauge penalty, empirical constitutive correction, ILU drop tolerance, or unexplained solver tolerance defines the method.
- Free approximation orders/ranks and acceptance thresholds must be set by governing physics or explicit error/convergence certificates.
- Full-order simulations and experiments are validation tools only.

## Remaining production obligations

The mathematical unstructured sparse core and sparse snapshot-free reduction are present, but production-scale underwater WPT still requires:

1. CAD/mesh import and conforming mesh generation for actual round/rounded-square coils, package, and seawater domains;
2. a memory-scalable multilevel/auxiliary-space realization of the physical `H^-1` action and a certified replacement for the current complete sparse-LU correctness fallback on very large meshes;
3. continuous-domain certification for the nonlinear tetrahedral material/reduction problem over thermal state, geometry, frequency, and source parameters; the present finite candidate-set certificate is not treated as a continuous-domain proof;
4. spatial mesh-discretization error and certified outer-domain truncation contributions in the unified output/state certificate;
5. certified open/infinite seawater electromagnetic and thermal outer-boundary treatment;
6. scalable partial thermal eigensolution with a certified lower bound for the first omitted eigenvalue;
7. terminal-current constrained solid-conductor ports where impressed closed-current ports are not the intended excitation;
8. parameterization across geometry/operator families that alter the thermal spectrum, beyond the implemented static-`U` analytic network at a fixed operator family;
9. certified compression/minimalization of large analytic state-space realizations and a globally convergent network-growth proof.

The current sparse LU used for exact `H^-1` Riesz actions and correctness fallback is intentionally described as a correctness baseline, not as evidence that million-degree-of-freedom production scaling has already been solved.
