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
       snapshot-free reduced EM          certified thermal spectrum
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
4. **Snapshot-free electromagnetic reduction.** The reduced EM space is grown from physical residual Riesz lifts; no FEM/Maxwell solution snapshots are used.
5. **Certified thermal rank selection.** Thermal modes come from the generalized heat eigenproblem. The retained rank is either the complete discrete spectrum or the smallest rank satisfying the spectral-tail error certificate.
6. **Nonlinear copper temperature law without conductivity linearization.** On tetrahedra, the reciprocal conductivity induced by linear copper resistivity is represented by a barycentric geometric series whose minimal order is determined by a rigorous relative remainder bound. Retained terms are integrated exactly against Nedelec fields.
7. **Exact P1-weighted Joule projection.** `q_em,r` and its Jacobian use analytic barycentric integration; no tetrahedron-centre loss sampling is required.
8. **Copper/seawater loss separation from the same field state.** Both affine and certified nonlinear tetrahedral paths support `P_Cu` and `P_sea` without add-on resistance formulas.
9. **Field-derived multiport outputs.** Joint multi-RHS reduction returns reciprocal `Z`, `R`, `L`, and `M`, with passivity and port-power/Joule-power checks.
10. **Constitutive remainder propagated to outputs.** The certified conductivity-series error is propagated through the EM inverse to deterministic bounds on `Z/R/L/M`, electromagnetic state, and projected heat source `q_em,r`. If the inverse perturbation condition fails, the result is explicitly uncertified rather than weakened heuristically.
11. **Intrinsic analytic neural evolution.** Initial coordinates and static operating parameters are network-internal zero-dynamics analytic nodes. One graph represents `(a0,U,t) -> a(t)` for a fixed spatial/thermal operator family; no external condition encoder generates network weights.
12. **Residual-grown analytic topology.** Candidate response neurons are selected by the unresolved physical residual and exact heat-source Jacobian, followed by full nonlinear residual re-evaluation.
13. **Resonance-stable analytic evaluation.** The fast polynomial-exponential compiler has an independent exact state-space realization backend, so exact/near resonance requires no closeness threshold.
14. **Continuous affine-domain certification.** For affine EM parameter dependence, branch-and-bound proves a residual bound over a continuous parameter box or returns `violated` / `indeterminate`.

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
||e_T|| <= (eta_NN + eta_ROM + eta_EM) / kappa,
```

where the nonlinear tetrahedral constitutive remainder now provides an executable contribution to `eta_EM` through the certified heat-source error bound.

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
- [`docs/TETRAHEDRAL_CORE.md`](docs/TETRAHEDRAL_CORE.md): unstructured Nedelec/P1 field chain, certified nonlinear material integration, and output error propagation.
- [`docs/COMPATIBLE_EM.md`](docs/COMPATIBLE_EM.md): reciprocal `A-psi` electromagnetic formulation and multiport field outputs.
- [`docs/SPATIAL_3D.md`](docs/SPATIAL_3D.md): shared spatial-discretization foundations.
- [`docs/NONLINEAR_CONSTITUTIVE.md`](docs/NONLINEAR_CONSTITUTIVE.md): nonlinear temperature-dependent material laws.
- [`docs/PARAMETRIC_ANALYTIC.md`](docs/PARAMETRIC_ANALYTIC.md): intrinsic `(a0,U,t)` analytic network and parameter-domain residual growth.

## Main implementation layers

- `sdfmpneo/spatial/tetra3d.py`: unstructured tetrahedral topology, Nedelec geometry, and P1 thermal assembly
- `sdfmpneo/spatial/barycentric_polynomial.py`: exact barycentric polynomial algebra/integration
- `sdfmpneo/em/tetra.py`: affine tetrahedral reciprocal `A-psi` correctness path
- `sdfmpneo/em/tetra_nonlinear.py`: certified nonlinear tetrahedral material/field path
- `sdfmpneo/em/reciprocal_series.py`: rigorous reciprocal copper series and remainder bounds
- `sdfmpneo/em/tetra_nonlinear_diagnostics.py`: nonlinear material-region Joule powers
- `sdfmpneo/em/reduced.py`: Cholesky-Riesz snapshot-free single/multi-RHS EM reduction
- `sdfmpneo/em/ports.py`: field-derived multiport `Z/R/L/M`
- `sdfmpneo/thermal/spectral.py`: thermal spectrum and tail-certified rank selection
- `sdfmpneo/analytic`: intrinsic analytic DAG, parameter algebra, fast compiler, and stable realization
- `sdfmpneo/training`: physical residual and residual-driven analytic-network growth
- `sdfmpneo/certification`: state, EM-domain, multiport, and constitutive-to-output certificates
- `sdfmpneo/tetra_core.py`: one-call affine or certified-nonlinear tetrahedral electrothermal core
- `sdfmpneo/model.py`: fixed and parameter-conditioned arbitrary-time online query interfaces

## Non-negotiable rules

- No labelled FEM/Maxwell/experimental solution data in training.
- No full-order solution snapshots are required for EM basis construction.
- No geometry-specific impedance solver is a core dependency.
- No empirical near/far split, fixed neural width/depth, artificial thermal time constant, gauge penalty, or empirical constitutive correction defines the method.
- Free approximation orders/ranks must be set by governing physics or explicit error/convergence certificates.
- Full-order simulations and experiments are validation tools only.

## Remaining production obligations

The mathematical unstructured core is present, but production-scale underwater WPT still requires:

1. CAD/mesh import and conforming mesh generation for actual round/rounded-square coils, package, and seawater domains;
2. sparse end-to-end Nedelec assembly/solve and a verified high-contrast preconditioner for realistic copper/seawater ratios;
3. continuous-domain certification for the nonlinear tetrahedral material problem, geometry, and frequency;
4. mesh-discretization and linear-solver error contributions in the unified output/state certificate;
5. certified open/infinite seawater electromagnetic and thermal outer-boundary treatment;
6. scalable partial thermal eigensolution with a certified lower bound for the first omitted eigenvalue;
7. terminal-current constrained solid-conductor ports where impressed closed-current ports are not the intended excitation;
8. parameterization across geometry/operator families that alter the thermal spectrum, beyond the already implemented static-`U` analytic network at a fixed operator family;
9. certified compression/minimalization of large analytic state-space realizations and a globally convergent network-growth proof.
