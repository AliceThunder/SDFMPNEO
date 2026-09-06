# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator for Underwater WPT**

SDF-MPNEO is a solution-data-free electromagnetic–thermal surrogate framework for underwater wireless power transfer. The current baseline retains conductor, package, seawater, and passive media in the spatial electromagnetic and thermal physics; fluid velocity is outside the present scope.

The method does **not** depend on any geometry-specific impedance solver. The electromagnetic branch is derived from compatible magnetoquasistatic field equations, reduced deterministically without solution snapshots, quasi-statically eliminated inside the slower thermal dynamics, and coupled to an analytic neural evolution operator.

## Unified map

```text
3-D geometry/materials + operating condition + T0 + arbitrary query time
                              |
                              v
                 compatible EM + thermal spatial operators
                              |
                 +------------+------------+
                 |                         |
                 v                         v
        snapshot-free reduced EM      certified thermal spectrum
                 |                         |
                 +------ quasi-static -----+
                         EM elimination
                              |
                              v
                  q_em,r(a), dq_em,r/da
                              |
                              v
                 analytic neural evolution
                              |
                              v
          a(t), T(x,t), Z/R/L/M, P_Cu, P_sea
                              |
                              v
              state + electromagnetic output certificates
```

## Core design

1. **Full-equation electromagnetic origin.** Copper skin/proximity effects and seawater induced-current losses originate from the field solution rather than from add-on resistance formulas.
2. **Deterministic electromagnetic reduction without solution snapshots.** The electromagnetic reduced space is generated from the physical operator and residual Riesz lifts; no Maxwell/FEM solution snapshot is required.
3. **Shared 3-D spatial media.** The executable core constructs an orthogonal 3-D cell complex shared by electromagnetic and thermal physics, with cellwise conductor/package/seawater material fields.
4. **Exact compatible topology and gauges.** The incidence matrices satisfy `C @ G = 0`; a deterministic tree-cotree magnetic gauge and one scalar-potential reference per conducting component remove nullspaces without penalty parameters or rank thresholds.
5. **Reciprocal electromagnetic coordinates.** With `psi = phi/(j omega)`, the gauge-eliminated `A-psi` field matrix is complex symmetric for reciprocal real material Hodge operators, so reciprocity is structural rather than imposed afterwards.
6. **Exact nonlinear temperature feedback.** The nonlinear spatial core reconstructs `T(a)` and evaluates analytic material laws directly, including the reciprocal copper conductivity induced by a linear resistivity law. An affine conductivity approximation is not required.
7. **Physical residual norm.** The electromagnetic Riesz metric is magnetic energy plus Joule energy dissipated per electrical radian. Residual lifts and basis orthogonalisation are performed in Cholesky Riesz coordinates.
8. **Direct heat-source and loss outputs.** Joule losses are projected directly to thermal coordinates, while `P_Cu` and `P_sea` are obtained from the same field state and regional Hodge operators.
9. **Multiport field outputs.** Joint multi-RHS reduction supports reciprocal `Z`, `R`, `L`, and `M` matrices; passivity, reciprocity, port-power/Joule-power consistency, and residual-to-output error bounds are executable checks.
10. **Analytic neural dynamics.** Response neurons form arbitrary-depth analytic DAGs. A fast polynomial-exponential compiler and an independent resonance-stable state-space realization backend evaluate the same network with no time stepping.
11. **Residual-grown topology.** Candidate analytic neurons are selected from unresolved physical residual directions using the exact reduced heat-source Jacobian and full nonlinear residual re-evaluation.
12. **Certified ranks/domains.** Thermal rank can be selected from a spectral-tail bound; affine electromagnetic thermal-state boxes can be certified by branch-and-bound residual upper bounds. Unresolved work budgets return `indeterminate`, never a false certificate.

## Reduced coupled system

After spatial reduction,

```text
A_em,r(a; mu) c = b_em,r(mu),
c(a;mu) = A_em,r(a;mu)^(-1) b_em,r(mu),
q_em,r(a;mu) = q_Cu,r + q_sea,r,
da/dt + Lambda_T a = g_em(a;mu).
```

The analytic network evaluates

```text
a(t) = N_analytic(mu, a0, t)
```

directly at arbitrary `t`, with no thermal time marching during inference. Training/growth uses

```text
R = da/dt + Lambda_T a - g_em(a;mu).
```

## Electromagnetic reduction and certification

For `H = L L^H`, the residual dual norm is

```text
||r||_(H^-1) = ||L^-1 r||_2.
```

One common reduced space is grown over the joint set

```text
thermal state x port/excitation RHS.
```

For affine thermal-state dependence, the current branch-and-bound certificate proves a residual upper bound over a continuous parameter box or returns `violated` / `indeterminate` rather than treating a finite sample set as a proof.

For multiport outputs, the field stability constant

```text
beta_em = sigma_min(L^-1 A L^-H)
```

converts port residuals into entrywise bounds for `Z`, `R`, and `L/M`.

## Thermal rank certificate

For the M-orthonormal thermal spectrum, if the first omitted eigenvalue is `lambda_(r+1)`, the omitted initial M-norm is `E0`, and the certified source dual bound is `Q`, then

```text
||T_tail(t)||_M
<= exp(-lambda_(r+1)t) E0
 + (1-exp(-lambda_(r+1)t)) Q/lambda_(r+1).
```

The verification implementation chooses the smallest rank satisfying the requested state or output tolerance. The additional nonlinear reduced-dynamics error is certified separately by the coupled residual/contraction analysis.

## Run

```bash
python -m pip install -e '.[dev]'
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q
```

GitHub Actions also runs the full test suite on every push to `main` and every pull request targeting `main`.

## Documentation

- [`docs/SDFMPNEO_theory.tex`](docs/SDFMPNEO_theory.tex): governing equations, deterministic reduction, analytic-neuron theory, stability, residual growth, and unified error framework.
- [`docs/SDFMPNEO_implementation.md`](docs/SDFMPNEO_implementation.md): software architecture and implementation contract.
- [`docs/MVP_CORE.md`](docs/MVP_CORE.md): exact current executable status and remaining obligations.
- [`docs/COMPATIBLE_EM.md`](docs/COMPATIBLE_EM.md): reciprocal gauge-eliminated `A-psi` electromagnetic formulation, multiport outputs, and field certificates.
- [`docs/SPATIAL_3D.md`](docs/SPATIAL_3D.md): shared 3-D electromagnetic–thermal spatial discretization.
- [`docs/NONLINEAR_CONSTITUTIVE.md`](docs/NONLINEAR_CONSTITUTIVE.md): direct nonlinear temperature-dependent material laws.

## Current implementation layers

- `sdfmpneo/em/grid3d.py`: orthogonal 3-D node-edge-face complex, tree-cotree gauge, and material Hodge assembly
- `sdfmpneo/em/compatible.py`: reciprocal gauge-eliminated magnetoquasistatic `A-psi` operator
- `sdfmpneo/em/nonlinear.py`: exact nonlinear temperature-dependent spatial electromagnetic problem
- `sdfmpneo/em/reduced.py`: Cholesky-Riesz snapshot-free single/multi-RHS electromagnetic reduction
- `sdfmpneo/em/ports.py`: field-derived multiport `Z/R/L/M`
- `sdfmpneo/em/diagnostics.py`: region-separated Joule diagnostics
- `sdfmpneo/thermal/grid3d.py`: heterogeneous 3-D finite-volume thermal operator
- `sdfmpneo/thermal/spectral.py`: deterministic thermal spectrum and tail-certified rank selection
- `sdfmpneo/analytic`: analytic neural DAG, fast compiler, and resonance-stable state-space realization
- `sdfmpneo/training`: physical residual and residual-driven analytic-network growth
- `sdfmpneo/certification`: contraction/state bounds, affine continuous-domain EM residual bounds, and residual-to-port-output bounds
- `sdfmpneo/model.py`: unified executable fixed-operating-condition online query interface

## Non-negotiable scope rules

- No labelled FEM/Maxwell/experimental solution data in training.
- No full-order solution snapshots are required to construct the reduced electromagnetic space.
- No geometry-specific impedance solver is a core dependency.
- No empirical near/far split, fixed neural width/depth, artificial thermal time constant, or empirical constitutive correction is part of the method definition.
- Copper and seawater losses originate from the electromagnetic field solution and conductivity Hodge operators.
- Fluid velocity is outside the current model scope; seawater remains an explicit electromagnetic and thermal medium.
- Full-order simulations and experiments are validation tools only, not training-label generators.

## Important current limits

The current real 3-D geometry engine is rectilinear, not yet a curved unstructured coil/package mesh. End-to-end sparse high-contrast field solution with verified linear-solve error, continuous certification for nonlinear constitutive/geometry/frequency parameters, scalable low-spectrum extraction with a verified first-omitted eigenvalue bound, certified open/seawater outer-domain treatment, solid-conductor terminal-current ports, and a fully parameter-conditioned analytic network over arbitrary operating condition `U` remain active implementation tasks. See `docs/MVP_CORE.md` for the precise status.
