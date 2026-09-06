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
        snapshot-free reduced EM      thermal spectral space
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
                 a(t), T(x,t), Z, losses
                              |
                              v
                         certificates
```

## Core design

1. **Full-equation electromagnetic origin.** Copper skin/proximity effects and seawater induced-current losses are intended to arise from the electromagnetic field solution rather than from add-on resistance formulas.
2. **Deterministic electromagnetic reduction without solution snapshots.** The electromagnetic reduced space is generated from the physical operator and residual Riesz lifts; no Maxwell/FEM solution snapshot is required.
3. **Shared 3-D spatial media.** The executable core now constructs a real orthogonal 3-D cell complex shared by electromagnetic and thermal physics, with cellwise conductor/package/seawater material fields.
4. **Exact compatible topology.** The generated incidence matrices satisfy `C @ G = 0`; magnetic and scalar potential nullspaces are removed by coordinates, not penalty parameters.
5. **Physical residual norm.** The 3-D electromagnetic Riesz metric is derived from magnetic energy plus Joule energy dissipated per electrical radian. Residual lifts and basis orthogonalisation are performed in Cholesky Riesz coordinates.
6. **Direct reduced heat-source projection.** Joule losses are projected directly to thermal reduced coordinates; full 3-D loss fields are not required online.
7. **Region-separated diagnostics.** Copper and seawater Joule powers can be evaluated from disjoint cell masks using the same field solution and Hodge operators.
8. **Analytic neural dynamics.** Every response neuron is compiled to a closed-form polynomial–exponential function; arbitrary-depth response chains remain inside the same analytic algebra.
9. **Residual-grown topology.** Candidate analytic neurons are scored using the physical residual and exact reduced electromagnetic heat-source Jacobian, with full nonlinear residual re-evaluation before acceptance.
10. **No empirical model partitioning.** No empirical near/far split, fixed neural width/depth, artificial thermal time constant, or geometry-specific impedance correction is part of the method definition.

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

## Snapshot-free electromagnetic reduced-space construction

```text
source-compatible initial Riesz direction
              |
              v
reduced EM solve over verification states
              |
              v
full operator residual r = b - A V c
              |
              v
dual coordinate y = L^-1 r,  H = L L^H
              |
              v
H-normalized lift q = L^-H y / ||y||
              |
              v
enrich V and repeat until the required certificate is met
```

The current executable verifier still uses a deterministic finite candidate set; continuous geometry/temperature-domain certification remains a pending theory-to-code item.

## Run

```bash
python -m pip install -e .
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q
```

## Documentation

- [`docs/SDFMPNEO_theory.tex`](docs/SDFMPNEO_theory.tex): governing equations, deterministic reduction, analytic-neuron theory, stability, residual growth, and unified error framework.
- [`docs/SDFMPNEO_implementation.md`](docs/SDFMPNEO_implementation.md): software architecture and implementation contract.
- [`docs/MVP_CORE.md`](docs/MVP_CORE.md): exact current executable status and remaining obligations.
- [`docs/COMPATIBLE_EM.md`](docs/COMPATIBLE_EM.md): compatible gauge-eliminated `A-phi` electromagnetic formulation.
- [`docs/SPATIAL_3D.md`](docs/SPATIAL_3D.md): first real 3-D shared electromagnetic–thermal spatial discretization.

## Current implementation layers

- `sdfmpneo/em/grid3d.py`: real orthogonal 3-D node-edge-face complex and material Hodge assembly
- `sdfmpneo/em/compatible.py`: compatible gauge-eliminated magnetoquasistatic `A-phi` operator
- `sdfmpneo/em/reduced.py`: Cholesky-Riesz snapshot-free electromagnetic reduction
- `sdfmpneo/em/diagnostics.py`: region-separated Joule diagnostics
- `sdfmpneo/thermal/grid3d.py`: heterogeneous 3-D finite-volume thermal operator
- `sdfmpneo/thermal/spectral.py`: deterministic thermal spectral coordinates
- `sdfmpneo/analytic`: analytic polynomial–exponential neural algebra and compiled DAG
- `sdfmpneo/training`: physical residual and residual-driven analytic-network growth
- `sdfmpneo/certification`: contraction and state-error certificate interfaces

## Non-negotiable scope rules

- No labelled FEM/Maxwell/experimental solution data in training.
- No full-order solution snapshots are required to construct the reduced electromagnetic space.
- No geometry-specific impedance solver is a core dependency.
- No empirical near/far split, fixed neural width/depth, artificial thermal time constant, or empirical constitutive correction is part of the method definition.
- Copper and seawater loss diagnostics originate from the electromagnetic field solution and conductivity Hodge operators.
- Fluid velocity is outside the current model scope; seawater remains an explicit electromagnetic and thermal medium.
- Full-order simulations and experiments are validation tools only, not training-label generators.

## Important current limits

The present real 3-D core is rectilinear, not yet a curved unstructured coil/package mesh. Dense gauge null-space construction, true nonlinear copper/seawater constitutive separation, continuous parameter-domain certification, scalable thermal spectral-tail selection, and a verified large-scale high-contrast linear solver remain active implementation tasks. See `docs/MVP_CORE.md` for the precise status.
