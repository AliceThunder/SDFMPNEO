# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator for Underwater WPT**

SDF-MPNEO is a solution-data-free electromagnetic–thermal surrogate framework for underwater wireless power transfer. Conductor, package, seawater and passive media remain explicit spatial physics. Fluid velocity is outside the present scope.

Version **0.9.0** closes the main software-integration gaps identified after 0.8.1: factorization-free Riesz is now the nonlinear production default, top-level thermal truncation can use a certified partial eigenspectrum, geometry can be organized as a discrete union of independently certified charts, error propagation is typed by physical layer/norm, CAD boundary heuristics have been removed, and external validation cases require traceable provenance.

The current executable path is

```text
UWPT chart / CAD / materials + a0 + U + arbitrary t
                         |
                         v
              tagged tetrahedral physics
                         |
          +--------------+--------------+
          |                             |
          v                             v
 compatible nonlinear EM          P1 thermal operator
          |                             |
          v                             v
 H=K+D physical metric        certified partial spectrum
          |                             |
          v                             |
 factorization-free Riesz              |
          |                             |
          v                             |
 snapshot-free EM ROM -----------------+
          |
          v
 q_em(a,U), dq_em/da, Z/R/L/M, P_Cu, P_sea
          |
          v
 electrothermal vector field F_G(a,U)
          |
          v
 residual-grown intrinsic analytic DAG
          |
          v
       (G,U,a0,t) -> a(t)
          |
          v
 T(x,t), Tmax and certified outputs
          |
          v
 typed deterministic error propagation
```

## Scientific rules

The implementation follows these non-negotiable rules:

- no labelled FEM/Maxwell/COMSOL/experimental solution data are used for training;
- no full-order electromagnetic solution snapshots are required for the production ROM basis;
- no geometry-specific impedance solver is a core dependency;
- ranks, approximation orders and acceptance criteria are driven by governing physics or explicit certificates rather than unexplained hyperparameters;
- a missing proof returns an uncertified/indeterminate result instead of being repaired by a fitted factor;
- external solver/experimental data are validation-only and must remain excluded from training.

## 0.9 production defaults

### Factorization-free nonlinear Riesz path

For the gauge-reduced reciprocal tetrahedral operator

```text
A(a) = K + i D(a),
H(a) = K + D(a),
K,D >= 0,
```

physical coercivity gives

```text
beta_H >= 1/sqrt(2),
||e||_H <= sqrt(2) ||r||_(H^-1).
```

`TetrahedralElectroThermalCore.build_reduced_electromagnetics()` now selects the **Morse face-circulation magnetic auxiliary + state-aware conductive scalar-tree** preconditioner by default. The resulting outer `CertifiedPCGRieszAction` is factorization-free in the production ROM path. Complete sparse LU is retained only for explicit compatibility/correctness backends and separate verification fallbacks.

The production default is regression-tested with `scipy.sparse.linalg.splu` disabled.

### Certified partial thermal spectrum

When a thermal truncation request supplies all of

```text
initial_temperature_deviation_free,
source_dual_bound,
requested_state_tolerance,
```

the top-level core first tries a **partial sparse eigensolve**. Retained rank is not supplied as a free hyperparameter: the code starts from the smallest admissible rank and accepts the first rank whose rigorous thermal-tail bound is below the requested tolerance.

The first omitted thermal eigenvalue has an independent lower bound. The built-in path uses a conservative Li-Yau bound from domain volume and material extrema. If that proof cannot certify the request, the implementation falls back to the full discrete spectrum rather than weakening the theorem.

The selected backend is recorded as

```text
partial_certified
full_certified_fallback
full_discrete
```

so downstream studies can report exactly how the thermal basis was obtained.

## Compatible electromagnetic–thermal core

The unstructured path uses first-order Nedelec edge elements for electromagnetics and P1 finite elements for thermal physics on the same tetrahedral mesh. Exact incidence satisfies

```text
C @ G = 0.
```

Tree-cotree magnetic gauge elimination and one scalar-potential reference per conducting component remove nullspaces without penalty parameters or numerical rank tuning.

With

```text
psi = phi/(j omega),
```

reciprocal real materials yield a complex-symmetric `A-psi` field matrix, making reciprocity structural instead of an output post-processing operation.

For closed one-ampere port sources collected in `B`,

```text
A X = B,
Z = j*omega*B^T X,
R = Re(Z),
L = Im(Z)/omega.
```

The same field state supplies copper and seawater Joule powers, and regression tests enforce reciprocity, passivity and port-power/Joule-power closure.

`SolidTerminalPortSet` additionally supports terminal-current constrained solid conductors through P1-consistent terminal surface loads with exact zero-net-current balance.

## Nonlinear material coupling

Copper uses reciprocal conductivity induced by linear resistivity,

```text
rho(T) = rho_ref [1 + alpha(T-T_ref)],
sigma(T) = sigma_ref / [1 + alpha(T-T_ref)].
```

Inside each tetrahedron the denominator is P1. The reciprocal and its derivative are represented by certified barycentric geometric series. The minimum series order is chosen from the declared pointwise remainder budget, and retained polynomial terms are integrated analytically against Nedelec fields.

The same constitutive representation is used by

- nonlinear field assembly;
- `dA/da`;
- projected Joule heat source `q_em`;
- exact heat-source Jacobian `dq_em/da`;
- regional copper/seawater loss evaluation.

Constitutive remainder is propagated through an inverse-perturbation theorem. If the perturbation condition fails, the result remains uncertified.

## Snapshot-free electromagnetic reduction

For reduced basis `V`,

```text
A_r(a) = V^H A(a) V,
b_r = V^H b,
x_r(a) = V A_r(a)^-1 b_r.
```

The unresolved full-space residual is

```text
r(a) = b - A(a) x_r(a),
```

and the local physical metric gives

```text
eta(a) = sqrt(2) ||r(a)||_(H(a)^-1).
```

A certified Riesz action supplies both an approximate lift and a rigorous dual-norm interval. The unresolved state/excitation generates the next enrichment; no full-order equilibrium solution `A(a)^-1 b` is used as a basis snapshot.

For a joint multiport source matrix, every port column participates in one residual-greedy basis construction, and each reduced column is independently certified. The same ROM directly supplies certified `Z/R/L/M` and projected heat-source quantities.

## Intrinsic analytic neural evolution

The analytic model is not a time-stepper and not a hypernetwork. Initial coordinates and static parameters are zero-dynamics analytic nodes inside one fixed DAG. Within one continuous geometry chart the graph represents

```text
(G,U,a0,t) -> a(t)
```

directly at arbitrary `t`.

The physical residual is

```text
R = da/dt - F_G(a,U)
  = da/dt + Lambda_G a - g_em,G(a,U).
```

Candidate response neurons are proposed from unresolved physical residual/tangent information and accepted only after complete nonlinear residual re-evaluation. Discrete construction samples may propose candidates, but the stopping criterion is a caller-supplied continuous-domain certificate.

The graph has both a fast polynomial-exponential compiler and an independent state-space realization backend, so exact and near resonance do not rely on a closeness threshold.

## Multi-chart geometry

A single fixed-connectivity affine chart is not treated as a universal geometry model. SDF-MPNEO 0.9 introduces

```text
CertifiedMultiChartGeometryFamily
MultiChartGeometryAnalyticEvolutionOperator
```

to represent

```text
G = union_s G_s.
```

Each chart may own a different reference mesh, FE dimension and geometry parameter list. It keeps its own certified local thermal atlas and analytic operator. Charts must expose a compatible retained thermal rank and physical operating interface before they can be combined.

This allows, for example, circle and rounded-square/remeshed geometry families to remain distinct certified charts instead of forcing a topology change into one fictitious affine deformation. The implementation deliberately performs **no cross-chart interpolation of network weights**.

Within a chart, the existing `AffineTetrahedralGeometryChart` proves non-degeneracy and H(curl)/P1 quadratic-form distortion bounds over continuous parameter boxes. `CertifiedGeometryElectroThermalFamily` aligns retained thermal spectral subspaces within compatible reference-domain charts.

## Continuous-domain certification

Executable continuous-domain certificates include:

- affine reduced EM parameter boxes;
- nonlinear thermal-state EM boxes;
- thermal state + source + frequency operating boxes;
- fixed-topology affine geometry families;
- electrothermal state/operating Jacobian and contraction bounds;
- continuous analytic residual bounds;
- continuous `[a0,G,U,t]` cross-geometry analytic residual branches when the required physical geometry derivative proof is supplied.

`certify_electrothermal_domain_bounds()` provides a certified bound for the complete physical state Jacobian

```text
F(a,U) = -Lambda a + q_em(a,U),
||dF/da|| <= ||Lambda|| + ||dq_em/da||.
```

`GeometryDerivativeProof` carries theorem-derived `|dF/dG|` bounds for a declared chart/domain, and `compose_geometry_physical_lipschitz_proof()` combines geometry, state, operating and contraction bounds into the proof object consumed by the cross-geometry branch-and-bound certifier.

A missing chart-wise geometry derivative theorem remains an explicit proof obligation; the composer does not infer one from finite differences.

## Typed error propagation

Error terms are no longer allowed to be added merely because they are finite scalars. `ProofNode` carries

```text
name,
physical layer,
norm,
bound,
provenance,
optional proved propagation target.
```

`compose_typed_error_certificate()` rejects incompatible accumulation layers/norms.

For electrothermal propagation, EM/constitutive/algebraic/mesh/outer terms must first be converted by proved sensitivities to a common thermal-residual/heat-source norm. Analytic residual is accumulated in that same norm, then the exact comparison-equation gain is applied:

```text
G(T,kappa) = (1-exp(-kappa*T))/kappa,
G(T,0) = T.
```

Only after this residual-to-state propagation may direct thermal-ROM/thermal-outer state errors be added in the thermal-state norm. Output sensitivities then propagate to `T(x,t)`, `Tmax`, or other requested quantities.

The long-time gain is accepted only when the contraction margin is positive.

## Spatial and infinite-domain certificates

The spatial layer includes fail-closed interfaces for

- conforming H(curl)-P1 mesh approximation error;
- propagation of field-energy error to multiport and Joule outputs;
- conductive infinite-seawater electromagnetic outer-domain truncation;
- homogeneous infinite-seawater thermal truncation via heat-kernel/Newton-potential bounds and the parabolic maximum principle.

A mesh refinement sequence is **not** automatically promoted to a mathematical mesh-error certificate. Problem-specific regularity and reliability/interpolation constants must have theorem-level provenance.

## CAD and mesh pipeline

The optional automatic CAD dependency is declared as

```bash
python -m pip install -e '.[cad]'
```

The Gmsh pipeline builds round or rounded-square spiral conductors, rectangular conductor sections, package volumes, seawater domain and terminal/outer physical groups.

The old center-of-mass threshold for seawater outer-boundary identification has been removed. The outer artificial boundary is identified from the OCC spherical surface type. Terminal entities are resolved on preserved conductor CAD before Boolean operations, checked for uniqueness, and verified to remain part of the conductor boundary afterwards. Ambiguous topology is rejected instead of resolved by an empirical distance cutoff.

## Independent validation

External validation is explicitly separated from training. Maxwell, COMSOL or experiment cases require a `ValidationEvidenceManifest` with

```text
reference_kind
source_identifier
source_version
provenance
used_for_training = False
```

and cannot be marked as training data. Validation reports preserve source identity/version and can test whether external references lie inside declared model certificates.

The repository does **not** fabricate Maxwell/COMSOL/experimental evidence. The harness is implemented; real independent datasets remain a publication/experimental deliverable.

## Numerical fail-closed policy

The CI suite treats

```text
numpy.exceptions.ComplexWarning
scipy.linalg.LinAlgWarning
```

as errors.

Complex-dtype topology matrices are verified to be exactly real integer matrices before conversion to integer sparse form. Any imaginary or non-integer entry is rejected. Exact singular local LU pivots are converted to deterministic `ValueError` instead of allowing a warning and a corrupted factor to continue.

## Run

```bash
python -m pip install -e '.[dev]'
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q \
  -W error::numpy.exceptions.ComplexWarning \
  -W error::scipy.linalg.LinAlgWarning
```

For optional automatic Gmsh CAD/meshing:

```bash
python -m pip install -e '.[dev,cad]'
```

GitHub Actions runs the strict repository-wide regression suite on every push to `main` and pull request targeting `main`.

## Main implementation layers

- `sdfmpneo/spatial/tetra3d.py`: unstructured tetrahedral topology, Nedelec geometry and P1 thermal assembly
- `sdfmpneo/spatial/uwpt_geometry.py`: round/rounded-square UWPT geometry definitions and tagged mesh import
- `sdfmpneo/spatial/gmsh_pipeline.py`: optional automatic OCC/Gmsh CAD and conforming meshing
- `sdfmpneo/spatial/geometry_chart.py`: certified fixed-connectivity affine geometry chart
- `sdfmpneo/em/tetra_nonlinear.py`: certified nonlinear sparse tetrahedral `A-psi` material/field path
- `sdfmpneo/em/reciprocal_series.py`: rigorous reciprocal copper series and derivative remainder bounds
- `sdfmpneo/em/sparse_reduced.py`: snapshot-free sparse physical-energy EM reduction
- `sdfmpneo/em/certified_riesz.py`: certified PCG Riesz action
- `sdfmpneo/em/morse_face_auxiliary.py`: topology-generated Morse face-circulation magnetic auxiliary
- `sdfmpneo/em/scalar_tree_auxiliary.py`: state-aware conductive scalar-tree auxiliary
- `sdfmpneo/em/morse_block_riesz.py`: factorization-free physical Riesz composition
- `sdfmpneo/em/terminal_ports.py`: solid-conductor terminal-current ports
- `sdfmpneo/thermal/partial_spectral.py`: sparse low-mode solve and omitted-eigenvalue certificate
- `sdfmpneo/thermal/atlas.py`: certified retained thermal spectral-subspace alignment
- `sdfmpneo/analytic`: intrinsic analytic DAG, compiler and stable realization
- `sdfmpneo/analytic/multichart_operator.py`: chart-dispatched analytic evolution
- `sdfmpneo/training`: residual/tangent growth and closed-loop solution-data-free trainer
- `sdfmpneo/geometry_family.py`: continuous chart electrothermal family
- `sdfmpneo/geometry_multichart.py`: finite union of independently certified geometry charts
- `sdfmpneo/certification`: continuous-domain, output, spatial, proof-composition and typed propagation certificates
- `sdfmpneo/validation.py`: independent validation evidence/provenance contract
- `sdfmpneo/tetra_core.py`: top-level tetrahedral electrothermal construction and production EM reduction
- `sdfmpneo/model.py`: fixed/parametric arbitrary-time online query interfaces

## Documentation

- `docs/PRODUCTION_STATUS_0_9.md`: authoritative 0.9 implementation status and remaining real obligations
- `docs/SDFMPNEO_theory.tex`: governing equations and theory
- `docs/SDFMPNEO_implementation.md`: software architecture and implementation contract
- `docs/MVP_CORE.md`: executable-core status, retained for continuity but synchronized with 0.9
- `docs/TETRAHEDRAL_CORE.md`: unstructured Nedelec/P1 field chain
- `docs/SPARSE_ENERGY_SOLVER.md`: physical-energy full-order solve theorem
- `docs/CERTIFIED_RIESZ_ACTION.md`: certified inexact Riesz action contract
- `docs/SPARSE_ENERGY_REDUCTION.md`: snapshot-free sparse residual-Riesz reduction
- `docs/NONLINEAR_CONSTITUTIVE.md`: nonlinear temperature-dependent material laws
- `docs/PARAMETRIC_ANALYTIC.md`: intrinsic analytic network and parameter-domain residual growth

## Remaining real obligations

The software architecture is substantially closed, but the following are still genuine research/evidence tasks:

1. populate traceable independent Maxwell/COMSOL and experimental UWPT validation evidence, kept strictly outside training;
2. provide problem-specific H(curl)-P1 regularity/reliability constants for each declared material/geometry chart;
3. provide theorem-derived chart-wise geometry derivative bounds needed to close every intended `[a0,G,U,t]` physical proof automatically;
4. demonstrate memory/iteration/rank/certificate scaling on realistically sized UWPT CAD meshes;
5. add certified compression/minimalization of large analytic state-space realizations;
6. establish a global convergence/completeness theorem for residual-grown analytic-network construction.

These are explicit obligations, not hidden safety factors. Unsupported claims remain uncertified.
