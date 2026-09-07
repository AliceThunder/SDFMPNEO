# SDF-MPNEO 0.9 production status

This document is the authoritative implementation-status snapshot for the 0.9.x code path. It distinguishes executable/certified code from proof or validation evidence that still has to be supplied for a specific underwater-WPT study.

## Executable production chain

```text
UWPT chart / CAD / material definition
        |
        +--> tagged conforming tetrahedral mesh
        |       |
        |       +--> exact incidence + gauges
        |       +--> P1 thermal M,K
        |       +--> partial certified thermal spectrum
        |       +--> first-order Nedelec A-psi EM
        |
        +--> nonlinear reciprocal copper + seawater conductivity
        |       |
        |       +--> sparse A(a)=K+iD(a)
        |       +--> physical H(a)=K+D(a)
        |       +--> Morse face-circulation magnetic auxiliary
        |       +--> state-aware conductive scalar tree
        |       +--> certified PCG Riesz action
        |       +--> snapshot-free residual-grown EM ROM
        |       +--> multiport Z/R/L/M + regional Joule power
        |
        +--> electrothermal vector field F_G(a,U)
        |       |
        |       +--> exact q_em and dq_em/da
        |       +--> continuous-domain physical bounds
        |
        +--> residual-grown intrinsic analytic DAG
        |       |
        |       +--> (G,U,a0,t) -> a(t) within one chart
        |       +--> multi-chart dispatch across remeshed shape families
        |       +--> arbitrary-time stable realization
        |
        +--> typed certificates
                |
                +--> EM/constitutive/algebraic/spatial -> thermal residual
                +--> finite-time residual -> thermal state
                +--> thermal state -> Tmax / temperature outputs
                +--> field state -> Z/R/L/M / heat-source outputs
```

## Production defaults that changed in 0.9.0

### Factorization-free Riesz default

`TetrahedralElectroThermalCore.build_reduced_electromagnetics()` now uses the factorization-free Morse physical auxiliary factory by default. The magnetic action is generated from the dual-tree face-circulation topology; the scalar conductive action is state-aware. Complete sparse LU is no longer silently selected by the nonlinear production entry. The older physical-block factory remains available only when a caller explicitly supplies a compatibility block-action factory, and full-order sparse LU remains a correctness/verification fallback outside the online ROM equilibrium.

### Partial thermal spectrum default under a truncation request

When `initial_temperature_deviation_free`, `source_dual_bound`, and `requested_state_tolerance` are supplied together, the top-level core first attempts a partial sparse eigensolve. Rank is not a user hyperparameter: the code starts from the smallest admissible rank and accepts the first rank whose thermal-tail theorem is certified using an independent lower bound for the first omitted eigenvalue. The built-in lower bound is Li-Yau from domain volume and material extrema. If that conservative proof cannot meet the requested tolerance, the code falls back to the full discrete spectrum rather than weakening the certificate.

The core records `thermal_spectrum_backend` as one of:

- `partial_certified`;
- `full_certified_fallback`;
- `full_discrete`.

`full_thermal_spectrum` is therefore intentionally `None` when a true partial path was sufficient.

### Multi-chart geometry atlas

A single fixed-connectivity affine geometry chart is no longer presented as a universal geometry model. `CertifiedMultiChartGeometryFamily` represents

```text
G = union_s G_s
```

where each `G_s` is an independently certified geometry/operator chart with its own reference mesh and local thermal atlas. Different charts may have different full FE dimensions and different geometry parameter lists. They must expose a compatible retained thermal rank before being combined. `MultiChartGeometryAnalyticEvolutionOperator` dispatches chart-specific fixed analytic DAGs without interpolating weights or pretending a discrete topology change is continuous.

Within an individual chart, `GeometryConditionedAnalyticEvolutionOperator` continues to represent one intrinsic analytic map

```text
(G,U,a0,t) -> a(t)
```

and physical geometry dependence is evaluated by the actual chart field, not by a network that generates another network's parameters.

## Continuous-domain proof composition

`certify_electrothermal_domain_bounds()` now exposes both the Joule-source Jacobian bound and a bound for the complete physical state Jacobian

```text
F(a,U) = -Lambda a + q_em(a,U),
||dF/da|| <= ||Lambda|| + ||dq_em/da||.
```

`GeometryDerivativeProof` carries theorem-derived `|dF/dG|` bounds for one declared chart/domain. `compose_geometry_physical_lipschitz_proof()` combines these with continuous electrothermal state/operating bounds and the contraction margin into the exact `GeometryPhysicalLipschitzProof` required by the `[a0,G,U,t]` analytic residual branch-and-bound certifier.

This closes the software composition gap. It does **not** fabricate a missing geometry theorem: if a particular chart has no certified geometry derivative provider, that cross-geometry branch remains uncertified.

## Typed error propagation

`ProofNode` now records the physical `layer` and `norm` of every bound. `compose_typed_error_certificate()` refuses to add nodes unless they already share, or have a proved scalar propagation to, one common accumulation quantity.

For the electrothermal chain, `compose_electrothermal_error_certificate()` enforces the following order:

1. EM/constitutive/algebraic/mesh/outer terms must first be propagated to a common thermal-residual / heat-source norm by the corresponding proved output sensitivity.
2. Analytic-network residual is accumulated in that same thermal-residual norm.
3. The exact comparison-equation gain is applied:

   ```text
   G(T,kappa) = (1-exp(-kappa*T))/kappa,
   G(T,0) = T.
   ```

   The long-time form is allowed only for positive contraction margin.
4. Direct thermal-ROM / thermal outer-state terms may then be added in the declared thermal-state norm.
5. Output sensitivities propagate the resulting state bound to temperature/Tmax or other requested quantities.

This prevents field-energy error, heat-source error and thermal-state error from being silently summed as dimensionless numbers.

## CAD and mesh topology handling

The optional CAD extra is declared as

```bash
python -m pip install -e '.[cad]'
```

The Gmsh pipeline no longer identifies the seawater artificial boundary using a fitted fraction of the water radius. It identifies OCC spherical boundary entities by CAD surface type and fails if such an entity is absent. Coil terminal surfaces are resolved on the preserved conductor CAD before Boolean operations and their entity membership is checked after the operations. A floating-point tie in endpoint-based terminal identification is rejected instead of broken with an empirical distance threshold.

Geometry/mesh discretization error remains a separate certificate. A real UWPT chart is certified only if its regularity and interpolation/reliability constants have a valid theorem/provenance; mesh refinement observations are not promoted to proof constants.

## Numerical fail-closed policy

The CI suite treats both

```text
numpy.exceptions.ComplexWarning
scipy.linalg.LinAlgWarning
```

as errors.

Topology matrices that are stored in complex dtype are validated as exactly real integer matrices before construction of an integer sparse matrix. A nonzero imaginary component or non-integer entry is a topology failure. Local dense LU blocks convert LAPACK's exact-zero-pivot warning into a deterministic `ValueError`; the code never continues with a singular factor and never relies on warning suppression.

## Independent validation contract

External evidence is deliberately separated from training. A Maxwell, COMSOL or experimental `IndependentValidationCase` requires a `ValidationEvidenceManifest` containing:

- reference kind;
- source identifier;
- source version;
- provenance;
- explicit `used_for_training=False`.

The report records source identity/version and tests whether the reference is inside the model's declared absolute certificate when such a bound is available.

The repository does **not** claim that independent external evidence exists merely because the harness exists. Real solver/experimental datasets must be generated or supplied independently and versioned before publication claims are made.

## What is still an actual obligation

The following items remain real work rather than already-implemented software tasks:

1. **Independent validation evidence.** Populate traceable Maxwell/COMSOL and experimental cases across representative geometry, frequency, offset/misalignment, source and thermal conditions. They must stay excluded from training.
2. **Problem-specific mesh theorem data.** Prove or provide the regularity/reliability constants for each material/geometry chart required for a finite certified spatial-discretization term.
3. **Geometry derivative theorems.** Supply theorem-derived chart-wise `dF/dG` bounds so the implemented proof composer can automatically certify all intended geometry domains.
4. **Production-scale evidence.** Run realistic UWPT meshes and report memory, iteration counts, ROM rank, thermal rank, analytic-DAG size and certificate tightness. Correctness-scale tests are not a substitute for a scaling study.
5. **Analytic realization compression.** Add certified redundancy removal/minimalization for large polynomial-exponential/state-space realizations.
6. **Global growth convergence.** Establish a theorem for convergence/completeness of the residual-grown analytic dictionary under the declared physical domain assumptions.

## Validation of the codebase

The repository-wide GitHub Actions test command is intentionally strict:

```bash
pytest -q \
  -W error::numpy.exceptions.ComplexWarning \
  -W error::scipy.linalg.LinAlgWarning
```

The 0.9 production closure added regression tests for:

- top-level no-sparse-LU nonlinear Riesz default;
- top-level partial thermal spectrum selection;
- CAD entity fail-closed identification;
- typed electrothermal error propagation;
- geometry physical-proof composition;
- multi-chart geometry/operator dispatch;
- validation evidence provenance;
- exact topology conversion and singular local-LU failure.

A green CI therefore verifies both legacy physics regressions and the new production-closure contracts. It does not replace independent physical validation.
