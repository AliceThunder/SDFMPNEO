> 版本说明：本文保留先前版本的架构/证明背景。0.10 科研训练与推理入口、修复及运行方式见 [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md)；其中的旧完成状态不作为当前验收结论。

# Executable core status — 0.9

The repository has moved beyond the original MVP. This file is retained for continuity; the authoritative production-status document is `PRODUCTION_STATUS_0_9.md`.

## Run

```bash
python -m pip install -e '.[dev]'
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q \
  -W error::numpy.exceptions.ComplexWarning \
  -W error::scipy.linalg.LinAlgWarning
```

Optional automatic Gmsh CAD/meshing:

```bash
python -m pip install -e '.[dev,cad]'
```

## Current production chain

```text
UWPT geometry/chart + material regions
        |
        +--> exact tetrahedral topology / gauges
        |
        +--> P1 thermal M,K
        |       |
        |       +--> partial certified low spectrum
        |       +--> Li-Yau/external omitted eigenvalue lower bound
        |       +--> deterministic full-spectrum correctness fallback
        |
        +--> Nedelec reciprocal A-psi electromagnetics
                |
                +--> nonlinear copper/seawater D(a)
                +--> H(a)=K+D(a), beta_H >= 1/sqrt(2)
                +--> Morse face magnetic auxiliary
                +--> conductive scalar-tree auxiliary
                +--> certified PCG Riesz action
                +--> snapshot-free multi-RHS residual ROM
                +--> Z/R/L/M + P_Cu/P_sea + q_em,dq_em/da
                         |
                         +--> electrothermal F_G(a,U)
                         +--> residual-grown intrinsic analytic DAG
                         +--> direct arbitrary-time state
                         +--> typed continuous error propagation
```

## Implemented and regression-tested

- orthogonal 3-D and unstructured tetrahedral electromagnetic–thermal correctness paths;
- exact compatible incidence with `C @ G = 0`;
- tree-cotree magnetic gauge and conducting-component scalar-potential references;
- first-order Nedelec electromagnetic assembly and shared P1 thermal assembly;
- reciprocal complex-symmetric `A-psi` field coordinates;
- nonlinear reciprocal copper conductivity without conductivity linearization;
- exact barycentric Joule projection and heat-source Jacobian;
- physical-energy coercivity independent of copper/seawater conductivity contrast;
- factorization-free Morse/scalar-tree Riesz as the **top-level nonlinear production default**;
- snapshot-free sparse single/multi-RHS EM ROM construction;
- certified multiport `Z/R/L/M` and solid terminal-current ports;
- copper/seawater regional loss separation from the same field state;
- partial sparse thermal eigensolution connected to top-level core construction;
- rigorous omitted-mode lower-bound path and fail-closed full-spectrum fallback;
- intrinsic `(a0,U,t)` analytic evolution and chart-wise `(G,U,a0,t)` evolution;
- residual-grown solution-data-free closed-loop trainer;
- continuous nonlinear thermal, source, frequency and fixed-chart geometry certificates;
- continuous cross-geometry analytic residual proof interface;
- finite-time residual-to-state propagation, including `kappa=0` and finite-horizon noncontractive cases;
- mesh/spatial, electromagnetic outer-domain and infinite-water thermal outer-domain certificate interfaces;
- maximum-temperature output sensitivity;
- typed proof nodes that prevent incompatible physical norms/layers from being added;
- automatic composition of electrothermal state/operating bounds with theorem-derived geometry derivative bounds;
- discrete multi-chart atlas for remeshed/topologically distinct geometry families;
- optional Gmsh CAD pipeline without the former empirical outer-radius boundary classifier;
- independent-validation evidence manifests that are explicitly excluded from training;
- warning-as-error CI for complex-to-real loss and singular dense local factors.

## Production defaults

### Electromagnetic Riesz

`TetrahedralElectroThermalCore.build_reduced_electromagnetics()` uses

```text
make_morse_auxiliary_physical_pcg_riesz_factory(problem)
```

unless an older compatibility block-action factory is deliberately supplied. The production ROM therefore does not silently fall back to complete sparse LU.

### Thermal spectrum

With a declared thermal truncation request, the top-level builder attempts a partial spectrum first. Rank is chosen by the thermal tail certificate, not by a caller-provided neural/model width. The current backend is recorded as `partial_certified`, `full_certified_fallback`, or `full_discrete`.

### Error propagation

EM field error is not numerically added directly to thermal-state error. Each contribution must first be propagated to a common declared physical quantity/norm. For thermal residual terms the exact finite-time gain is

```text
G(T,kappa) = (1-exp(-kappa*T))/kappa,
G(T,0) = T.
```

Direct thermal-state truncation terms are added only afterwards in the same state norm, followed by an output sensitivity.

## Geometry status

Inside a compatible fixed-connectivity chart, `AffineTetrahedralGeometryChart` certifies element non-degeneracy and H(curl)/P1 quadratic-form distortion over a continuous parameter box. `CertifiedGeometryElectroThermalFamily` aligns retained thermal subspaces to one chart reference.

For geometry families that require remeshing or a topology/shape-family change, `CertifiedMultiChartGeometryFamily` represents a finite union of independently certified charts. Different charts may have different full FE dimensions and geometry coordinates. No cross-chart interpolation or learned weight generation is introduced.

The software proof composer for `[a0,G,U,t]` is complete, but each intended chart must still provide theorem-derived bounds for `dF/dG`; unsupported charts remain uncertified.

## CAD status

Round and rounded-square spiral definitions, package/seawater geometry, tagged Gmsh import and optional automatic CAD/meshing are present.

The outer seawater boundary is identified from OCC spherical surface type, not from a fitted center-of-mass/radius fraction. Terminal surfaces are identified on persistent copper CAD before Boolean operations and verified afterwards. Ambiguous entity identity fails closed.

`mesh_size` and centerline chord error remain explicit scientific inputs expected to be linked to spatial/geometry error budgets.

## Validation status

`validation.py` provides a strict independent-validation harness. Maxwell, COMSOL and experimental cases require a `ValidationEvidenceManifest` with source identity, version and provenance and are prohibited from being marked as training labels.

This is an **interface and evidence-governance completion**, not a claim that external evidence has already been collected. Actual independent Maxwell/COMSOL/experimental data remain a publication validation deliverable.

## Remaining real obligations

1. Acquire/version independent Maxwell/COMSOL and experimental UWPT evidence over representative geometry/frequency/misalignment/thermal regimes.
2. Provide chart- and material-specific regularity/reliability theorem constants needed for finite certified mesh-discretization errors.
3. Provide theorem-derived geometry derivative bounds for every intended multi-chart parameter domain.
4. Demonstrate factorization-free Riesz, partial thermal spectrum, ROM and analytic-DAG scaling on realistic production meshes.
5. Add certified compression/minimalization of large analytic state-space realizations.
6. Establish a global convergence/completeness theorem for the residual-grown analytic dictionary.

These are research/evidence obligations. They are not replaced by fitted constants, finite-difference derivatives, sampled convergence ratios or hidden tolerances.
