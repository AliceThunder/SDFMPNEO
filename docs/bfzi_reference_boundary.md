# BFZI Reference Boundary for SDF-MPNEO

BFZI is used as a design reference for deterministic electromagnetic computation. SDF-MPNEO does **not** copy BFZI source code, module structure, function names, sample tables, fitted constants, quadrature implementations, solver tolerances, cache files, or version-specific correction formulas.

## General principles retained

### 1. Separate geometry compression from environment response

Repeated coil geometry should be represented through a compact structural description before any expensive field operator is evaluated. The reduced model therefore treats source generation and environment response as separate backend responsibilities.

SDF-MPNEO generalises this further: structural geometry descriptors feed a neural/physics trial-space generator, while the environment is exposed only through matrix-free operator actions.

### 2. Exploit regularity without making it part of the model definition

A deterministic backend may exploit regular grids, convolution, hierarchical kernels, affine families, symmetry or repeated-turn structure. These are implementation strategies, not neural-model assumptions.

The SDF-MPNEO core therefore sees only

- `apply_system(V)`,
- `rhs_fields()`,
- `port_feedback(V)`,
- `apply_dissipation(V)`.

FFT Green convolution, DSE, FMM and H2 implementations are interchangeable behind this interface.

### 3. Preserve physical projection explicitly

Environmental current/field spaces must satisfy their physical compatibility constraints before reduction. SDF-MPNEO uses gauge-reduced de Rham generators and runtime topology validation rather than reproducing a particular finite-volume projector.

### 4. Keep global and local physics corrections conceptually separate

Long-range/environment operators and local singular/near-source effects have different numerical structure. The backend may combine separate deterministic kernels, but every contribution must enter the same variational/operator contract and the same independent certification path.

### 5. Cache only signature-stable deterministic quantities

Reusable geometry/operator artifacts must be keyed by all parameters that alter their meaning. A cache miss is recomputed; a signature change creates a distinct artifact. Stale geometry/frequency/material caches must never be silently reused.

### 6. Compare acceleration at module and end-to-end levels

A fast source kernel is not useful if another stage dominates total runtime. SDF-MPNEO benchmarks will therefore report at least

- neural basis generation,
- deterministic seed compilation/cache hit,
- EM operator actions,
- reduced solve,
- thermo-fluid residual/Jacobian,
- certification,
- total forward time.

## Explicitly not copied from BFZI

The following remain BFZI-specific and are not transplanted:

- affine rounded-square branch formulas and finite-comb implementation;
- BFZI source-family classes and data layouts;
- exact Green/self/near-cell formulas used there;
- FFT lattice construction code;
- finite-volume scalar-potential projector implementation;
- GMRES restart/tolerance settings;
- local source-shape mass correction formulas;
- terminal-lead correction code;
- sample definitions, caches, benchmark scripts and versioned output formats;
- names such as FMAS/GSO or BFZI internal configuration keys in new core APIs.

## SDF-MPNEO-specific development

The new method is defined by elements that are not part of BFZI:

1. solution-data-free neural trial-space generation;
2. physics-seed + neural-enrichment decomposition;
3. shared excitation-independent multiport EM basis;
4. de Rham hard constraints and gauge-rank validation;
5. matrix-free projection directly onto learned trial spaces;
6. coupled mixed-dimensional thermal + RANS-SST equilibrium;
7. implicit differentiation through the nonlinear equilibrium;
8. positive solution-free Basis-Operator Cubature;
9. adaptive nested rank with an independent physical certifier.

BFZI can become one deterministic EM backend used for verification/benchmarking, but it is not the definition of SDF-MPNEO.
