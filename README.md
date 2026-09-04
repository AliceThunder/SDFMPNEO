# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Equilibrium Operator for Underwater WPT**

SDF-MPNEO is a physics-embedded neural reduced-order framework for strongly coupled electromagnetic–thermal–fluid modelling of underwater wireless power transfer systems.

The neural network does **not** regress impedance, temperature, or flow solutions from labelled FEM/CFD data. It generates geometry-conditioned admissible trial-space enrichments around deterministic solution-free physics seeds; the physical states are then obtained by solving reduced governing equations.

---

## Documentation map

The repository documentation is deliberately organized so the same architecture can guide **PPT preparation, paper writing, and code implementation** without the three descriptions diverging.

### 1. Master design — authoritative architecture

[`docs/master_design.md`](docs/master_design.md)

Use this as the **single source of truth** for:

- physical states and coupling directions;
- mathematical architecture;
- solution-data-free training definition;
- physics seed + neural enrichment;
- de Rham hard constraints;
- matrix-free EM formulation;
- mixed-dimensional thermal model;
- RANS-SST flow model;
- strong coupled equilibrium;
- BOC, certification and adaptive rank;
- development milestones and required invariants.

### 2. PPT guide

[`docs/ppt_guide.md`](docs/ppt_guide.md)

Defines:

- recommended 15-slide and 8-slide storylines;
- what each slide must explain;
- required figures and equations;
- contribution framing;
- statements to avoid.

### 3. Paper-writing guide

[`docs/paper_guide.md`](docs/paper_guide.md)

Maps the architecture to:

- Abstract;
- Introduction;
- Problem Formulation;
- Methods;
- solution-data-free training;
- certification;
- Results and ablations;
- Discussion and Conclusion;
- required figures/tables and terminology.

### 4. Code implementation plan

[`docs/implementation_plan.md`](docs/implementation_plan.md)

Defines:

- target source tree;
- data contracts and tensor shapes;
- geometry/topology/seed/backend interfaces;
- implementation milestones;
- acceptance tests;
- failure handling;
- runtime instrumentation;
- definition of done for the first publishable prototype.

### 5. PPT–paper–code traceability

[`docs/traceability_matrix.md`](docs/traceability_matrix.md)

Maps every core concept to:

```text
PPT slide <-> paper section <-> code/test evidence
```

Any architecture change should update all three surfaces.

### 6. Forward tensor graph

[`docs/forward_graph.md`](docs/forward_graph.md)

Contains the lower-level tensor/dataflow contract used by the current foundation implementation.

### 7. BFZI reference boundary

[`docs/bfzi_reference_boundary.md`](docs/bfzi_reference_boundary.md)

Defines what general computational ideas may be learned from the independent BFZI repository and what BFZI-specific code, formulas, naming, cache logic, and implementation details must **not** be copied into SDF-MPNEO.

---

## Core state

- seawater induced current density `J_s`
- unified conductor/package/seawater temperature state `Theta`
- seawater velocity `v`
- SST turbulence closure states `k_t, omega_t`

Pressure is an internal incompressibility constraint variable when required by a reference discretization; it is not a separate surrogate state.

---

## Core architecture

```text
Unified coil/package geometry
        |
Shared geometry encoder
        |
+-------+----------+
|       |          |
EM      Thermal    Flow-SST
head    head       head
|       |          |
+-------+----------+
        |
Physics seeds + neural orthogonal enrichment
        |
Hard topology / positivity / metric constraints
        |
Matrix-free EM block actions -> small multiport solve
        |
Coupled mixed-dimensional thermal + RANS-SST equilibrium
        ^
        |  T -> sigma_sea, rho_Cu, mu, k, rho -> EM / flow
        |
Independent residual certifier
        |
Adaptive rank / physics fallback
        |
Z, losses, T, v, physical error indicators
```

---

## Non-negotiable modelling rules

1. **No solution labels in training.** No Maxwell, COMSOL, CFD, experimental, impedance, field, or temperature targets are used to train the neural basis generator.
2. **Multiphysics is retained.** Electromagnetic, thermal, and fluid feedbacks are solved as one coupled physical equilibrium; the architecture is not an EM-only surrogate.
3. **Fast/slow separation is physical, not decoupling.** Harmonic electromagnetic states are eliminated inside the slower thermo-fluid equilibrium while temperature-dependent material properties feed back into the EM and flow operators.
4. **No dense production EM operator.** The backend exposes block operator actions and direct reduced source projections; full `[N,N]` Green/resistance/inductance matrices are reference-only.
5. **Hard constraints where possible.** Solenoidal current/velocity spaces, basis rank, turbulence positivity and conductor-package heat exchange are enforced structurally rather than by penalty tuning.
6. **Physics-seeded neural enrichment.** Deterministic zero-label operator modes form the mandatory basis prefix; the network learns only the orthogonal remainder.
7. **Adaptive capacity is residual-driven.** Nested reduced bases are enlarged only when independent physical indicators require it.
8. **Full-order fallback is a safety path, not training-data generation.**
9. **BFZI is an external deterministic reference/backend option, not the definition of SDF-MPNEO.**

---

## Main implementation layers

- `topology`: gauge-reduced de Rham generators and harmonic spaces
- `networks`: shared geometry encoder plus coordinate-conditioned basis enrichment decoders
- `reduction`: physics-seed fusion, metric orthogonalisation and nested ranks
- `physics/em_matrix_free`: generic matrix-free electromagnetic projection and multiport reduced solve
- `physics/thermal`: mixed-dimensional 1D conductor + 3D package/seawater conjugate heat transfer
- `physics/flow`: incompressible RANS-SST reduced physics helpers
- `hyperreduction`: positive solution-free Basis-Operator Cubature
- `solvers`: pseudo-transient damped Newton equilibrium
- `training`: implicit differentiation through equilibrium without Newton unrolling
- `backends`: replaceable deterministic Python/C++/CUDA physics kernels
- `certification`: independent residual/goal indicators supplied by the backend

The current foundation branch focuses on fixing these interfaces, invariants, and the complete forward graph before the independent reference multiphysics backend is fully implemented.
