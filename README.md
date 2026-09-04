# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Equilibrium Operator for Underwater WPT**

SDF-MPNEO is a physics-embedded neural reduced-order framework for strongly coupled electromagnetic-thermal-fluid modelling of underwater wireless power transfer systems.

The neural network does **not** regress impedance, temperature, or flow solutions from labelled FEM/CFD data. It generates geometry-conditioned admissible trial-space enrichments around deterministic solution-free physics seeds; the physical states are then obtained by solving reduced governing equations.

## Core state

- seawater induced current density `J_s`
- unified conductor/package/seawater temperature state `Theta`
- seawater velocity `v`
- SST turbulence closure states `k_t, omega_t`

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
Z, losses, T, v, physical error indicators
```

## Non-negotiable modelling rules

1. **No solution labels in training.** No Maxwell, COMSOL, CFD, experimental, impedance, field, or temperature targets are used to train the neural basis generator.
2. **Multiphysics is retained.** Electromagnetic, thermal, and fluid feedbacks are solved as one coupled physical equilibrium; the architecture is not an EM-only surrogate.
3. **Fast/slow separation is physical, not decoupling.** Harmonic electromagnetic states are eliminated inside the slower thermo-fluid equilibrium while temperature-dependent material properties feed back into the EM and flow operators.
4. **No dense production EM operator.** The backend exposes block operator actions and direct reduced source projections; full `[N,N]` Green/resistance/inductance matrices are reference-only.
5. **Hard constraints where possible.** Solenoidal current/velocity spaces, basis rank, turbulence positivity and conductor-package heat exchange are enforced structurally rather than by penalty tuning.
6. **Physics-seeded neural enrichment.** Deterministic zero-label operator modes form the mandatory basis prefix; the network learns only the orthogonal remainder.
7. **Adaptive capacity is residual-driven.** Nested reduced bases are enlarged only when independent physical indicators require it.
8. **Full-order fallback is a safety path, not training-data generation.**

## Implementation layers

- `topology`: gauge-reduced de Rham generators and harmonic spaces
- `networks`: shared geometry encoder plus coordinate-conditioned basis enrichment decoders
- `reduction`: physics-seed fusion, metric orthogonalisation and nested ranks
- `physics/em`: generic matrix-free electromagnetic projection and multiport reduced solve
- `physics/thermal`: mixed-dimensional 1D conductor + 3D package/seawater conjugate heat transfer
- `physics/flow`: incompressible RANS-SST reduced physics helpers
- `hyperreduction`: positive solution-free Basis-Operator Cubature
- `solvers`: pseudo-transient damped Newton equilibrium
- `training`: implicit differentiation through equilibrium without Newton unrolling
- `backends`: replaceable deterministic Python/C++/CUDA physics kernels
- `certification`: independent residual/goal indicators supplied by the backend

A BFZI/DSE implementation may be used as one deterministic electromagnetic backend/reference, but SDF-MPNEO is defined by the generic operator-action contract rather than by any BFZI-specific implementation.
