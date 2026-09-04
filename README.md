# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Equilibrium Operator for Underwater WPT**

SDF-MPNEO is a physics-embedded neural reduced-order framework for strongly coupled electromagnetic–thermal–fluid modelling of underwater wireless power transfer systems.

The neural network does **not** regress impedance, temperature, or flow solutions from labelled FEM/CFD data. It generates geometry-conditioned admissible reduced trial spaces; the physical states are then obtained by solving reduced governing equations.

## Core state

- seawater induced current density `J_s`
- conductor/package/seawater temperature `T_c, T`
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
|       |          +--> divergence-free velocity basis
|       |          +--> positive SST closure bases
|       +-------------> mixed-dimensional thermal basis
+---------------------> divergence-free seawater-current basis
        |
Hard physical structure / metric orthogonalisation
        |
Harmonic EM Schur elimination
        |
Coupled thermo-fluid reduced equilibrium
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
3. **Fast/slow separation is physical, not decoupling.** Harmonic electromagnetic states are Schur-eliminated inside the slower thermo-fluid equilibrium while temperature-dependent conductivity and resistivity feed back into the electromagnetic operator.
4. **Hard constraints where possible.** Solenoidal current/velocity spaces, passivity, reciprocity, positivity of turbulence states, and conductor-package heat exchange are enforced by construction rather than penalty tuning.
5. **Adaptive capacity is residual-driven.** Nested reduced bases are enlarged only when independent physical residual indicators require it.
6. **Full-order fallback is a safety path, not training data generation.**

## Planned implementation layers

- `geometry`: unified coil/package parameterisation and reference-domain mappings
- `topology`: discrete de Rham incidence operators and harmonic spaces
- `networks`: shared geometry encoder plus EM/thermal/flow basis decoders
- `physics/em`: BFZI/DSE-backed matrix-free electromagnetic operators and Schur impedance
- `physics/thermal`: mixed-dimensional 1D conductor + 3D package/seawater conjugate heat transfer
- `physics/flow`: incompressible RANS-SST reduced operator
- `solvers`: pseudo-transient damped Newton and implicit differentiation
- `reduction`: metric orthogonalisation, nested ranks, basis-operator cubature
- `certification`: independent residual and goal-oriented indicators
- `backends`: replaceable Python/C++ kernels

The initial repository version focuses on fixing these interfaces and the complete forward graph before high-performance kernels are introduced.
