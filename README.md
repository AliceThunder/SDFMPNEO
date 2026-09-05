# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator for Underwater WPT**

SDF-MPNEO is a solution-data-free electromagnetic–thermal surrogate framework for underwater wireless power transfer. The current baseline retains conductor, package, seawater, and other passive media in the spatial electromagnetic and thermal physics, while fluid velocity is outside the present scope.

The method no longer depends on any geometry-specific impedance solver. The electromagnetic branch is derived directly from the full magnetoquasistatic field equations, reduced deterministically, and then quasi-statically eliminated inside the slower thermal dynamics.

## Unified online map

```text
geometry + operating condition + initial thermal state + arbitrary query time
                                      |
                                      v
                    certified reference-domain operators
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
          reduced electromagnetic physics       reduced thermal physics
                    |                                   |
                    +----------> quasi-static <---------+
                                  EM elimination
                                      |
                                      v
                         reduced heat-source map
                                      |
                                      v
                         analytic neural evolution
                                      |
                                      v
                           a(t), T(x,t), Z, losses
                                      |
                                      v
                               error certificate
```

## Core design

1. **Full-equation electromagnetic origin.** The electromagnetic model starts from the magnetoquasistatic field equations in conductor, package, seawater, and insulating regions. Copper skin/proximity effects and seawater induced-current losses arise from the field solution itself rather than from add-on resistance formulas.
2. **Deterministic electromagnetic reduction without solution snapshots.** The electromagnetic reduced space is generated from the physical operator, excitation range, and worst-case residual Riesz lifts. No Maxwell/FEM solution trajectory is used as a training label or basis snapshot.
3. **Uniform temperature/geometry validity.** Geometry parameters and reduced thermal coordinates are included in one certified electromagnetic parameter domain. The reduced space is enriched until the supremum electromagnetic residual over that domain satisfies the required output-error bound.
4. **Certified operator separation.** Geometry coefficients and temperature-dependent constitutive laws are represented by exact algebraic separation when available, otherwise by deterministic polynomial/Chebyshev expansions with verified remainder bounds. This makes reduced electromagnetic assembly fast without empirical interpolation.
5. **Direct reduced heat-source projection.** Copper and seawater Joule losses are projected to the thermal reduced coordinates through preassembled quadratic coupling tensors. Full three-dimensional loss fields are not reconstructed during online evaluation.
6. **Analytic neural dynamics.** A neuron is an exactly solvable local dissipative response operator. The network output is a closed-form polynomial–exponential time function and requires no temporal marching at inference.
7. **Solution-data-free training.** The analytic network is trained from the true reduced electromagnetic–thermal residual, not from labelled transient solutions.
8. **Residual-grown topology.** Analytic neurons are added according to unresolved physical residual directions rather than a prescribed empirical width/depth.
9. **Unified certification.** Electromagnetic reduction, operator separation, thermal reduction, analytic-network residual, and numerical linear-solve errors propagate to bounds for temperature, impedance, mutual inductance, and losses.
10. **No empirical model partitioning.** Reduced ranks, network complexity, constitutive expansion order, and numerical tolerances are selected from rigorous output-error requirements and convergence conditions.

## Reduced coupled system

After deterministic spatial reduction, the electromagnetic state satisfies

```text
A_em,r(a; mu) c = b_em,r(mu)
```

where `a` is the reduced thermal state. Since the electromagnetic time scale is much faster than the thermal time scale,

```text
c(a; mu) = A_em,r(a; mu)^(-1) b_em,r(mu)
```

is quasi-statically eliminated. Copper and seawater losses are then projected directly to the thermal reduced coordinates,

```text
q_em,r(a; mu) = q_Cu,r(c,a;mu) + q_sea,r(c,a;mu)
```

and the closed thermal dynamics become

```text
da/dt + Lambda_T(mu) a = g_em(a; mu).
```

The analytic network directly evaluates

```text
a(t) = N_analytic(mu, a0, t)
```

and is trained from

```text
R = da/dt + Lambda_T a - g_em(a; mu).
```

No time stepping is required during inference.

## Electromagnetic reduced-space construction

The electromagnetic basis is grown without full-solution snapshots:

```text
initial source/constraint-compatible space
              |
              v
solve reduced EM system over certified parameter domain
              |
              v
compute full operator residual r = b - A V c
              |
              v
certified worst-case parameter eta*
              |
              v
Riesz lift w = H^(-1) r(eta*)
              |
              v
H-orthogonalize and enrich V
              |
              v
stop only when the EM output certificate is satisfied
```

This same mechanism handles geometry variation and temperature-dependent copper/seawater conductivity because both enter the common parameter `eta = (geometry, operating condition, thermal coordinate)`.

## Theory and implementation specification

- [`docs/SDFMPNEO_theory.tex`](docs/SDFMPNEO_theory.tex): full governing equations, deterministic electromagnetic/thermal reduction, analytic-neuron theory, residual enrichment, stability, and unified certification.
- [`docs/SDFMPNEO_implementation.md`](docs/SDFMPNEO_implementation.md): code architecture, data structures, offline compiler, online evaluator, and verification sequence.

## Planned implementation layers

- `geometry`: unified reference-domain mapping for conductor/package/seawater geometry
- `physics/em`: compatible magnetoquasistatic full operator, constitutive laws, deterministic reduced basis, reduced operator, quasi-static elimination, loss tensors
- `physics/thermal`: conductor/package/seawater thermal operator and certified spectral reduction
- `physics/coupling`: reduced electromagnetic-to-thermal heat-source closure
- `networks/analytic`: analytic response neurons, multiplicative interaction graph, exact time derivatives
- `training`: physical-residual minimisation and residual-driven neuron enrichment
- `certification`: electromagnetic residual/inf-sup bound, thermal residual, analytic-network residual, constitutive-expansion remainder, and output bounds
- `compiler`: offline symbolic/DAG compression and C++ export

## Non-negotiable scope rules

- No labelled FEM/Maxwell/experimental solution data in training.
- No full-order solution snapshots are required to construct the reduced electromagnetic or thermal spaces.
- No geometry-specific impedance solver is a core dependency.
- No empirical near/far split, fixed reduced rank, fixed network width/depth, artificial thermal time constant, or empirical constitutive correction is part of the method definition.
- Copper loss and seawater loss originate from the electromagnetic field equations and their temperature-dependent constitutive laws.
- Fluid velocity is outside the current model scope; seawater remains an explicit electromagnetic and thermal medium.
- Full-order simulations and experiments are validation tools only, not training-label generators.
