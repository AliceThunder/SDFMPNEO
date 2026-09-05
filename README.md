# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator for Underwater WPT**

SDF-MPNEO is a solution-data-free electromagnetic–thermal evolution framework for underwater wireless power transfer. The current baseline intentionally excludes fluid dynamics while retaining conductor, package, seawater, and other spatial media in the electromagnetic and thermal physics.

The online objective is

```text
geometry + constant operating condition + initial thermal state + arbitrary query time
                                      |
                                      v
                        analytic neural evolution
                                      |
                                      v
                             thermal state T(x,t)
                                      |
                                      v
                 deterministic temperature-dependent EM operator
                                      |
                                      v
                Z, R, L, M, Cu loss, seawater loss, certificates
```

## Core design

1. **Deterministic spatial physics.** Thermal coordinates are obtained from the physical thermal operator, not from solution snapshots and not from a neural basis generator.
2. **Analytic neural dynamics.** A neuron is an exactly solvable local dissipative response operator. The network output is a closed-form polynomial–exponential time function and requires no temporal marching at inference.
3. **Deterministic electromagnetic feedback.** Temperature-dependent copper resistivity and seawater conductivity are evaluated by the physical electromagnetic backend; the neural network does not replace the electromagnetic operator.
4. **Solution-data-free training.** Training minimizes the true reduced electromagnetic–thermal residual. No Maxwell/FEM/experimental solution trajectory is used as a label.
5. **Residual-grown topology.** Analytic neurons are added according to unresolved physical residual directions rather than a prescribed empirical width/depth.
6. **Unified certification.** Thermal-reduction, neural-residual, and electromagnetic numerical errors are propagated to temperature and impedance output bounds.
7. **No empirical model partitioning.** Reduced rank, network complexity, and electromagnetic numerical accuracy are selected from physical error requirements and certified convergence conditions.

## Reduced governing dynamics

After deterministic thermal spectral reduction,

```text
    da/dt + Lambda(mu) a = g_em(a; mu)
```

where `Lambda` contains physical thermal decay rates and `g_em` is supplied by the deterministic temperature-dependent electromagnetic operator.

The analytic network directly evaluates

```text
    a(t) = N_analytic(mu, a0, t)
```

and is trained from the residual

```text
    R = da/dt + Lambda a - g_em(a; mu).
```

For a certified contractive operating domain, the remaining physical residual is converted into a uniform-in-time state-error bound and then into error bounds for `Tmax`, impedance, losses, and related outputs.

## Theory

The complete mathematical specification, derivations, assumptions, theorem statements, proof outlines, adaptive growth rule, and unified error-certification framework are in:

- [`docs/SDFMPNEO_theory.tex`](docs/SDFMPNEO_theory.tex)

## Current implementation layers

- `geometry`: unified coil/package/seawater parameterisation and reference-domain mappings
- `physics/em`: BFZI/DSE-backed temperature-dependent deterministic electromagnetic operator
- `physics/thermal`: conductor/package/seawater thermal operator and certified spectral reduction
- `networks/analytic`: analytic response neurons, multiplicative interaction graph, exact time derivatives
- `training`: physical-residual minimisation and residual-driven neuron enrichment
- `certification`: independent higher-space residual, electromagnetic numerical error, and output bounds
- `backends`: replaceable Python/C++ kernels

## Non-negotiable scope rules

- No labelled FEM/Maxwell/experimental solution data in training.
- No empirical near/far split, fixed reduced rank, fixed network width/depth, or artificial thermal time constant as part of the method definition.
- Fluid velocity is outside the current model scope; seawater remains an explicit electromagnetic and thermal medium.
- Full-order transient simulation and experiments are validation tools only, not training-label generators.
