# Exact nonlinear temperature-dependent conductivity

The electromagnetic reduced interface no longer requires conductivity to be affine in the reduced thermal coordinates.

## Generic operator interface

`ReducedEMModel` accepts any electromagnetic problem that provides

```text
operator(a)
operator_derivatives(a)
loss_operator(j,a)
loss_operator_derivative(j,k,a)
b
H_metric
n_thermal
n_em
```

`ParametricEMProblem` remains the affine verification backend, while `NonlinearSpatialAphiProblem` is the direct nonlinear spatial correctness backend. The legacy class name is retained for API compatibility; its internal field coordinate is the reciprocal scaled `A-psi` formulation.

## Direct nonlinear spatial problem

The nonlinear problem reconstructs

```text
T(a) = T_ref + sum_k a_k Phi_k
```

on material cells, evaluates each analytic material law, assembles `M_sigma(T(a))`, and rebuilds the reciprocal compatible field operator. The scalar coordinate is

```text
psi = phi/(j omega),
E = -j omega (R_A alpha + G_c psi).
```

The material derivative is analytic,

```text
d sigma / d a_k = (d sigma / d T) Phi_k,
```

and is propagated through Hodge assembly to `dA/da_k` and the projected heat-source derivative.

## Copper law supported exactly

For

```text
rho(T) = rho_ref [1 + alpha (T-T_ref)],
```

the implementation uses

```text
sigma(T) = sigma_ref / [1 + alpha (T-T_ref)]
```

with

```text
dsigma/dT = -sigma_ref alpha / [1 + alpha (T-T_ref)]^2.
```

No affine conductivity approximation is introduced. States with non-positive resistivity denominator are rejected as physically inadmissible.

## Other material laws

The executable material library includes:

- `ConstantConductivity`;
- `AffineConductivity` when the constitutive law itself is physically specified as affine over its admissible range;
- `ReciprocalLinearResistivity`;
- `CompositeCellConductivity`, which assigns different analytic laws to disjoint conductor/seawater/package cell masks.

Additional laws implement the same `evaluate(T)` and `derivative(T)` contract.

## Exact heat-source Jacobian

For reduced electromagnetic coordinate state `x(a)`,

```text
q_j(a) = x^H H_j(a) x.
```

The derivative is

```text
dq_j/da_k
= 2 Re[(dx/da_k)^H H_j x]
  + x^H (dH_j/da_k) x,
```

with reduced field sensitivity

```text
dc/da_k = -A_r(a)^(-1) [dA_r/da_k] c.
```

Regression tests compare `dA/da_k` and the final `dq/da_k` against stable centred finite differences. The finite-difference step is only a regression cross-check and is not a model or training parameter.

## Excitation dependence

The reduced electromagnetic model now supports arbitrary RHS vectors. Thus the heat source can be evaluated as

```text
q_em,r(a; b)
```

for a declared port-current excitation rather than being permanently tied to one nominal source `problem.b`. Joint multi-RHS reduction ensures that the reduced field space spans all declared excitations before the same space is used for impedance or heat-source queries.

## Certification status

The nonlinear constitutive path solves the architectural temperature-feedback problem without empirical linearization. Current rigorous domain certification is stronger for the affine verification backend: continuous thermal-state boxes can be proved by branch-and-bound there. Extending this to nonlinear material laws requires verified interval/remainder bounds for `sigma(T)`, `d sigma/dT`, and the resulting field stability constants over the full admissible domain.

For production use each material law still requires:

1. a physically justified relation and admissible temperature range;
2. verified positivity/admissibility over the certified domain;
3. uncertainty/error propagation for material constants;
4. continuous-domain operator and output bounds;
5. a certified separated/compressed online representation if direct cell-Hodge rebuilding dominates large-mesh cost.

The direct nonlinear path remains the correctness reference; future compression must reproduce it within a verified remainder bound.
