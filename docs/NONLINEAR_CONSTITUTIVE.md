# Exact nonlinear temperature-dependent conductivity

The electromagnetic reduced interface no longer requires conductivity to be affine in the reduced thermal coordinates.

## Generic operator interface

`ReducedEMModel` now requires an electromagnetic problem to provide

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

The original `ParametricEMProblem` still implements this interface for affine verification problems, but it is no longer the only admissible backend.

## Direct nonlinear spatial problem

`NonlinearSpatialAphiProblem` reconstructs the cell temperature field

```text
T(a) = T_ref + sum_k a_k Phi_k
```

and evaluates the cell conductivity directly from analytic material laws. It then rebuilds the conductivity Hodge matrix and compatible `A-phi` operator at that state.

The derivative is also analytic,

```text
d sigma / d a_k = (d sigma / d T) Phi_k,
```

which is propagated through the Hodge assembly into `dA/da_k` and the projected loss derivative.

## Copper law supported exactly

For a linear resistivity law

```text
rho(T) = rho_ref [1 + alpha (T-T_ref)],
```

the code uses the exact reciprocal conductivity

```text
sigma(T) = sigma_ref / [1 + alpha (T-T_ref)]
```

with exact derivative

```text
dsigma/dT = -sigma_ref alpha / [1 + alpha (T-T_ref)]^2.
```

This is not replaced by an affine conductivity approximation. The implementation rejects states for which the resistivity denominator is non-positive.

## Other material laws

The executable core also includes:

- `ConstantConductivity`;
- `AffineConductivity` for a material law that is itself physically specified as affine over the certified range;
- `CompositeCellConductivity`, which combines disjoint conductor/seawater/package masks with different analytic laws.

Additional material laws can implement the same `evaluate(T)` and `derivative(T)` interface.

## Heat-source Jacobian

For reduced electromagnetic state `x(a)`,

```text
q_j(a) = x^H H_j(a) x.
```

The code evaluates

```text
dq_j/da_k = 2 Re[(dx/da_k)^H H_j x] + x^H (dH_j/da_k) x,
```

where

```text
dx/da_k = -A_r(a)^(-1) [dA_r/da_k] x_r.
```

The nonlinear regression tests compare both `dA/da_k` and `dq/da_k` against centred finite differences.

## What remains

This solves the architectural problem of nonlinear temperature feedback: no empirical linearization is required. It does **not** by itself certify a chosen empirical seawater conductivity law. For production use, each material law still requires:

1. a physically justified constitutive relation and admissible temperature range;
2. parameter-domain verification that conductivity remains physically admissible;
3. output-error propagation of any uncertainty in material constants;
4. a fast separated/compressed online representation if rebuilding cell Hodge matrices becomes the dominant cost on large meshes.

The direct nonlinear path is therefore the correctness reference implementation; later operator compression must reproduce it within a certified remainder bound.
