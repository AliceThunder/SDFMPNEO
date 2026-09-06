# Intrinsically parametric analytic neural evolution

The executable analytic network no longer needs to be restricted to one fixed initial state and one fixed excitation amplitude. For a fixed spatial discretization and fixed thermal spectrum, the network can represent

```text
(a0, U, t) -> a(t)
```

with `a0` and the declared operating parameters `U` appearing as nodes inside the analytic neural algebra.

## 1. Zero-dynamics parameter neurons

Every initial coordinate and operating parameter is represented as a static neural node,

```text
d p_j / dt = 0.
```

The initial-state contribution to thermal mode `i` is

```text
a0_i exp(-lambda_i t).
```

A generic network term therefore has the form

```text
c * p^gamma * t^m * exp(-(n dot lambda)t).
```

The parameter vector `p` contains the initial coordinates followed by the declared operating parameters.

This is not a conditioning MLP that predicts external coefficients. Static parameter nodes participate directly in neural products and response neurons.

## 2. Analytic closure

The parametric function algebra is closed under:

- addition;
- scalar multiplication;
- products of dynamic and/or static parameter nodes;
- exact time differentiation;
- exact parameter differentiation;
- the response operator
  ```text
  (d/dt + lambda_i) h = source, h(0)=0.
  ```

Consequently the complete network remains analytic in `(a0,U,t)`.

## 3. Exact operating sensitivities

`CompiledParametricAnalyticGraph.operating_jacobian(...)` differentiates the sparse analytic terms directly. The network does not use finite differences to obtain

```text
d a(t)/dU.
```

The parameter derivative commutes with the analytic time derivative, so the physical residual can also use exact

```text
d (da/dt) / dU.
```

## 4. Physical excitation map

The first executable operating-parameter physics interface is an exact affine electromagnetic RHS map,

```text
b(U) = b0 + B_U U.
```

This covers declared linear source/port-current amplitude coordinates. It does not claim that frequency, geometry, material parameters, or circuit constraints are RHS-only variables.

For this map the electromagnetic field is linear in the RHS at fixed thermal state. Therefore the explicit operating derivative of the reduced heat source is computed from directional field solves rather than finite differences.

The full residual sensitivity is

```text
dR/dU
= d(da/dt)/dU
  + Lambda da/dU
  - (dg/da) da/dU
  - (dg/dU)_explicit.
```

`ParametricElectroThermalResidual` evaluates this expression analytically.

## 5. Parameter-domain residual-grown topology

The parameter-domain grower admits products of:

- initial-state nodes;
- operating-condition nodes;
- existing analytic response nodes.

For a unit candidate response in target thermal mode `i`,

```text
(d/dt + lambda_i) h = psi,
```

the first residual variation remains

```text
D = e_i psi - J_g[:,i] h
```

at fixed `(a0,U,t)`. Across an externally supplied deterministic/certified parameter-time rule, the scalar tangent-optimal weight is

```text
w* = -<R,D>/<D,D>.
```

The complete nonlinear residual is recomputed before a candidate can be accepted.

The growth engine never invents parameter samples, time points, or quadrature weights. Certified adaptive parameter/time integration and global growth convergence remain outer-algorithm obligations.

## 6. Resonance-stable backend

The fast parametric compiler uses the sparse polynomial-exponential algebra. The same instantiated graph at a requested `(a0,U)` can independently compile to

```text
h(t) = c^T exp(A t) b.
```

Static parameter nodes become constant realizations, products use Kronecker sums, and response neurons add one state. Exact or nearly coalescing decay rates are handled by the matrix exponential without a near-resonance threshold.

The state-space realization is the stable correctness/reference backend for the fast canonical compiler.

## 7. Current online interface

`ParametricExecutableSDFMPNEOModel` directly queries

```text
evaluate(t, a0=..., operating=...)
```

and returns:

- `a(t)` and `da/dt`;
- reconstructed temperature;
- `da/dU` and `d(da/dt)/dU`;
- the actual electromagnetic RHS `b(U)`;
- reduced Joule heat source and its operating Jacobian;
- physical residual and its operating Jacobian;
- electromagnetic residual norm;
- optional multiport `Z/R/L/M` and output-error certificate;
- optional regional copper/seawater losses.

## 8. Scope boundary

The current parametric network holds the thermal spectrum and spatial electromagnetic operator family fixed and parameterizes the excitation through `b(U)`. The following remain separate extensions:

1. frequency-dependent operator parameters;
2. load/circuit constraints that change the coupled algebraic system;
3. nonlinear material parameters beyond the thermal state already represented by `a`;
4. geometry parameters that change topology/Hodge operators/thermal spectrum;
5. one certified parameter-domain growth/integration driver spanning all such variables.

These extensions must enter through explicit physical operator parameterizations and certificates rather than a hidden encoder.
