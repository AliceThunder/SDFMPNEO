# Compiled Analytic Deployment Plan

## Goal

After training, convert the fixed SDF-MPNEO response DAG into an equivalent static analytic feed-forward representation for faster deployment inference without retraining and without changing the learned function.

The target representation is

\[
a_j(p,t)=\sum_k c_{jk}\,p^{\gamma_k}\,t^{m_k}e^{-\rho_k t},\qquad p=(a_0,G,U).
\]

This is already the canonical algebra represented by `ParametricAnalyticSeries`.

## Requirements

1. Exact mathematical equivalence to the trained response DAG up to floating-point roundoff.
2. Preserve exact initial-condition structure, arbitrary-time evaluation, long-time stability semantics, and the exact stationary limit where defined.
3. No additional supervised training or distillation is required.
4. Preserve a reference DAG path for equivalence checks.
5. Do not replace the analytic model by an approximate ReLU/tanh MLP in the certified path.

## Planned compiler

1. Freeze the trained DAG topology and weights.
2. Compile each thermal output into sparse canonical terms keyed by `(parameter_exponents, time_power, decay_signature)`.
3. Merge identical terms exactly.
4. Perform common-subexpression elimination for repeated parameter monomials, decay factors, and polynomial-time factors.
5. Lower the resulting graph to batched tensor operations: powers/products, exponentials, elementwise products, and linear reductions.
6. Provide CPU vectorized and optional PyTorch/JAX/GPU backends.
7. Add ONNX/C export only where the required exact primitives and stationary-limit semantics can be preserved.

## Avoiding expression explosion

A complete symbolic expansion can increase the number of terms. The deployment compiler should therefore support a hybrid representation:

- canonical sparse series where expansion is compact;
- shared subexpressions where multiple terms use the same monomial or exponential factor;
- retained compact response blocks when expansion would increase runtime or memory.

The compiler should choose the cheapest exact representation, not blindly fully expand every node.

## Verification

For every compiled backend, compare against the original trained DAG over:

- random in-domain geometry/current/initial conditions;
- `t=0`;
- early transient times;
- intermediate and very long finite times;
- resonant/repeated-decay cases;
- `t=+inf` where supported.

Verify modal values, time derivatives, reconstructed temperatures, and stationary limits to tight floating-point tolerances.

## Status

Planned only. No deployment compiler is implemented by this document. Training and prediction semantics remain unchanged until a future implementation is reviewed and merged.
