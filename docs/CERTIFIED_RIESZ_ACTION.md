# Certified inexact physical Riesz actions

## 1. Purpose

The nonlinear SDF-MPNEO electromagnetic reduction and reduced multiport output
certificate require the physical dual norm

```text
||r||_(H^-1),    H=K+D > 0,
```

but the mathematical method must not require an exact factorization of `H`.
The production interface is therefore a **certified Riesz action**, not an
`H^-1` routine.

For a requested vector `r`, an implementation may return any approximation

```text
y_tilde ~= H^-1 r
```

provided it also proves an energy-error bound

```text
||H^-1 r - y_tilde||_H <= delta.
```

Then the true dual norm is enclosed by

```text
max(0, ||y_tilde||_H-delta)
<= ||r||_(H^-1)
<= ||y_tilde||_H+delta.
```

Every reducer/output acceptance decision uses this enclosure.  Exact inversion
is therefore an implementation option, not an assumption of the theorem.

## 2. Certified preconditioned action

Let `P` be Hermitian positive definite and suppose a proved spectral-equivalence
constant `m>0` satisfies

```text
H >= m P.
```

The preconditioner applies `P^-1`.  If `y_tilde` has Riesz residual

```text
s = r - H y_tilde,
```

then

```text
||H^-1 r-y_tilde||_H^2
 = s^H H^-1 s
 <= (1/m) s^H P^-1 s.
```

Hence the executable certificate is

```text
delta = sqrt[(s^H P^-1 s)/m].
```

No empirical residual tolerance appears.

## 3. Physical stopping rules

`CertifiedPCGRieszAction` exposes two scientific operations.

### Requested action accuracy

```text
solve(r, requested_energy_action_error=epsilon_H)
```

stops only after it proves

```text
delta <= epsilon_H.
```

The requested `epsilon_H` must come from the propagated state/output error
requirement.  It is not a Krylov `rtol`.

### Dual-norm threshold decision

```text
decide_dual_norm(r, threshold=tau)
```

stops as soon as the certified interval lies entirely on one side of `tau`:

```text
upper <= tau  -> below,
lower >  tau  -> above.
```

If the available preconditioner cannot separate the interval, the result is
`indeterminate`; the theorem is never weakened to force a decision.

The sparse electromagnetic reducer uses

```text
tau = beta_H * requested_energy_state_error,
beta_H = 1/sqrt(2),
```

so the stopping threshold is derived directly from the physical state-error
target.

## 4. Reduction with an inexact Riesz action

At candidate thermal state `a`, the reduced residual is

```text
r(a) = b - A(a)V[V^H A(a)V]^-1 V^H b.
```

A candidate is certified if the Riesz action proves

```text
upper(||r(a)||_(H(a)^-1))
<= beta_H * epsilon_state.
```

If it is not certified, the approximate action vector `y_tilde` is used as the
next residual-driven enrichment direction.  After every enrichment the complete
full residual is re-evaluated.  Therefore an inexact enrichment can affect
convergence speed, but it cannot create a false certificate.

The global basis is still stored in the reference metric `H0=H(0)` by small-Gram
whitening.  `H0` is a coordinate metric only; all error decisions use local
`H(a)`.

## 5. Reduced multiport output certificate

For reduced port state `X_r` and column residual `r_j`,

```text
||e_j||_H
<= beta_H^-1 ||r_j||_(H^-1).
```

For port source `b_i`,

```text
|Delta Z_ij|
<= omega ||b_i||_(H^-1) ||e_j||_H.
```

With certified Riesz-action upper bounds,

```text
|Delta Z_ij|
<= omega
   * upper(||b_i||_(H^-1))
   * upper(||r_j||_(H^-1)) / beta_H.
```

Thus `Z/R/L/M` certification also requires no exact `H^-1` application.

## 6. Implemented backends

### Sparse-LU reference action

`SparseLUReferenceRieszAction` retains the deterministic sparse-LU correctness
backend used by the earlier implementation.  It now implements the same action
protocol and is no longer embedded in the reduction theorem.

### Certified PCG action

`CertifiedPCGRieszAction` accepts any preconditioner that proves `H>=mP`.
The currently executable parameter-free preconditioner is
`DiagonalGershgorinEnergyPreconditioner`:

```text
P = diag(H).
```

For

```text
B = P^-1/2 H P^-1/2,
```

Gershgorin gives the proved lower bound

```text
m <= lambda_min(B).
```

If the lower bound is non-positive the constructor refuses the preconditioner;
it does not add a fitted diagonal shift.

This diagonal backend is a theorem/regression implementation, not the final
large-mesh preconditioner.  The next production step is an auxiliary-space or
multilevel `P^-1` with its own proved positive `m`.

## 7. Floating-point contract

Energy inner products and preconditioned residual products are enclosed using
standard `gamma_k` floating-point bounds scaled by the actual product magnitude.
There is no absolute `max(1,...)` floor for late Krylov vectors.  This prevents a
legitimate small positive search-direction energy from being misclassified as a
breakdown merely because its magnitude is below one.

## 8. Regression obligations

The repository tests enforce that:

1. the PCG action upper-bounds the observed energy error on independent direct
   verification systems;
2. threshold decisions are returned only after the certified interval lies on
   the declared side of the threshold;
3. an unprovable Gershgorin spectral-equivalence constant is rejected rather
   than repaired heuristically;
4. `SparseEnergyResidualGreedyEMReducer` builds and certifies with
   `scipy.sparse.linalg.splu` disabled when a certified PCG factory is supplied;
5. reduced multiport `Z/R/L/M` output certification also works with sparse LU
   disabled.

These tests separate the scientific certificate from any particular linear
solver implementation.
