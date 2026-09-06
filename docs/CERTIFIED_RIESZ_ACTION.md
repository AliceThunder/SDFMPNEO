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

Every reducer/output acceptance decision uses this enclosure. Exact inversion is
therefore an implementation option, not an assumption of the theorem.

## 2. Certified preconditioned action

Let `P` be Hermitian positive definite and suppose a proved spectral-equivalence
constant `m>0` satisfies

```text
H >= m P.
```

The preconditioner applies `P^-1`. If `y_tilde` has Riesz residual

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
requirement. It is not a Krylov `rtol`.

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

## 4. Physical magnetic/scalar block theorem

For the gauge-reduced tetrahedral `A-psi` energy metric, partition the state into
magnetic vector-potential coordinates and conducting scalar-potential
coordinates:

```text
H = [[K_A + D_AA, D_Apsi],
     [D_psiA,       D_psipsi]].
```

The magnetic gauge gives `K_A>0`; the conductivity Gram operator gives `D>=0`.
If a certified constant `gamma` satisfies

```text
D_AA <= gamma K_A,
```

then conductive Cauchy-Schwarz and the optimal Young inequality give

```text
H >= m_phys(gamma) diag(K_A,D_psipsi),
```

with

```text
m_phys(gamma)
= 2 / [2 + gamma + sqrt(gamma^2 + 4 gamma)].
```

No block weight, damping factor, or conductivity-ratio fit is selected by the
user.

### Replaceable block actions

The magnetic and scalar blocks do not need exact inverses. Suppose their actions
apply `P_K^-1` and `P_E^-1` and independently prove

```text
K_A      >= m_K P_K,
D_psipsi >= m_E P_E.
```

Then for

```text
P = diag(P_K,P_E)
```

the outer Riesz preconditioner has the rigorous lower bound

```text
H >= m_phys(gamma) min(m_K,m_E) P.
```

This is the key scalability interface: a future auxiliary-space magnetic action
and a future multilevel scalar action can replace the current block solvers
without changing the outer PCG certificate, reducer, or output theorem.

The default `SparseLUExactBlockPreconditioner` corresponds to

```text
P_K=K_A,  P_E=D_psipsi,  m_K=m_E=1.
```

It is a correctness backend only.

## 5. Certified gamma construction

The implementation first attempts a normalized Gershgorin certificate for
`K_A` and `D_AA`. When it proves a positive magnetic lower bound, `gamma` follows
without a factorization.

Real Nedelec magnetic blocks need not be diagonally dominant. In that case the
current correctness fallback uses a complete sparse factorization only to build
a certificate, not to apply the global `H^-1` action. It explicitly recomputes
factorization residuals and certifies an upper bound on `||K_A^-1||_inf`, then
uses

```text
gamma
<= lambda_max(K_A^-1/2 D_AA K_A^-1/2)
<= trace(K_A^-1 D_AA).
```

Each trace contribution is inflated by the certified magnetic-solve residual.
If the inverse/trace enclosure cannot be proved finite, construction fails.
There is no diagonal shift or relaxed gamma target.

A scalable factorization-free gamma certificate remains a production obligation.

## 6. Reduction with an inexact Riesz action

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
next residual-driven enrichment direction. After every enrichment the complete
full residual is re-evaluated. Therefore an inexact enrichment can affect
convergence speed, but it cannot create a false certificate.

The global basis is stored in the reference metric `H0=H(0)` by small-Gram
whitening. `H0` is a coordinate metric only; all error decisions use local
`H(a)`.

For a declared WPT `port_set`, all unit-port right-hand sides participate in one
joint multi-RHS greedy construction. Certification therefore covers every
port/state pair in the declared finite candidate set.

## 7. Reduced multiport output certificate

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

Thus `Z/R/L/M` certification also requires no exact global `H^-1` application.
The formal tetrahedral core stores the same physical-block Riesz factory in the
reduced model, so online multiport certification automatically uses the same
backend rather than silently reverting to the sparse-LU reference action.

## 8. Implemented backends

### Global sparse-LU reference action

`SparseLUReferenceRieszAction` remains only as a generic correctness/reference
implementation of the Riesz-action protocol. The formal nonlinear tetrahedral
core no longer selects it as its production path.

### Diagonal Gershgorin action

`DiagonalGershgorinEnergyPreconditioner` uses

```text
P = diag(H)
```

and proves a normalized Gershgorin lower bound. If that bound is non-positive,
the constructor refuses the action instead of adding a fitted shift.

### Physical block PCG action

The formal nonlinear tetrahedral core now uses

```text
make_physical_block_pcg_riesz_factory(problem)
```

which combines the physical block theorem, local `gamma` certificate, and
`CertifiedPCGRieszAction`.

The block implementation is itself injected through `BlockActionFactory`. Tests
show that both blocks can be replaced by certified non-LU actions while
`scipy.sparse.linalg.splu` is disabled when the corresponding proofs permit it.

## 9. Floating-point contract

Energy inner products, preconditioned residual products, Gershgorin radii, and
the residual-certified trace fallback are enclosed with outward floating-point
bounds scaled by the actual operation magnitude. There is no absolute
`max(1,...)` floor for late Krylov vectors.

## 10. Regression obligations

The repository tests enforce that:

1. the PCG action upper-bounds the observed energy error on independent direct
   verification systems;
2. threshold decisions are returned only after the certified interval lies on
   the declared side of the threshold;
3. an unprovable diagonal Gershgorin certificate is rejected rather than shifted;
4. the physical block theorem succeeds on coupled systems where global diagonal
   Gershgorin fails;
5. the residual-certified generalized-trace fallback handles a real
   non-diagonally-dominant magnetic block;
6. block actions can run with sparse LU disabled when their own certificate does
   not require LU;
7. the high-contrast tetrahedral reducer uses the physical-block Riesz factory;
8. the top-level nonlinear core cannot fall back to the global sparse-LU Riesz
   reference backend;
9. `build_ports -> joint multi-RHS ROM -> certified reduced Z/R/L/M` executes as
   one top-level chain.

These tests separate the scientific certificate from any particular linear
solver implementation.
