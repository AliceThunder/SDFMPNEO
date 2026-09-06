# Sparse physical-energy snapshot-free electromagnetic reduction

This document specifies the production electromagnetic reduction path used by the certified nonlinear tetrahedral SDF-MPNEO core.

## 1. Physical full-order problem

At a reduced thermal state `a`, the gauge-reduced reciprocal magnetoquasistatic system is

```text
A(a) x(a) = b,
A(a) = K + i D(a),
K >= 0,
D(a) >= 0.
```

Define the local physical energy metric

```text
H(a) = K + D(a).
```

For every complex field vector `v`,

```text
|v^H A(a) v|
= sqrt[(v^H K v)^2 + (v^H D(a) v)^2]
>= v^H H(a) v / sqrt(2).
```

Hence the coercivity constant in the `H(a)` norm satisfies

```text
beta_H >= 1/sqrt(2)
```

independently of copper/seawater conductivity contrast.

For any approximate state `x_r`, with residual

```text
r(a) = b - A(a) x_r,
```

the deterministic state certificate is

```text
||x(a)-x_r(a)||_H(a)
<= sqrt(2) ||r(a)||_(H(a)^-1).
```

This relation is the reduction error estimator. It does not use a solution snapshot or an external singular-value fit.

## 2. Reduced equilibrium

Let `V` be the electromagnetic reduced basis. The reduced state is obtained from

```text
A_r(a) c(a) = b_r,
A_r(a) = V^H A(a) V,
b_r = V^H b,
x_r(a) = V c(a).
```

The full electromagnetic matrix remains sparse. Only the intentionally small projected matrix `A_r` is dense.

The reduced residual is evaluated from the sparse full-order operator,

```text
r(a) = b - A(a) V c(a).
```

so the reduced-space error remains independently checkable after projection.

## 3. Snapshot-free residual-Riesz enrichment

For every candidate thermal state and every physical excitation, evaluate the certified upper bound of

```text
eta(a,b) = sqrt(2) ||r(a,b)||_(H(a)^-1).
```

The Riesz operation is not assumed exact. A `CertifiedRieszAction` returns an approximate lift `y_tilde`, an energy-action error bound `delta`, and a certified interval for the true dual norm. A candidate is accepted only if

```text
upper(||r||_(H^-1))
<= beta_H * requested_energy_state_error.
```

If the candidate is unresolved, the approximate certified lift is used for enrichment:

```text
y_tilde ~= H(a)^-1 r(a,b),
span(V_new) = span(V, y_tilde).
```

After every enrichment the complete residual is recomputed. Therefore an inexact Riesz lift may affect convergence speed but cannot create a false certificate.

No full-order equilibrium solution `A(a)^-1 b` is inserted into the basis. The basis is grown from unresolved governing-equation residuals rather than solution snapshots.

The finite-candidate construction terminates when the certified worst-case state error over all supplied state/excitation pairs is below the requested error. If the lift becomes numerically dependent or the full discrete space is exhausted before that condition is proved, the reducer returns an explicit uncertified/stalled result. The target is never relaxed.

## 4. Why the local energy metric is not frozen

The conductive energy changes with temperature because `D=D(a)`. Therefore the error theorem always uses the local metric

```text
H(a)=K+D(a).
```

Using a fixed reference metric to certify every thermal state would discard the contrast-independent physical theorem and could distort the residual bound.

A fixed reference metric is used only to store one stable global coordinate system for the basis:

```text
H0 = H(a=0).
```

This does not alter the enriched span or the local error estimator. It only chooses coordinates inside the already selected reduced subspace.

## 5. Reference-energy basis coordinates

After enrichment, the basis is stored such that

```text
V^H H0 V = I
```

up to floating-point backward error.

Rather than relying on repeated Gram-Schmidt sweeps, the implementation forms only the small Gram matrix

```text
G = W^H H0 W
```

for the current candidate basis `W`, computes

```text
G = L L^H,
```

and applies the low-dimensional whitening

```text
V = W L^(-H).
```

No dense full-order Cholesky factor of `H0` is formed. Only the reduced Gram matrix is factorized densely.

For high-contrast energy metrics, orthogonality regression uses a conditioning-aware floating-point error envelope based on

```text
|V|^T |H0| |V|
```

and the standard `gamma_k` arithmetic model; a fixed multiple of machine epsilon is not a valid bound when large contributions cancel.

## 6. Physical-block certified Riesz action

For the tetrahedral `A-psi` coordinates, partition the local energy metric as

```text
H = [[K_A + D_AA, D_Apsi],
     [D_psiA,       D_psipsi]].
```

The tree-cotree gauge makes `K_A` positive definite, while the conductivity Gram operator gives `D>=0`. If

```text
D_AA <= gamma K_A,
```

then

```text
H >= m_phys(gamma) diag(K_A,D_psipsi),
```

where

```text
m_phys(gamma)
= 2/[2+gamma+sqrt(gamma^2+4 gamma)].
```

The magnetic and scalar blocks are themselves replaceable certified actions. If

```text
K_A      >= m_K P_K,
D_psipsi >= m_E P_E,
```

then the actual preconditioner `P=diag(P_K,P_E)` satisfies

```text
H >= m_phys(gamma) min(m_K,m_E) P.
```

This bound is consumed by `CertifiedPCGRieszAction`. No user-selected Krylov tolerance, fitted damping, or conductivity-ratio correction enters the method.

The current default block implementations use complete sparse LU strictly as correctness-scale actions. Regression tests also replace both blocks with certified non-LU actions while `splu` is disabled, demonstrating that block LU is not part of the outer theorem.

For `gamma`, the fast path uses normalized Gershgorin. If the magnetic block is not diagonally dominant, the present correctness fallback computes a residual-certified upper bound through

```text
gamma
<= lambda_max(K_A^-1/2 D_AA K_A^-1/2)
<= trace(K_A^-1 D_AA).
```

That fallback still uses sparse factorization of `K_A` only to construct the `gamma` certificate. It is not a global `H^-1` Riesz solve.

## 7. Multiple excitations and ports

For port source matrix

```text
B = [b_1,...,b_p],
```

a single reduced space is constructed over the joint state-by-excitation domain. At one thermal state,

```text
A_r C = V^H B,
X_r = V C.
```

The top-level nonlinear core exposes this directly through

```text
build_reduced_electromagnetics(..., port_set=ports).
```

Thus every declared unit-port right-hand side participates in the same residual-greedy basis construction and certificate.

The reduced multiport impedance is

```text
Z_r = i omega B^T X_r,
R_r = Re(Z_r),
L_r = Im(Z_r)/omega.
```

For each reduced port column `j`, define

```text
r_j = b_j - A x_r,j,
epsilon_j <= beta_H^-1 upper(||r_j||_(H^-1)).
```

For source port `i`,

```text
|Delta Z_ij|
<= omega
   * upper(||b_i||_(H^-1))
   * upper(||r_j||_(H^-1))/beta_H.
```

Therefore

```text
|Delta R_ij| <= |Delta Z_ij|,
|Delta L_ij| <= |Delta Z_ij| / omega.
```

Mutual inductance entries inherit the same `L` bound. This gives certified `Z/R/L/M` from a reduced equilibrium solve without a full-order field equilibrium solve at query time.

The reduced model stores the same Riesz-action factory used during construction, so online multiport certification automatically reuses the physical-block PCG path and does not silently revert to the global sparse-LU reference action.

## 8. Reduced electromagnetic heat source

For thermal test mode `j`, the projected Joule source is

```text
q_j(a) = Re[x_r^H H_j(a) x_r].
```

The nonlinear tetrahedral implementation evaluates `H_j(a)` and its state derivatives as sparse matrices. The reduced heat-source Jacobian uses

```text
dc/da_k = -A_r^-1 [V^H (dA/da_k) V] c
```

and

```text
dq_j/da_k
= 2 Re[(dx/da_k)^H H_j x]
  + Re[x^H (dH_j/da_k) x].
```

Thus `q_em,r(a)` and `dq_em,r/da` remain available without invoking the legacy dense full-order compatibility operators.

## 9. What is and is not certified

The current finite candidate-set reducer certifies every state/excitation explicitly supplied to the construction set. It does **not** claim that a finite set proves an entire continuous thermal/geometry/frequency domain.

A continuous nonlinear-domain certificate remains a separate obligation. It must bound the residual over the continuous parameter domain rather than infer coverage from sampling density.

The formal nonlinear ROM no longer requires an exact global `H(a)^-1` action. The remaining large-scale Riesz obligations are narrower and explicit:

1. replace the default exact magnetic/scalar block actions by memory-scalable certified multilevel/auxiliary-space actions;
2. replace the non-diagonally-dominant sparse-LU `gamma` certificate fallback by a scalable factorization-free proof.

Neither obligation changes the residual theorem, the physical-block inequality, or the multiport output certificate.

## 10. Implementation map

```text
sdfmpneo/em/riesz_action.py
    CertifiedRieszAction
    SparseLUReferenceRieszAction

sdfmpneo/em/certified_riesz.py
    CertifiedPCGRieszAction
    CertifiedEnergyPreconditioner

sdfmpneo/em/block_riesz.py
    PhysicalBlockEnergyPreconditioner
    SparseLUExactBlockPreconditioner
    make_physical_block_pcg_riesz_factory

sdfmpneo/em/sparse_reduced.py
    SparseEnergyResidualGreedyEMReducer
    SparseEnergyReducedEMModel
    SparseEnergyReductionCertificate

sdfmpneo/em/ports.py
    evaluate_reduced_physical_certified
    CertifiedEnergyReducedMultiPortResult

sdfmpneo/tetra_core.py
    build_reduced_electromagnetics
```

The legacy dense `ResidualGreedyEMReducer` remains only for affine/orthogonal verification paths and is not the certified nonlinear production reduction path.
