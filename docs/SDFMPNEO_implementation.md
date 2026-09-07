> 版本说明：本文保留先前版本的架构/证明背景。0.10 科研训练与推理入口、修复及运行方式见 [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md)；其中的旧完成状态不作为当前验收结论。

# SDF-MPNEO implementation specification

This document is the executable architecture contract for the current SDF-MPNEO electromagnetic–thermal core. It must remain consistent with `docs/SDFMPNEO_theory.tex`, `docs/SPARSE_ENERGY_SOLVER.md`, `docs/CERTIFIED_RIESZ_ACTION.md`, and `docs/SPARSE_ENERGY_REDUCTION.md`.

The production scientific path obeys the following invariants:

- no geometry-specific impedance solver in the core path;
- no full-order electromagnetic or thermal **solution snapshots** for reduced-space construction;
- no transient solution labels for analytic-network training;
- no empirical fixed electromagnetic/thermal rank, neural width/depth, near/far split, penalty gauge, block damping, or unexplained solver tolerance;
- all accepted approximation orders and reduced dimensions are controlled by explicit error/convergence certificates;
- the certified nonlinear tetrahedral electromagnetic path remains sparse at full order;
- electromagnetic error certification always uses the state-dependent physical energy metric;
- the fixed reference energy metric is only a reduced-basis coordinate device;
- any missing error contribution is exposed as an uncertified/pending term rather than absorbed into a fitted safety factor.

---

## 1. Current package map

```text
sdfmpneo/
├── spatial/
│   ├── tetra3d.py                 # tetrahedral topology, Nedelec/P1 assembly
│   └── barycentric_polynomial.py  # exact polynomial integration
│
├── em/
│   ├── tetra_nonlinear.py         # certified nonlinear sparse A-psi problem
│   ├── reciprocal_series.py       # certified copper reciprocal law
│   ├── sparse_solver.py           # sparse full-order correctness solvers
│   ├── energy_solver.py           # H=K+D physical energy theorem
│   ├── riesz_action.py            # certified Riesz-action protocol + reference backend
│   ├── certified_riesz.py         # certified PCG action + preconditioner contract
│   ├── block_riesz.py             # physical magnetic/scalar block theorem
│   ├── sparse_reduced.py          # production snapshot-free sparse EM reduction
│   ├── ports.py                   # sparse port projection + certified Z/R/L/M
│   ├── tetra_nonlinear_diagnostics.py
│   ├── tetra.py                   # affine tetrahedral verification path
│   └── reduced.py                 # legacy dense verification reducer only
│
├── thermal/
│   └── spectral.py                # deterministic thermal spectrum/rank certificate
│
├── analytic/                      # intrinsic arbitrary-time analytic network
├── training/                      # physical residual and topology growth
├── certification/                 # state/output/material/coupling certificates
├── tetra_core.py                  # one-call nonlinear electrothermal production core
└── model.py                       # arbitrary-time prediction interfaces
```

The orthogonal-grid and affine tetrahedral implementations remain verification tools. They do not redefine the certified nonlinear production path.

---

## 2. Nonlinear tetrahedral electromagnetic operator

The production electromagnetic state uses gauge-reduced reciprocal `A-psi` coordinates. At thermal reduced state `a`,

```text
A(a)x(a)=b,
A(a)=K+iD(a),
K>=0,
D(a)>=0.
```

`K` is the gauge-reduced magnetic curl-curl contribution. `D(a)` is the conductive electric-energy contribution. Copper, seawater, and other conductive regions are represented by spatial constitutive laws inside `D(a)`.

Required production interface:

```python
class NonlinearTetrahedralApsiProblem:
    def operator_sparse(self, a): ...
    def operator_derivative_sparse(self, a, k): ...
    def loss_operator_sparse(self, j, a): ...
    def loss_operator_derivative_sparse(self, j, k, a): ...
    def constitutive_certificate(self, a): ...
```

Dense compatibility methods may exist only for small verification problems. Production reduction, loss evaluation, ports, and certification must not depend on them.

No empirical AC-resistance correction is added. Copper skin/proximity effects and seawater induced-current effects belong to the volumetric electromagnetic field problem.

---

## 3. Certified nonlinear material integration

For copper

```text
rho(T)=rho_ref[1+alpha(T-T_ref)],
sigma(T)=sigma_ref/d(T),
d(T)=1+alpha(T-T_ref).
```

Inside a P1 tetrahedron, `d(x)` is affine. Reciprocal and reciprocal-square terms are represented by certified geometric series about the exact element range centre. The retained order is the smallest integer satisfying the declared constitutive remainder bound.

Every retained polynomial term is integrated analytically against first-order Nedelec basis functions. The same constitutive representation is used by

```text
A(a),
dA/da,
loss operators,
loss derivatives,
regional copper/seawater powers.
```

The field and heat-source paths therefore cannot silently use inconsistent conductivity approximations.

---

## 4. Physical electromagnetic energy theorem

Define

```text
H(a)=K+D(a).
```

For every complex field vector `v`,

```text
|v^H A(a)v|
= sqrt[(v^H K v)^2+(v^H D(a)v)^2]
>= v^H H(a)v/sqrt(2).
```

Therefore

```text
beta_H>=1/sqrt(2)
```

and for residual `r=b-A(a)x_h`,

```text
||x-x_h||_H(a)
<= sqrt(2)||r||_(H(a)^-1).
```

This structural constant is independent of the copper/seawater conductivity ratio. A fitted minimum singular value or conductivity correction is not part of the production certificate.

The production method consumes the dual norm through a **certified Riesz action**. It does not require an exact global `H(a)^-1` solve.

---

## 5. Certified Riesz-action contract

A Riesz action approximates

```text
y=H^-1 r
```

by `y_tilde` and must prove

```text
||y-y_tilde||_H<=delta.
```

Then

```text
max(0,||y_tilde||_H-delta)
<= ||r||_(H^-1)
<= ||y_tilde||_H+delta.
```

The executable interface is conceptually

```python
class CertifiedRieszAction:
    def solve(self, rhs, requested_energy_action_error): ...
    def decide_dual_norm(self, rhs, threshold): ...
```

All reducer and output decisions use certified intervals, not an internal Krylov tolerance.

For a preconditioner `P` satisfying

```text
H>=mP,
```

the PCG residual `s=r-Hy_tilde` gives

```text
||H^-1r-y_tilde||_H^2
<= (1/m)s^H P^-1 s.
```

`CertifiedPCGRieszAction` stops only after a physical error target or a dual-norm threshold decision has been proved.

---

## 6. Physical magnetic/scalar block preconditioner

Partition the local energy metric into gauge-reduced magnetic coordinates and conductive scalar coordinates:

```text
H = [[K_A + D_AA, D_Apsi],
     [D_psiA,       D_psipsi]].
```

Tree-cotree gauge elimination makes `K_A>0`; conductivity assembly gives `D>=0`.

If a certified constant `gamma` satisfies

```text
D_AA<=gamma K_A,
```

then conductive Cauchy-Schwarz plus the optimal Young inequality gives

```text
H>=m_phys(gamma)diag(K_A,D_psipsi),
```

where

```text
m_phys(gamma)
=2/[2+gamma+sqrt(gamma^2+4gamma)].
```

No fitted block weight or damping parameter appears.

### 6.1 Replaceable block actions

If the magnetic and scalar block actions apply `P_K^-1` and `P_E^-1` and prove

```text
K_A      >= m_K P_K,
D_psipsi >= m_E P_E,
```

then

```text
P=diag(P_K,P_E)
```

satisfies

```text
H>=m_phys(gamma)min(m_K,m_E)P.
```

This is the production scalability interface. The current default `SparseLUExactBlockPreconditioner` uses exact block matrices as `P_K` and `P_E`, hence `m_K=m_E=1`, but it is only a correctness backend. Regression tests replace both blocks with non-LU certified actions while `scipy.sparse.linalg.splu` is disabled.

### 6.2 Gamma certificate

The first certificate attempts normalized Gershgorin. When the magnetic block is not diagonally dominant, the current correctness fallback uses an explicitly residual-certified sparse factorization of `K_A` to prove

```text
gamma
<= lambda_max(K_A^-1/2 D_AA K_A^-1/2)
<= trace(K_A^-1 D_AA).
```

The factorization residual is propagated into the trace upper bound. Failure produces an error; no diagonal shift is introduced.

This factorization is used only for `gamma` certificate construction. It is not a global `H^-1` Riesz solve. Replacing this fallback by a scalable factorization-free proof is a remaining production obligation.

---

## 7. Output-driven full-order correctness solve

The separate full-order verification solver remains available. Requested engineering-output accuracy determines its required field-state accuracy.

For closed port source `b_i`,

```text
|Delta Z_ij|
<= omega||b_i||_(H^-1)||Delta x_j||_H.
```

The correctness solver follows

```text
energy-preconditioned sparse Krylov
        |
        +--> recompute true residual
        |
        +--> physical H-error certificate
        |
        +--> if insufficient: complete sparse LU fallback
        |
        +--> recompute and certify again
```

This full-order `A` fallback is a verification/correctness path. It is not the equilibrium solve used by the production reduced model.

---

## 8. Production snapshot-free electromagnetic reduction

The certified nonlinear path uses `SparseEnergyResidualGreedyEMReducer` with the physical-block Riesz factory created by

```python
make_physical_block_pcg_riesz_factory(problem)
```

through `TetrahedralElectroThermalCore.build_reduced_electromagnetics()`.

### 8.1 Reduced equilibrium

```text
A_r(a)=V^H A(a)V,
b_r=V^H b,
A_r(a)c(a)=b_r,
x_r(a)=Vc(a).
```

Only `A_r` is dense. The full-order operator remains sparse.

### 8.2 Local residual certificate

```text
r(a)=b-A(a)x_r(a),
eta(a)=sqrt(2)||r(a)||_(H(a)^-1).
```

A state/excitation is accepted only when the certified dual-norm **upper bound** proves the declared energy-state target.

### 8.3 Residual-Riesz enrichment

For an unresolved state/excitation,

```text
y_tilde ~= H(a)^-1 r(a)
```

is obtained from the certified Riesz action and appended to the span. The complete residual is then recomputed. Inexact enrichment can change convergence rate but cannot create a false certificate.

No full-order solution state `A(a)^-1b` is used as a basis snapshot.

### 8.4 Reference coordinate metric

The error theorem always uses local `H(a)`. The basis coordinate system uses only

```text
H0=H(0).
```

For candidate basis matrix `W`, form the small Gram matrix

```text
G=W^H H0 W=LL^H
```

and whiten

```text
V=WL^(-H),
V^H H0 V=I
```

up to floating-point backward error. No dense full-order Cholesky factor is formed.

### 8.5 Finite versus continuous certification

The greedy constructor certifies the explicitly declared candidate state/excitation set. It does not convert sample density into a continuous-domain proof.

---

## 9. Joint WPT multiport reduction and outputs

A WPT port set contains unit closed impressed-current source columns

```text
B=[b_1,...,b_p].
```

The top-level nonlinear core accepts

```python
core.build_reduced_electromagnetics(
    candidate_thermal_states,
    port_set=ports,
    requested_energy_state_error=epsilon_H,
)
```

and calls the same `build_multi_rhs` construction for every port RHS. Thus one global basis covers the declared state-by-port domain.

At one thermal state

```text
A_r C=V^H B,
X_r=VC,
Z_r=i omega B^T X_r,
R_r=Re(Z_r),
L_r=Im(Z_r)/omega.
```

For reduced residual column `r_j`,

```text
||e_j||_H
<= beta_H^-1 upper(||r_j||_(H^-1)).
```

Hence

```text
|Delta Z_ij|
<= omega
   * upper(||b_i||_(H^-1))
   * upper(||r_j||_(H^-1))/beta_H.
```

and

```text
|Delta R_ij|<=|Delta Z_ij|,
|Delta L_ij|<=|Delta Z_ij|/omega.
```

Mutual inductance inherits the same inductance bound.

The reduced model stores its Riesz-action factory. `ports.evaluate_reduced_physical_certified()` reuses that factory by default, so online output certification cannot silently revert to the old global sparse-LU reference action.

Regression tests execute

```text
build_ports
-> joint multi-RHS build_reduced_electromagnetics
-> evaluate_reduced_physical_certified
-> certified Z/R/L/M
```

with `SparseLUReferenceRieszAction` disabled.

The port coordinate projection accepts the sparse gauge basis directly and must not densify it.

---

## 10. Reduced Joule source and Jacobian

For thermal test mode `j`,

```text
q_j(a)=Re[x_r^H H_j(a)x_r].
```

The reduced electromagnetic sensitivity is

```text
A_r dc/da_k
= -[V^H(dA/da_k)V]c
  + V^H db/da_k.
```

For the current impressed-source model `db/da_k=0`. Therefore

```text
dq_j/da_k
=2Re[(dx/da_k)^H H_jx]
 +Re[x^H(dH_j/da_k)x].
```

No finite-difference thermal perturbation is required. Copper and seawater losses are evaluated from the same field state and constitutive operators used by the electromagnetic equation.

---

## 11. Thermal spectral reduction

Solve

```text
K_T phi_i=lambda_i M_T phi_i,
phi_i^T M_T phi_j=delta_ij.
```

The thermal basis is deterministic physics, not network-generated.

The retained rank is not a configured integer. With a certified initial tail and source dual bound, the builder selects the smallest rank satisfying the spectral-tail target. Otherwise the correctness implementation retains the full discrete spectrum.

For first omitted eigenvalue `lambda_(r+1)`, initial tail `E0`, and source dual bound `Q`,

```text
||T_tail(t)||_M
<= exp(-lambda_(r+1)t)E0
 +[1-exp(-lambda_(r+1)t)]Q/lambda_(r+1).
```

A production partial eigensolver must retain a certified lower bound for the first omitted eigenvalue.

---

## 12. Closed electrothermal reduced operator

The reduced thermal dynamics are

```text
da/dt+Lambda_T a=g_em(a;U)+f_T(U).
```

`g_em` comes from the reduced electromagnetic equilibrium and projected Joule source. The electromagnetic field is quasi-static relative to thermal evolution; no electromagnetic transient time marching is introduced.

---

## 13. Intrinsic analytic evolution network

The network represents

```text
(a0,U,t)->a(t)
```

inside one analytic graph. Initial state and static operating conditions are graph-internal analytic nodes; no external conditioning network generates its coefficients.

Analytic atoms have the closed form

```text
t^m exp[-(n dot lambda)t].
```

The graph supplies both `a(t)` and `da/dt(t)` analytically. Exact/state-space evaluation handles repeated and near-repeated decay rates without an empirical resonance threshold.

No inference time stepping is allowed in the final arbitrary-time evaluator.

---

## 14. Residual-grown analytic topology

The thermal physical residual is

```text
R_T=da/dt+Lambda_T a-g_em(a;U)-f_T(U).
```

Candidate analytic response nodes are generated from active analytic/state/parameter nodes. Candidate selection uses unresolved physical residual information and exact reduced electromagnetic heat-source sensitivities.

After adding a candidate, the complete nonlinear residual is re-evaluated. A tangent approximation alone cannot certify acceptance.

The scientific method contains no fixed `num_layers` or `hidden_width`. Growth terminates only when the propagated certified target is met or certification is explicitly reported as unsuccessful.

---

## 15. Error decomposition

The electromagnetic contribution remains separated by physical cause:

```text
eta_EM
<= eta_constitutive
 + eta_algebraic
 + eta_EM_ROM
 + eta_mesh
 + eta_outer.
```

Current executable terms:

```text
eta_constitutive : nonlinear material-series remainder,
eta_algebraic    : certified sparse field/Riesz action error,
eta_EM_ROM       : reduced-space residual error on the declared certified set/domain.
```

Pending terms are explicit:

```text
eta_mesh,
eta_outer.
```

They cannot be hidden by increasing another tolerance.

For thermal evolution, if

```text
kappa
=lambda_min(Lambda_T)
 -sup lambda_max(sym(dg_em/da))
```

has a verified positive lower bound, a uniform-in-time estimate may use

```text
||e_T(t)||<=eta_total/kappa.
```

Otherwise only a finite-time certificate may be returned.

---

## 16. Offline and online contracts

### Offline fixed-operator-family construction

```text
conforming tetrahedral geometry/material regions
        |
        +--> P1 thermal M_T,K_T
        |       +--> certified thermal basis/rank
        |
        +--> sparse nonlinear A-psi operator
                +--> local H(a)
                +--> constitutive certificate
                +--> physical-block certified Riesz action
                +--> joint port RHS residual-Riesz ROM growth
                +--> reduced loss/Jacobian operators
        |
        +--> closed reduced electrothermal dynamics
                +--> residual-grown intrinsic analytic network
        |
        +--> combined certificates
```

No full-order solved field/transient trajectories are consumed as training data.

### Online arbitrary-time query

```python
def evaluate(model, operating, T0, t):
    a0 = model.thermal.project_initial(T0)
    a, da = model.analytic.evaluate(a0, operating, t)

    em = model.em.solve_reduced(a, operating)
    outputs = model.em.outputs(em, a, operating)
    cert = model.certification.evaluate(a, da, em, outputs)
    return Prediction(a, outputs, cert)
```

There is no loop over thermal time steps. Sparse full-order matrices may still be traversed for a-posteriori certification of reduced outputs, but the electromagnetic equilibrium solve remains reduced-order.

---

## 17. Verification-only interfaces

The following remain deliberately available for mathematical regression:

- orthogonal compatible grids;
- affine tetrahedral temperature dependence;
- `ParametricEMProblem`;
- `ReducedEMModel`;
- `ResidualGreedyEMReducer` with dense Cholesky-Riesz coordinates;
- `SparseLUReferenceRieszAction` as a generic global-Riesz correctness backend;
- dense direct solves for small manufactured systems.

They must not be cited as the certified nonlinear production architecture.

`TetrahedralElectroThermalCore.build_reduced_electromagnetics()` is reserved for the certified nonlinear physical-block sparse-energy path.

`TetrahedralElectroThermalCore.build_affine_verification_reduced_electromagnetics()` explicitly names the legacy affine verification path.

---

## 18. Remaining production obligations

Implementation work is ordered by unresolved scientific bottlenecks:

1. **Certified multilevel block actions.** Replace the default exact magnetic/scalar block LU actions by memory-scalable auxiliary-space/multilevel actions with proved `m_K` and `m_E`.
2. **Factorization-free gamma proof.** Replace the residual-certified sparse-LU generalized-trace fallback by a scalable proof of `D_AA<=gamma K_A`.
3. **Continuous nonlinear-domain certification.** Extend finite state/excitation certificates to the declared continuous thermal/geometry/frequency/source domain.
4. **CAD/conforming mesh pipeline.** Import/generate actual round/rounded-square underwater WPT conductors, package, and seawater domains while preserving compatible topology.
5. **Mesh and outer-domain error.** Add spatial discretization and open/infinite-domain truncation certificates.
6. **Scalable thermal spectrum.** Compute only required thermal modes and certify the first omitted eigenvalue.
7. **Solid-conductor terminal ports.** Add terminal-current constrained excitation when closed impressed-current cochains are not the intended physical port model.
8. **Geometry/operator-family parameterization.** Preserve the same certification logic when geometry changes the thermal/electromagnetic operators.
9. **Analytic-network global convergence/compression.** Certify topology growth over the full parameter domain and compress large exact state-space realizations without losing error bounds.

A numerical prediction may be produced for diagnostics when one of these certificates is unavailable, but it must be marked `certified=False`. No unavailable proof is replaced by an empirical safety factor.
