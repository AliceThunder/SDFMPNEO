# SDF-MPNEO implementation specification

This document is the executable architecture contract for the current SDF-MPNEO electromagnetic–thermal core. It must remain consistent with `docs/SDFMPNEO_theory.tex`, `docs/SPARSE_ENERGY_SOLVER.md`, and `docs/SPARSE_ENERGY_REDUCTION.md`.

The production scientific path obeys the following invariants:

- no geometry-specific impedance solver in the core path;
- no full-order electromagnetic or thermal **solution snapshots** for reduced-space construction;
- no transient solution labels for analytic-network training;
- no empirical fixed electromagnetic/thermal rank, neural width/depth, near/far split, penalty gauge, or solver tolerance;
- all accepted approximation orders and reduced dimensions are controlled by explicit error/convergence certificates;
- the certified nonlinear tetrahedral electromagnetic path remains sparse at full order;
- the state-dependent physical energy metric is used for electromagnetic error certification;
- the fixed reference energy metric is only a reduced-basis coordinate device, never a substitute for the local error theorem;
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
│   ├── sparse_solver.py           # sparse energy/full-order correctness solvers
│   ├── energy_solver.py           # H=K+D physical energy theorem
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
├── tetra_core.py                  # one-call unstructured electrothermal builder
└── model.py                       # arbitrary-time prediction interfaces
```

The orthogonal-grid and affine tetrahedral implementations remain useful for mathematical verification. They are not allowed to redefine the certified nonlinear production path.

---

## 2. Nonlinear tetrahedral electromagnetic operator

The production electromagnetic state uses gauge-reduced reciprocal `A-psi` coordinates. At thermal reduced state `a`,

```text
A(a) x(a) = b.
```

The sparse operator is decomposed physically as

```text
A(a) = K + i D(a),
K >= 0,
D(a) >= 0.
```

`K` is the gauge-reduced magnetic curl-curl contribution and `D(a)` is the conductive electric-energy contribution. Copper, seawater, and all other conductive regions are represented through spatial constitutive laws inside `D(a)`.

Required production interface:

```python
class NonlinearTetrahedralApsiProblem:
    def operator_sparse(self, a): ...
    def operator_derivative_sparse(self, a, k): ...
    def loss_operator_sparse(self, j, a): ...
    def loss_operator_derivative_sparse(self, j, k, a): ...
    def constitutive_certificate(self, a): ...
```

Dense methods may exist only as compatibility wrappers for small verification tests. Production reduction, loss evaluation, ports, and certificates must not depend on them.

No empirical AC-resistance correction is added. Copper skin/proximity and seawater induced-current effects belong to the volumetric electromagnetic field model.

---

## 3. Certified nonlinear material integration

For copper resistivity

```text
rho(T) = rho_ref [1 + alpha(T-T_ref)]
```

conductivity is

```text
sigma(T) = sigma_ref / d(T),
d(T) = 1 + alpha(T-T_ref).
```

Inside a P1 tetrahedron, `d(x)` is affine. The reciprocal and reciprocal-square terms are represented by certified geometric series about the exact element range centre. The retained order is the smallest integer satisfying the declared constitutive remainder bound.

Every retained polynomial term is integrated analytically against first-order Nedelec basis functions through barycentric monomial identities. The same constitutive representation is used by

```text
A(a),
dA/da,
loss operators,
loss derivatives,
regional copper/seawater powers.
```

The field and heat-source paths therefore cannot silently use inconsistent conductivity approximations.

---

## 4. Physical electromagnetic energy metric

At every thermal state define

```text
H(a) = K + D(a).
```

For any complex field vector `v`,

```text
|v^H A(a) v|
= sqrt[(v^H K v)^2 + (v^H D(a) v)^2]
>= v^H H(a) v / sqrt(2).
```

Thus the production stability constant is structural:

```text
beta_H >= 1/sqrt(2).
```

For residual

```text
r = b - A(a) x_h,
```

the state error satisfies

```text
||x-x_h||_H(a)
<= sqrt(2) ||r||_(H(a)^-1).
```

This theorem replaces the old production concept of a fixed Riesz metric plus an externally estimated minimum singular value. The old construction may remain in verification modules, but it is not the nonlinear production certificate.

Required energy interface:

```python
class ApsiEnergyMetric:
    def solve(self, rhs): ...      # H(a) y = rhs
    def norm(self, x): ...         # sqrt(x^H H(a) x)
    def dual_norm(self, r): ...    # sqrt(r^H H(a)^-1 r)
```

The current implementation uses sparse LU for the exact `H(a)^-1` action. This is a correctness baseline. A later scalable implementation must preserve the same mathematical interface and certificate.

---

## 5. Output-driven full-order sparse correctness solve

A scientific linear-solver tolerance is not configured independently. Instead, requested engineering-output accuracy determines the required field-state accuracy.

For closed port source `b_i`,

```text
|Delta Z_ij|
<= omega ||b_i||_(H^-1) ||Delta x_j||_H.
```

Therefore a requested per-entry impedance error `epsilon_Z` determines the required energy-state error.

The correctness solver follows:

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

No ILU drop tolerance, fitted fill factor, or relaxed residual threshold defines correctness. The sparse LU fallback is not claimed to be production-scalable; it is the deterministic correctness reference until the multilevel `H^-1` stage is implemented.

---

## 6. Production snapshot-free electromagnetic reduction

The certified nonlinear path uses `SparseEnergyResidualGreedyEMReducer`.

### 6.1 Reduced equilibrium

For basis `V`,

```text
A_r(a) = V^H A(a) V,
b_r = V^H b,
A_r(a)c(a)=b_r,
x_r(a)=V c(a).
```

Only `A_r` is dense. The full-order `A(a)` remains sparse.

### 6.2 Local residual certificate

```text
r(a) = b - A(a)x_r(a),
eta(a) = sqrt(2)||r(a)||_(H(a)^-1).
```

The reducer accepts a state/excitation only when the declared energy-state error target is satisfied. It never converts this target into an unrelated algebraic residual tolerance.

### 6.3 Residual-Riesz enrichment

For the worst certified residual in the current construction domain,

```text
y = H(a)^-1 r(a).
```

Append `y` to the reduced span. No full-order solution state `A(a)^-1b` is used as a training snapshot.

For multiple excitations, construct one space over the joint state-by-RHS domain.

### 6.4 Local metric versus reference coordinate metric

The error theorem always uses the local physical metric

```text
H(a)=K+D(a).
```

The basis is stored using only the intrinsic reference-state metric

```text
H0 = H(a=0).
```

`H0` is a coordinate metric, not an error surrogate.

For candidate basis matrix `W`, form only the small Gram matrix

```text
G = W^H H0 W = L L^H
```

and whiten

```text
V = W L^(-H).
```

Hence

```text
V^H H0 V = I
```

up to floating-point backward error, without a dense full-order Cholesky factor.

High-contrast orthogonality tests use a conditioning-aware arithmetic envelope based on

```text
|V|^T |H0| |V|
```

and the standard `gamma_k` floating-point model. A fixed `C*eps*n` test is invalid when the physical energy entries undergo large cancellation.

### 6.5 Finite versus continuous certification

The current greedy constructor certifies the explicitly declared candidate state/excitation set. This finite certificate must not be presented as a proof over a continuous parameter domain.

A production continuous-domain layer must bound the residual supremum over thermal state, geometry, frequency, source, and any other declared parameters using deterministic interval/spectral arguments or an equivalent rigorous method.

---

## 7. Certified reduced multiport outputs

Let port source matrix be

```text
B=[b_1,...,b_p].
```

At one thermal state solve only the reduced system

```text
A_r C = V^H B,
X_r = V C.
```

Then

```text
Z_r = i omega B^T X_r,
R_r = Re(Z_r),
L_r = Im(Z_r)/omega.
```

For each port column,

```text
r_j = b_j - A x_r,j,
epsilon_j = sqrt(2)||r_j||_(H^-1).
```

The output certificate is

```text
|Delta Z_ij|
<= omega ||b_i||_(H^-1) epsilon_j,

|Delta R_ij| <= |Delta Z_ij|,
|Delta L_ij| <= |Delta Z_ij|/omega.
```

Mutual-inductance entries inherit the same inductance bound.

The full-order sparse operator is used only for residual/Riesz certification. No full-order electromagnetic equilibrium solve is hidden inside the reduced multiport query.

The port coordinate projection must accept the sparse gauge basis directly. `build_ports()` must not call `toarray()` on the full-order gauge basis.

---

## 8. Reduced Joule source and Jacobian

For each thermal test mode `j`,

```text
q_j(a) = Re[x_r^H H_j(a) x_r].
```

The loss matrices `H_j(a)` are sparse full-order physical operators projected through the reduced state. The reduced electromagnetic sensitivity is

```text
A_r dc/da_k
= -[V^H (dA/da_k) V]c
  + V^H db/da_k.
```

For the current impressed-source model `db/da_k=0`.

Then

```text
dq_j/da_k
= 2 Re[(dx/da_k)^H H_j x]
  + Re[x^H (dH_j/da_k) x].
```

No finite-difference thermal perturbation is required.

Copper and seawater total losses are evaluated from the same electromagnetic state and constitutive operators used in the field equation.

---

## 9. Thermal spectral reduction

Solve

```text
K_T phi_i = lambda_i M_T phi_i,
phi_i^T M_T phi_j = delta_ij.
```

The thermal basis is deterministic physics, not network-generated.

The retained rank is not a configured integer. With a certified initial tail and source dual bound, the builder selects the smallest rank satisfying the spectral-tail state target. Otherwise the correctness implementation retains the full discrete spectrum.

For first omitted eigenvalue `lambda_(r+1)`, initial tail `E0`, and source dual bound `Q`,

```text
||T_tail(t)||_M
<= exp(-lambda_(r+1)t) E0
 + [1-exp(-lambda_(r+1)t)]Q/lambda_(r+1).
```

Production scaling requires a partial eigensolver together with a certified lower bound on the first omitted eigenvalue.

---

## 10. Closed electrothermal reduced operator

The reduced thermal dynamics are

```text
da/dt + Lambda_T a = g_em(a;U) + f_T(U).
```

`g_em` is obtained from the reduced electromagnetic equilibrium and the projected Joule source:

```python
class ElectroThermalClosure:
    def g(self, a, U):
        c = em.solve_reduced(a, U)
        return em.project_heat_source(a, U, c)

    def jacobian(self, a, U):
        return em.exact_reduced_heat_jacobian(a, U)
```

The electromagnetic field is quasi-static relative to the thermal evolution, so no electromagnetic transient time marching is introduced into the thermal surrogate.

---

## 11. Intrinsic analytic evolution network

The network represents the arbitrary-time map

```text
(a0,U,t) -> a(t)
```

inside one analytic graph. Initial state and static operating conditions are graph-internal analytic nodes; an external conditioning network does not generate the network coefficients.

Analytic atoms are closed under the required operations and have the form

```text
t^m exp[-(n dot lambda)t].
```

The graph supplies both

```text
a(t),
da/dt(t)
```

analytically.

The response operator uses exact/state-space or entire-function evaluation so repeated/near-repeated decay rates do not require an empirical resonance threshold.

No inference time stepping is allowed in the final arbitrary-time evaluator.

---

## 12. Residual-grown analytic topology

The physical thermal residual is

```text
R_T = da/dt + Lambda_T a - g_em(a;U) - f_T(U).
```

Candidate analytic response nodes are generated from already active analytic/state/parameter nodes. Candidate selection is driven by unresolved physical residual information and exact reduced electromagnetic heat-source sensitivities.

After adding a candidate, the complete nonlinear residual is re-evaluated. A tangent approximation alone cannot certify acceptance.

The scientific method contains no fixed `num_layers` or `hidden_width`. Growth terminates only when the propagated certified error target is met or the procedure explicitly reports that certification has not been achieved.

---

## 13. Error decomposition

The electromagnetic contribution to the coupled thermal error budget must keep distinct causes separate:

```text
eta_EM
<= eta_constitutive
 + eta_algebraic
 + eta_EM_ROM
 + eta_mesh
 + eta_outer.
```

Current executable terms include:

```text
eta_constitutive : nonlinear material-series remainder,
eta_algebraic    : sparse field/Riesz algebraic error where certified,
eta_EM_ROM       : reduced-space residual error on the declared certified set/domain.
```

Pending production terms are explicit:

```text
eta_mesh,
eta_outer.
```

They cannot be hidden by increasing another tolerance.

For the thermal evolution, if

```text
kappa
= lambda_min(Lambda_T)
  - sup lambda_max(sym(dg_em/da))
```

has a verified positive lower bound, a uniform-in-time estimate may use

```text
||e_T(t)|| <= eta_total/kappa.
```

Otherwise only a finite-time certificate may be returned.

---

## 14. Offline construction contract

The current fixed-operator-family offline flow is

```text
conforming tetrahedral geometry/material regions
        |
        +--> P1 thermal M_T,K_T
        |       +--> certified thermal basis/rank
        |
        +--> sparse nonlinear A-psi operator
                +--> local physical energy H(a)
                +--> constitutive certificate
                +--> sparse residual-Riesz EM basis growth
                +--> joint physical port RHS coverage
                +--> reduced loss/Jacobian operators
        |
        +--> closed reduced electrothermal dynamics
                +--> residual-grown intrinsic analytic network
        |
        +--> combined certificates
```

No full-order solved field/transient trajectories are consumed as training data.

When geometry/frequency/operator-family parameterization is enabled, the same construction must be extended with rigorous reference maps/operator remainders and continuous-domain certification. Sampling density alone cannot define validity.

---

## 15. Online query contract

For a compiled fixed operator family:

```python
def evaluate(model, operating, T0, t):
    a0 = model.thermal.project_initial(T0)
    a, da = model.analytic.evaluate(a0, operating, t)

    em = model.em.solve_reduced(a, operating)
    outputs = model.em.outputs(em, a, operating)
    cert = model.certification.evaluate(a, da, em, outputs)

    return Prediction(
        thermal_coeff=a,
        Z=outputs.Z,
        R=outputs.R,
        L=outputs.L,
        M=outputs.M,
        P_cu=outputs.P_cu,
        P_sea=outputs.P_sea,
        certificate=cert,
    )
```

There is no loop over thermal time steps.

The current correctness implementation may still traverse sparse full-order matrices for a-posteriori certification of reduced electromagnetic outputs. The **equilibrium solve** remains reduced-order. A later deployment compiler may replace these certification actions with offline-separated bounds/tensors once rigorous continuous-domain envelopes are available.

---

## 16. Verification-only interfaces

The following remain deliberately available for mathematical regression:

- orthogonal compatible grids;
- affine tetrahedral temperature dependence;
- `ParametricEMProblem`;
- `ReducedEMModel`;
- `ResidualGreedyEMReducer` with dense Cholesky-Riesz coordinates;
- dense direct solves for small manufactured systems.

They are validation tools. They must not be cited as the certified nonlinear production architecture.

`TetrahedralElectroThermalCore.build_reduced_electromagnetics()` is reserved for the certified nonlinear sparse-energy path.

`TetrahedralElectroThermalCore.build_affine_verification_reduced_electromagnetics()` explicitly names the legacy affine verification path.

---

## 17. Remaining production obligations

The next implementation work is ordered by the actual unresolved bottlenecks:

1. **Scalable physical-energy inverse.** Replace exact sparse-LU `H(a)^-1` Riesz actions with a multilevel/auxiliary-space realization whose inexactness is independently certified. No heuristic ILU drop parameters may define the scientific result.
2. **Scalable full-order correctness fallback.** Remove dependence on complete sparse LU for very large `A(a)` while preserving the true-residual state certificate.
3. **Continuous nonlinear-domain certification.** Extend finite state/excitation certificates to the declared continuous thermal/geometry/frequency/source domain.
4. **CAD/conforming mesh pipeline.** Import/generate actual round/rounded-square underwater WPT conductors, package, and seawater domains while preserving compatible topology.
5. **Mesh and outer-domain error.** Add spatial discretization and open/infinite-domain truncation certificates.
6. **Scalable thermal spectrum.** Compute only required thermal modes and certify the first omitted eigenvalue.
7. **Solid-conductor terminal ports.** Add terminal-current constrained excitation when closed impressed-current cochains are not the intended physical port model.
8. **Geometry/operator-family parameterization.** Preserve the same certification logic when geometry changes the thermal/electromagnetic operators.
9. **Analytic-network global convergence/compression.** Certify topology growth over the full parameter domain and compress large exact state-space realizations without losing error bounds.

A numerical prediction may be produced for diagnostics when one of these certificates is unavailable, but it must be marked `certified=False`. No unavailable proof is replaced by an empirical safety factor.
