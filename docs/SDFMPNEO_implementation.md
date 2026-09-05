# SDF-MPNEO implementation specification

This document translates `docs/SDFMPNEO_theory.tex` into an executable software architecture. The implementation must preserve the following invariants:

- no geometry-specific impedance solver in the core path;
- no full-order solution snapshots for electromagnetic or thermal reduced-basis construction;
- no transient solution labels for analytic-network training;
- no empirical fixed reduced ranks or neural width/depth;
- every approximation that affects outputs must expose an error estimator/remainder bound.

## 1. Package layout

```text
sdfmpneo/
├── geometry/
│   ├── parameter_domain.py
│   ├── reference_map.py
│   ├── topology_chart.py
│   └── bounds.py
│
├── physics/
│   ├── em/
│   │   ├── spaces.py
│   │   ├── full_operator.py
│   │   ├── excitation.py
│   │   ├── constitutive.py
│   │   ├── riesz.py
│   │   ├── infsup.py
│   │   ├── reduced_basis.py
│   │   ├── reduced_operator.py
│   │   ├── operator_separation.py
│   │   ├── loss_tensors.py
│   │   └── outputs.py
│   │
│   ├── thermal/
│   │   ├── full_operator.py
│   │   ├── spectral_basis.py
│   │   ├── canonical_coordinates.py
│   │   ├── reduced_operator.py
│   │   └── error_estimator.py
│   │
│   └── coupling/
│       ├── electrothermal.py
│       └── jacobian.py
│
├── analytic/
│   ├── atom.py
│   ├── series.py
│   ├── phi_functions.py
│   ├── neuron.py
│   ├── graph.py
│   ├── growth.py
│   └── compiler.py
│
├── training/
│   ├── residual.py
│   ├── verified_quadrature.py
│   └── optimizer.py
│
├── certification/
│   ├── em_state.py
│   ├── operator_remainder.py
│   ├── thermal_state.py
│   ├── contraction.py
│   ├── output_bound.py
│   └── certificate.py
│
├── offline/
│   └── build_model.py
│
├── online/
│   └── evaluate.py
│
└── model.py
```

High-order sparse assembly may initially be implemented in Python/SciPy. Performance-critical electromagnetic assembly and reduced tensor contractions should later move to C++ with `pybind11`.

---

## 2. Full electromagnetic operator

The core electromagnetic state is the compatible discrete magnetoquasistatic state

```text
u = [A_edge, phi_node, gauge/constraint variables]
```

with

```text
A_em(eta) u = b_em(mu)
eta = (geometry, operating condition, thermal coordinates)
```

### Required interface

```python
class FullEMOperator:
    def assemble(self, mu, a_T): ...
    def apply(self, mu, a_T, u): ...
    def apply_adjoint(self, mu, a_T, v): ...
    def residual(self, mu, a_T, u): ...
    def port_functionals(self, mu): ...
```

The first implementation may assemble a sparse matrix. The later high-performance version should provide matrix-free `apply` and separated reduced assembly.

Copper and seawater conductivity are supplied only through physical constitutive laws:

```python
sigma_cu = copper_conductivity(T)
sigma_sea = seawater_conductivity(T)
```

No added AC-resistance correction belongs in the core electromagnetic model. Skin and proximity effects are captured by the volumetric field solution.

---

## 3. Reference-domain geometry

### Required map

```python
class ReferenceMap:
    def x(self, xhat, geometry): ...
    def jacobian(self, xhat, geometry): ...
    def det_jacobian(self, xhat, geometry): ...
    def inverse_jacobian(self, xhat, geometry): ...
```

The geometry module must expose verified bounds

```text
J_min > 0
J_max < infinity
||DF|| <= C_F
||DF^-1|| <= C_F
```

for every certified geometry chart.

Electromagnetic edge fields use the covariant Piola map. Thermal scalar fields use the standard scalar pullback.

A geometry whose map loses non-degeneracy or changes topology is not extrapolated. It creates a new topology chart/model.

---

## 4. Thermal spectral reduction

Solve

```text
K_T phi_i = lambda_i M_T phi_i
```

with mass orthonormality.

### Required interface

```python
@dataclass
class ThermalReducedModel:
    Phi: Array
    lambdas: Array
    T_boundary: Array
    certificate: ThermalCertificate

    def project_initial(self, T0): ...
    def evaluate_temperature(self, a, points=None): ...
```

The reduced rank is not a configuration constant. The builder enlarges the retained spectral cluster until the propagated output-error target is satisfied.

Eigenvalue crossings are handled at the spectral-subspace level. Canonical coordinates are constructed from projectors/anchor charts, not by sorting eigenvectors and matching signs.

---

## 5. Solution-data-free electromagnetic basis

This is the central replacement for snapshot-based reduced-basis construction.

### 5.1 Reference Riesz map

Choose one fixed Hermitian positive-definite matrix/operator `H` on the gauge-stabilized electromagnetic state space.

```python
class EMRieszMap:
    def solve(self, r):        # H w = r
        ...
    def dual_norm(self, r):    # sqrt(r^* H^-1 r)
        ...
```

The factorization/preconditioner of `H` is reused for every enrichment.

### 5.2 Initial basis

If

```text
b(mu) = sum_q theta_q(mu) b_q + e_b
```

initialize with

```text
H^-1 b_q
```

plus required port/gauge/constraint-compatible modes.

No electromagnetic field solution snapshot is used.

### 5.3 Minimum-residual reduced solve

```python
class EMReducedBasis:
    V: Array  # H-orthonormal

    def solve_minres(self, operator, mu, a_T): ...
    def residual(self, operator, mu, a_T, c): ...
```

The reduced coefficient solves

```text
c = argmin ||b - A V c||_(H^-1).
```

This guarantees that enlarging `V` cannot increase the residual.

### 5.4 Certified state estimator

```text
Delta_em(eta)
 = (||r||_(H^-1) + eps_op ||u_r||_H + eps_b)
   / beta_LB(eta)
```

`beta_LB` is a verified lower bound for the scaled electromagnetic operator inf-sup constant.

### 5.5 Worst-case enrichment

```python
while worst_output_bound > target:
    eta_star = certified_global_max(error_estimator, domain)
    c = reduced.solve_minres(..., eta_star)
    r = full_operator.residual(..., V @ c)
    w = riesz.solve(r)
    V = H_orthonormal_append(V, w)
```

The global maximization must return certified upper/lower bounds. Initial implementation can use deterministic interval branch-and-bound over the bounded parameter domain.

### Why this solves temperature variation

The search domain is

```text
eta = (geometry, frequency/load/excitation, a_T)
```

and therefore includes the full certified temperature family. The electromagnetic basis is accepted only after the supremum residual over this domain satisfies the requested output error. No assumption of temperature-invariant basis quality is made.

---

## 6. Inf-sup lower bound

The scaled operator is

```text
S(eta) = H^-1/2 A_em(eta) H^-1/2.
```

The exact stability factor is its minimum singular value. The implementation needs a verified lower bound:

```python
class EMInfSupCertificate:
    def lower_bound(self, parameter_box): ...
```

Use the certified separated operator representation and interval/spectral perturbation bounds to enclose `sigma_min(S)` over each parameter box. Branch boxes until the lower bound is strictly positive or the box is declared outside the certified domain.

No pointwise random sampling is sufficient for this certificate.

---

## 7. Certified operator separation

Online electromagnetic assembly must not traverse the full mesh.

Target representation:

```text
A_em(eta) = sum_q theta_q(eta) A_q + E_A(eta)
||E_A|| <= eps_op
```

### 7.1 Constitutive laws

For each certified temperature interval use:

- exact algebraic representation when possible;
- otherwise Chebyshev/polynomial approximation with verified remainder;
- for certified tabulated material data, interval spline/polynomial interpolation with derivative-based error enclosure.

```python
@dataclass
class ConstitutiveExpansion:
    coefficients: Array
    remainder_bound: float
```

The polynomial order grows until the contribution of the remainder to the final engineering-output bound is acceptable.

### 7.2 Geometry coefficients

Reference-domain metric/Jacobian terms use exact parameter separation when available. Otherwise use deterministic multivariate polynomial/Chebyshev approximation with verified remainder bounds.

### 7.3 Reduced projection

Offline precompute

```text
A_q,r = V^* A_q V.
```

Online:

```python
A_r = sum(theta_q(eta) * A_q_r for q in terms)
```

No full-order matrix is assembled online.

### 7.4 Tensor compression

If polynomial temperature dependence creates high-order coefficient tensors, use deterministic SVD/hierarchical factorization. Discarded terms are allowed only when their aggregate norm is rigorously bounded and added to `eps_op`.

---

## 8. Reduced electromagnetic online solve

```python
class ReducedEMOperator:
    def assemble(self, mu, a_T): ...
    def solve(self, mu, a_T): ...
```

At each requested thermal state:

```text
A_em,r(mu,a_T) c = b_em,r(mu).
```

For multiple ports/right-hand sides, factor `A_em,r` once and reuse the factorization.

The reduced rank `r_em` is whatever dimension the certificate requires; it is not hard coded.

---

## 9. Direct reduced copper/seawater loss tensors

Do not reconstruct the full three-dimensional Joule-loss field online.

For each thermal mode `k`, precompute separated reduced quadratic forms

```text
H_Cu,k(eta)
H_sea,k(eta)
```

so that

```text
q_em,r[k] = c^H (H_Cu,k + H_sea,k) c.
```

Total loss forms are

```text
P_Cu  = c^H H_Cu,P  c
P_sea = c^H H_sea,P c.
```

This preserves the different spatial locations of copper and seawater heating while avoiding full field reconstruction.

### Required interface

```python
@dataclass
class EMOutputs:
    Z: Array
    R: Array
    L: Array
    M: Array
    P_cu: float
    P_sea: float
    q_thermal: Array
    certificate: EMCertificate
```

---

## 10. Closed electrothermal operator

```python
class ElectroThermalClosure:
    def g(self, mu, a_T):
        c = em.solve(mu, a_T)
        q = losses.project(mu, a_T, c)
        return thermal.mass_inverse(q)

    def jacobian(self, mu, a_T):
        # differentiate the small reduced EM system and loss tensors
        ...
```

The thermal reduced dynamics are

```text
da/dt + Lambda_T a = g_em(a;mu) + f_T(mu).
```

The Jacobian is computed by implicit differentiation of the reduced electromagnetic system:

```text
A_r dc/da_j = -(dA_r/da_j)c + db_r/da_j.
```

No finite-difference temperature perturbation is required.

---

## 11. Analytic neural graph

The network represents the arbitrary-time solution map of the already closed reduced dynamics.

### Analytic atom

```python
@dataclass(frozen=True)
class AtomKey:
    power: int
    decay_multiindex: tuple[int, ...]
```

An analytic series represents

```text
sum_k coeff_k * t^m_k * exp(-(n_k dot lambda) t).
```

### Required exact operations

```python
series.add(...)
series.multiply(...)
series.time_derivative(...)
response_operator(lambda_i, source_series, initial_value)
```

The response operator uses stable entire `phi` functions so resonant cases never depend on an empirical `abs(lambda-rho) < tol` switch.

### Analytic neuron

```python
class AnalyticNeuron:
    target_mode: int
    parents: tuple[int, ...]
    weight: Parameter
    initial_rule: ...
```

Each neuron represents

```text
(d/dt + lambda_i) h = S(parents, static parameters).
```

The network output and its time derivative are both exact recursive analytic evaluations.

---

## 12. Residual-grown neural topology

```python
R = da_dt + Lambda_T * a - closure.g(mu, a) - f_T
```

Candidate neurons are generated hierarchically from products of active analytic nodes and static parameter/initial-state nodes.

Select the candidate that maximizes the certified Riesz residual correlation, append it, re-optimize all active coefficients, and repeat only while the propagated output certificate remains above target.

There is no configured `num_layers` or `hidden_width` in the scientific method.

---

## 13. Unified certificate object

```python
@dataclass
class Certificate:
    geometry_valid: bool
    material_valid: bool
    em_infsup_lb: float
    em_state_error: float
    operator_remainder: float
    thermal_rom_error: float
    neural_residual_error: float
    reduced_linear_solve_error: float
    contraction_lb: float | None
    output_bounds: dict[str, float]
    long_time_certified: bool
```

If

```text
kappa = lambda_min(Lambda_T)
        - sup lambda_max(sym(dg_em/da))
```

has a verified positive lower bound, use the uniform-in-time estimate

```text
||e_T(t)|| <= eta_total / kappa.
```

Otherwise return a finite-time bound only and set `long_time_certified=False`.

---

## 14. Offline compiler

```python
def build_model(config):
    geometry = build_reference_charts(config.geometry_domain)
    thermal = build_certified_thermal_rom(geometry, config.output_targets)

    em_sep = build_certified_operator_separation(
        geometry=geometry,
        thermal_domain=thermal.allowed_coordinate_domain,
        material_laws=config.material_laws,
        output_targets=config.output_targets,
    )

    em_basis = build_em_basis_by_worst_residual(
        separated_operator=em_sep,
        parameter_domain=(config.mu_domain, thermal.allowed_coordinate_domain),
        output_targets=config.output_targets,
    )

    em_rom = project_em_operators_and_loss_tensors(em_sep, em_basis, thermal)
    closure = ElectroThermalClosure(em_rom, thermal)

    analytic_graph = build_base_analytic_graph(thermal)
    analytic_graph = grow_by_physical_residual(
        analytic_graph, closure, config.output_targets
    )

    certificate = certify_complete_model(
        geometry, thermal, em_sep, em_rom, closure, analytic_graph
    )

    return compile_deployment_model(...)
```

---

## 15. Online evaluator

```python
def evaluate(model, geometry, operating, T0, t):
    chart = model.geometry.require_valid_chart(geometry)
    a0 = model.thermal.project_initial(T0, chart)

    a, da = model.analytic.evaluate(geometry, operating, a0, t)
    em = model.em.evaluate(geometry, operating, a)
    cert = model.certification.evaluate(geometry, operating, a, da, em)

    return Prediction(
        thermal_coeff=a,
        Tmax=model.thermal.max_temperature_bound(a, cert),
        Z=em.Z,
        R=em.R,
        L=em.L,
        M=em.M,
        P_cu=em.P_cu,
        P_sea=em.P_sea,
        certificate=cert,
    )
```

No loop over time steps appears in the online evaluator.

---

## 16. Development order

### Stage A — mathematical unit tests

Implement small manufactured magnetoquasistatic matrices and thermal systems. Verify:

- minimum-residual monotonicity;
- Riesz enrichment reduces the worst residual;
- operator-separation remainder enters the state estimator correctly;
- analytic atom multiplication/derivative/response closure;
- resonant `phi`-function evaluation;
- contractivity/error-bound identities.

### Stage B — fixed geometry full field

Implement one fixed coil/package/seawater geometry with compatible edge/nodal discretization. Verify copper skin/proximity and seawater induced currents against a high-order full field solve used only for validation.

### Stage C — deterministic EM reduction

Construct `V_em` exclusively from excitation/constraint Riesz initialization plus worst-residual enrichments. Compare reduced fields/impedance/losses with the full field solver only after the reduced space has been built.

### Stage D — temperature dependence

Enable `sigma_Cu(T)` and `sigma_sea(T)`, certify the constitutive expansion, and enlarge the electromagnetic parameter domain to thermal coordinates `a_T`.

### Stage E — coupled heat-source tensors

Implement direct reduced copper/seawater heat-source projections and verify energy consistency:

```text
sum physical Joule power = corresponding integrated thermal source
```

up to the certified quadrature/operator remainder.

### Stage F — analytic evolution network

Train/grow the analytic graph from the closed reduced physical residual. Use full transient simulation only as an external validation benchmark.

### Stage G — parameterized geometry

Enable reference-domain geometry charts and certify the same electromagnetic/thermal machinery over the declared geometry family.

---

## 17. Acceptance criteria

A model build is valid only if all of the following are available:

1. geometry non-degeneracy certificate;
2. thermal reduced-space error estimator;
3. electromagnetic inf-sup lower bound;
4. supremum electromagnetic residual bound over `(mu, a_T)`;
5. constitutive/geometry operator-separation remainder;
6. reduced linear-solve residual;
7. analytic-network physical residual bound;
8. contractivity or finite-time stability bound;
9. propagated bounds for the requested engineering outputs.

If any required bound is unavailable, the implementation may still return a numerical prediction for development diagnostics, but it must mark the result `certified=False` and must not present it as a certified SDF-MPNEO result.
