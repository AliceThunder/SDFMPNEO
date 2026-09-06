# Executable core status

Run:

```bash
python -m pip install -e '.[dev]'
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q
```

The repository also runs the full `pytest` suite in GitHub Actions on pushes and pull requests to `main`.

## Implemented core

The executable chain now contains:

- shared orthogonal 3-D electromagnetic/thermal material cells;
- exact node-edge-face incidence with `C @ G = 0`;
- deterministic tree-cotree magnetic gauge and conducting-component scalar gauge;
- conductor/package/seawater reluctivity/conductivity Hodge assembly;
- reciprocal scaled `A-psi` magnetoquasistatic formulation with `psi=phi/(j omega)`;
- physical electromagnetic Riesz metric from magnetic energy plus Joule energy per electrical radian;
- Cholesky-Riesz residual norms, lifts, and basis enrichment;
- snapshot-free single- and multi-RHS electromagnetic reduction;
- exact nonlinear cell conductivity through analytic material laws and exact operator derivatives;
- reciprocal copper conductivity induced by a linear copper resistivity law without conductivity linearization;
- reduced electromagnetic heat-source projection and exact heat-source Jacobian;
- copper/seawater regional Joule loss diagnostics from the same field solution;
- field-derived multiport `Z`, `R`, `L`, and `M` outputs;
- executable reciprocity, passivity, and port-power/Joule-power consistency checks;
- residual-to-`Z/R/L/M` output error bounds using an electromagnetic stability constant;
- continuous thermal-state box residual certification for affine electromagnetic operators by branch-and-bound;
- heterogeneous 3-D thermal finite-volume mass/conduction assembly;
- thermal generalized eigenmodes and a spectral-tail rank certificate derived from initial-tail energy, a source dual bound, and requested output/state accuracy;
- analytic polynomial-exponential neural algebra with arbitrary-depth response DAGs;
- compressed fast analytic evaluation using unique basis functions;
- an independent exact state-space analytic realization backend that handles exact and near resonance without a closeness threshold;
- deterministic product-candidate generation and residual-driven neuron scoring;
- closed-form tangent-optimal candidate weight plus full nonlinear residual re-evaluation;
- contraction and uniform-in-time coupled state-error certificate interfaces;
- a unified fixed-operating-condition online query interface returning thermal state, temperature, heat source, physical residual, multiport impedance, and optional regional losses.

The code has no geometry-specific impedance-solver dependency and uses no FEM/Maxwell solution snapshots or transient solution labels.

## Current physical chain

```text
3-D cell geometry/materials
        |
        +--> thermal FV: M_T, K_T
        |        |
        |        +--> thermal spectrum Phi_T, Lambda_T
        |        +--> certified retained rank r
        |                    |
        |                    +--> T(a) on cells
        |                    +--> sigma_Cu(T), sigma_sea(T), ...
        |
        +--> exact topology: G_full, C
                 |
                 +--> tree-cotree R_A + conducting G_c
                 |
                 +--> M_nu, M_sigma(T(a))
                         |
                         +--> reciprocal A-psi field system
                                   |
                                   +--> multi-RHS snapshot-free EM reduction
                                   |
                                   +--> q_em,r(a), d q_em,r/da
                                   +--> P_Cu, P_sea
                                   +--> Z/R/L/M + output certificates
                                             |
                                             +--> analytic neural residual/growth
                                             +--> arbitrary-time online query
```

## Reciprocal electromagnetic core

The scalar-potential coordinate is

```text
psi = phi/(j omega).
```

The gauge-eliminated field system is

```text
[ K_A + j*w R_A^H M_sigma R_A,  j*w R_A^H M_sigma G_c ] [alpha]   [R_A^H J_s]
[ j*w G_c^H M_sigma R_A,         j*w G_c^H M_sigma G_c ] [ psi ] = [    0     ],
```

where

```text
K_A = R_A^H C^H M_nu C R_A,
E   = -j*w (R_A alpha + G_c psi).
```

For reciprocal real Hodge operators this matrix is complex symmetric. Multiport reciprocity therefore follows from the field structure rather than post-processing.

The Riesz metric is

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2*w)) L_E^H M_sigma,ref L_E.
```

For `H_em=L L^H`,

```text
||r||_(H^-1) = ||L^-1 r||_2,
H^-1 r       = L^-H L^-1 r.
```

## Multiport outputs

For one-ampere divergence-free impressed-current port cochains collected in `B`,

```text
A(a) X = B,
Psi = B^T X,
Z = j*w*Psi,
R = Re(Z),
L = Im(Z)/w.
```

Off-diagonal `L` entries are mutual inductances for this port definition. The code verifies

```text
Z^T = Z,
R >= 0,
1/2 Re(I^H Z I) = 1/2 E^H M_sigma E.
```

One reduced space is grown over all declared thermal states and port RHS columns. With electromagnetic stability

```text
beta_em = sigma_min(L^-1 A L^-H),
```

entrywise output errors obey

```text
|Delta Z_ij|
<= w ||b_i||_(H^-1) ||r_j||_(H^-1) / beta_em,
```

with corresponding `R` and `L/M` bounds.

## Exact nonlinear conductivity path

`NonlinearSpatialAphiProblem` (legacy class name retained for API compatibility) now uses the reciprocal `A-psi` coordinates internally. It reconstructs

```text
T(a) = T_ref + sum_k a_k Phi_k
```

and evaluates analytic material laws directly.

For linear copper resistivity,

```text
rho(T) = rho_ref [1 + alpha(T-T_ref)],
sigma(T) = sigma_ref / [1 + alpha(T-T_ref)],
```

with exact derivatives propagated through `dA/da` and `dq_em,r/da`.

## Thermal rank certificate

For first omitted eigenvalue `lambda_(r+1)`, omitted initial M-norm `E0`, and a certified source bound

```text
||q(t)||_(M^-1) <= Q,
```

the projection tail satisfies

```text
||T_tail(t)||_M
<= exp(-lambda_(r+1)t) E0
 + (1-exp(-lambda_(r+1)t)) Q/lambda_(r+1).
```

The current verification implementation computes the complete spectrum and chooses the smallest `r` satisfying the requested state or output tolerance. Coupling error due to evaluating the nonlinear source on the reduced state is handled separately by the physical residual/contraction certificate.

## Continuous affine electromagnetic-domain certificate

For strictly affine thermal-state dependence, a parameter box is bounded using the center residual, a lower bound on the reduced operator singular value, and explicit residual derivative bounds. Branch-and-bound returns exactly one of:

```text
certified
violated
indeterminate
```

Exhausting a computational work budget produces `indeterminate`; it never produces a certificate.

This is currently a continuous certificate for affine thermal-state boxes, not yet for nonlinear constitutive, geometry, or frequency parameters.

## Analytic network and resonance-stable realization

The fast compiler represents nodes as finite sums

```text
sum_k c_k t^(m_k) exp(-rho_k t).
```

The same DAG can independently compile to finite-dimensional exact realizations

```text
h(t) = c^T exp(A t) b.
```

Addition uses block diagonals, products use Kronecker sums, and a response neuron adds one linear state. Repeated or nearly repeated decay rates therefore become ordinary confluent/Jordan structure and require no empirical near-resonance threshold.

The realization backend serves as a stable correctness/reference path for the much faster canonical polynomial-exponential kernel.

## Residual-driven growth

For a unit candidate response in target mode `i`,

```text
(d/dt + lambda_i) h = psi,
```

with heat-source Jacobian `J_g`, the first residual variation is

```text
D = e_i psi - J_g[:,i] h.
```

For a supplied deterministic quadrature rule,

```text
w*    = -<R,D>/<D,D>,
Delta =  <R,D>^2/<D,D>.
```

The complete nonlinear residual is then recomputed before accepting the candidate.

## Deliberate current limits

1. **Curved geometry:** the spatial correctness core is rectilinear. Real round/rounded-square coil and package boundaries require an unstructured compatible mesh or rigorously structure-preserving mapped complex.
2. **Sparse large-scale field solution:** tree-cotree construction is scalable, but the current verification assemblers still densify major matrices. A production sparse high-contrast solver/preconditioner with a verified linear-solve error bound is required.
3. **Material validation:** the nonlinear constitutive interface is implemented; production still requires physically validated seawater laws, admissible ranges, and uncertainty propagation for material constants.
4. **General continuous-domain certification:** continuous affine thermal-state certification exists. Nonlinear constitutive, geometry, frequency, and source domains still need verified interval/remainder bounds.
5. **Scalable thermal eigenspectrum:** the rank certificate exists, but current verification obtains the full spectrum. Production needs low-mode extraction plus a verified lower bound on the first omitted eigenvalue.
6. **Outer domain:** physical open/seawater electromagnetic and thermal boundary treatment and any finite-domain truncation still require independent certification.
7. **Solid-conductor terminals:** the current multiport implementation is an impressed-current/stranded-source formulation. Terminal-current constrained solid-conductor ports are a separate formulation still to implement.
8. **Parameter-conditioned analytic network:** the executable online model currently represents one fixed operating condition. Geometry/operating condition `U` must become an explicit analytic-network parameter for one trained model to answer arbitrary conditions.
9. **Analytic realization scaling:** the resonance-stable realization is exact but Kronecker product dimensions can grow rapidly; certified realization compression/minimalization remains to be implemented.
10. **Global growth convergence:** the tangent-optimal update plus nonlinear residual safeguard is executable, but a globally convergent certified network-growth loop over the full parameter domain remains to be completed.

These limits are part of the implementation contract; the repository does not claim that the final curved-geometry, large-scale globally certified underwater WPT surrogate is already complete.
