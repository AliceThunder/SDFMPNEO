# Executable core status

Run:

```bash
python -m pip install -e .
python examples/minimal_core.py
python examples/spatial_core.py
pytest -q
```

## Implemented core

The repository now contains an executable end-to-end mathematical core with:

- deterministic thermal generalized-eigenvalue reduction;
- a shared orthogonal 3-D material grid for electromagnetic and thermal physics;
- exact node-edge-face incidence construction with `C @ G = 0`;
- deterministic tree-cotree magnetic-vector-potential gauge elimination;
- conducting-component scalar-potential gauge elimination with no penalty parameter;
- cellwise reluctivity/conductivity Hodge assembly for conductor/package/seawater discontinuities;
- compatible magnetoquasistatic `A-phi` assembly;
- physical electromagnetic Riesz metric from magnetic energy plus Joule energy dissipated per electrical radian;
- Cholesky Riesz coordinates for residual norms, Riesz lifts, and H-orthogonal basis enrichment;
- snapshot-free electromagnetic residual-Riesz basis construction;
- exact nonlinear temperature-dependent conductivity through analytic material laws;
- exact copper conductivity induced by a linear resistivity law, without affine conductivity linearization;
- reduced thermal heat-source projection and analytic heat-source Jacobian;
- region-separated Joule diagnostics from the same field solution (`copper`, `seawater`, or arbitrary cell masks);
- heterogeneous 3-D thermal finite-volume mass/conduction assembly using two-half-cell interface resistance;
- sparse-to-spectral thermal interface for verification meshes;
- sparse analytic polynomial-exponential neural algebra;
- exact first-order response compilation back into the same analytic algebra;
- arbitrary-depth chained analytic-response DAGs;
- exact analytic time derivatives of the complete compiled graph;
- vectorized final analytic kernel with unique basis-term sharing;
- deterministic product-candidate dictionary generation;
- tangent physical-residual scoring using the exact reduced electromagnetic heat-source Jacobian;
- closed-form first-order optimal candidate weight and predicted residual reduction;
- full nonlinear residual re-evaluation before a growth proposal can be accepted;
- contraction and uniform-in-time state-error certificate interfaces.

The code has no geometry-specific impedance-solver dependency and uses no FEM/Maxwell solution snapshots or transient solution labels.

## Current physical chain

```text
3-D cell geometry/material fields
        |
        +--> thermal FV: M_T, K_T
        |        |
        |        +--> thermal spectrum Phi_T, Lambda_T
        |                    |
        |                    +--> T(a) on cells
        |                    +--> analytic sigma_Cu(T), sigma_sea(T)
        |
        +--> exact topology: G_full, C
                 |
                 +--> tree-cotree R_A + conducting G_c
                 |
                 +--> M_nu, M_sigma(T(a))
                         |
                         +--> compatible A-phi system
                                   |
                                   +--> snapshot-free EM reduction
                                   |
                                   +--> q_em,r(a), d q_em,r / d a
                                   +--> P_Cu, P_sea diagnostics
                                             |
                                             +--> analytic neural residual/growth
```

## Compatible electromagnetic core

The gauge-eliminated block is

```text
[ R_A^H C^H M_nu C R_A + j*w R_A^H M_sigma R_A,   R_A^H M_sigma G_c ]
[ j*w G_c^H M_sigma R_A,                            G_c^H M_sigma G_c   ].
```

The electric field is

```text
E = [-j*w R_A, -G_c] [alpha; phi].
```

For the 3-D spatial builder, the residual Riesz metric is

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2*w)) L_E^H M_sigma,ref L_E.
```

With `H_em = L L^H`, Riesz calculations are performed as

```text
||r||_(H^-1) = ||L^-1 r||_2,
H^-1 r       = L^-H L^-1 r.
```

The implementation therefore does not explicitly invert `H` or repeatedly solve a full Hermitian Riesz system during greedy enrichment.

See [`COMPATIBLE_EM.md`](COMPATIBLE_EM.md), [`SPATIAL_3D.md`](SPATIAL_3D.md), and [`NONLINEAR_CONSTITUTIVE.md`](NONLINEAR_CONSTITUTIVE.md).

## Tree-cotree gauge

A deterministic spanning tree is built from the ordered edge graph. Magnetic-vector-potential values on tree edges are fixed to zero; cotree edges become independent coordinates. For a connected graph,

```text
#tree   = #nodes - 1,
#cotree = #edges - #nodes + 1.
```

This removes the previous dense `ker(G^T)` SVD from the default spatial implementation. The current verification assembler still stores the cotree selector densely when forming small matrices; sparse end-to-end storage remains a scaling task.

## Exact nonlinear conductivity path

`NonlinearSpatialAphiProblem` reconstructs

```text
T(a) = T_ref + sum_k a_k Phi_k
```

on cells and evaluates analytic material laws directly.

For copper with

```text
rho(T) = rho_ref [1 + alpha(T-T_ref)],
```

the implementation uses

```text
sigma(T) = sigma_ref / [1 + alpha(T-T_ref)]
```

and its exact derivative. The electromagnetic operator derivative and final reduced heat-source Jacobian therefore include the nonlinear material dependence without an affine conductivity approximation.

## Region-separated losses

The thermal equation continues to use the unified spatially projected source. Diagnostics additionally evaluate

```text
P_region(a) = 1/2 E^H M_sigma,region(a) E
```

for disjoint copper/seawater masks using the same field solution.

## Analytic compiler closure

Every compiled node is a finite sparse sum

```text
sum_k c_k t^(m_k) exp(-(n_k dot lambda) t).
```

For

```text
(d/dt + lambda_i) h = source, h(0)=0,
```

the response is compiled back into the same algebra. Response nodes can therefore be parents of later response nodes at arbitrary depth.

After compilation,

```text
a(t)  = C phi(t),
da(t) = C dphi(t),
```

where each unique analytic basis term is evaluated once.

## Residual-driven growth

For a unit candidate response in target mode `i`,

```text
(d/dt + lambda_i) h = psi,
```

and electromagnetic heat-source Jacobian `J_g`, the first residual variation is

```text
D = e_i psi - J_g[:,i] h.
```

For a deterministic quadrature rule,

```text
w*    = -<R,D>/<D,D>,
Delta =  <R,D>^2/<D,D>.
```

The complete nonlinear physical residual is recomputed before accepting the enriched graph.

## Deliberate current limits

1. **Curved geometry:** the real spatial core is currently an orthogonal rectilinear complex. Curved round/rounded-square coils and package boundaries still require an unstructured compatible mesh or structure-preserving mapped complex.
2. **Sparse end-to-end gauge algebra:** tree-cotree gauge construction is now topological and linear-time in the edge graph, but the verification assembler still densifies `R_A`, `C`, and Hodge blocks before the small-mesh solve.
3. **Material-law validation:** nonlinear analytic conductivity is now supported directly. Production still requires the physically validated seawater law, admissible temperature ranges, and uncertainty/error propagation for material constants.
4. **High material contrast:** physical Riesz coordinates are implemented, but a production sparse high-contrast `A-phi` linear solver/preconditioner with verified linear-solve error is still required for realistic copper/seawater ratios on large meshes.
5. **Electromagnetic domain certification:** greedy verification is still performed on a deterministic finite candidate set, not the continuous geometry/temperature parameter domain.
6. **Thermal rank certification:** the thermal spatial operator is real and sparse, but the current spectral core retains the full spectrum on verification meshes. A rigorous spectral-tail/output-error estimator is still required before scalable rank truncation is enabled.
7. **Outer thermal domain:** the current thermal core provides exact homogeneous Dirichlet half-cell boundary resistance or leaves outer treatment external. The physical open/seawater-domain boundary model and domain truncation must be independently certified.
8. **Floating-point analytic resonance:** exact resonance is handled analytically. Very near non-exact resonances still require an error-controlled evaluation/precision certificate.
9. **Global growth convergence:** the closed-form candidate weight is exact for the tangent residual model and full residual re-evaluation provides a safeguard, but the globally convergent certified growth strategy remains to be completed.

These limits are part of the implementation contract. The executable core verifies the physical and analytic structure without claiming that the final curved-geometry, high-contrast, globally certified underwater WPT solver is already complete.
