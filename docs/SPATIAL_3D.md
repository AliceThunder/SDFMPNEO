# Shared 3-D electromagnetic–thermal spatial core

The executable core constructs electromagnetic and thermal operators from one orthogonal rectilinear 3-D material grid. Conductor, package, seawater, and passive regions are represented as spatial cell fields rather than hand-written small matrices.

## 1. Exact discrete topology

`RectilinearComplex3D` creates oriented nodes, edges, and faces. The edge-to-node gradient incidence matrix `G_full` and face-to-edge curl incidence matrix `C` satisfy

```text
C @ G_full = 0
```

exactly in sparse integer arithmetic.

## 2. Deterministic tree-cotree magnetic gauge

A deterministic spanning tree is generated from the ordered edge graph. Tree-edge vector-potential coordinates are fixed to zero and cotree edges are retained,

```text
A = R_A alpha.
```

For a connected graph,

```text
#tree   = #nodes - 1,
#cotree = #edges - #nodes + 1.
```

No SVD rank threshold, gauge penalty, or iterative gauge solve is used. For each connected conducting component, one scalar-potential reference node is removed deterministically, so `G_c` has no constant-potential nullspace.

## 3. Orthogonal material Hodge matrices

For an x-directed primal edge of length `dx`,

```text
M_sigma,e = (1/dx) sum_adjacent_cells sigma_c (dy_c/2)(dz_c/2).
```

For an x-normal primal face of area `dy dz`,

```text
M_nu,f = (1/(dy dz)) sum_adjacent_cells nu_c (dx_c/2).
```

The y/z formulas are analogous. Discontinuous conductor/package/seawater coefficients remain spatially distinct through their primal-dual cell contributions.

## 4. Spatial reciprocal A-psi assembly

The scaled scalar coordinate is

```text
psi = phi/(j omega),
```

so

```text
E = -j omega (R_A alpha + G_c psi).
```

The affine verification path `build_compatible_aphi_from_cells(...)` constructs

```text
C, G_c, R_A, M_nu, M_sigma,0, M_sigma,k
```

from cell fields and sends them to the reciprocal gauge-eliminated field core. The function/class names retain `aphi` for API compatibility; the internal coordinate convention is `A-psi`.

`NonlinearSpatialAphiProblem` likewise retains its legacy class name but uses the same reciprocal `A-psi` coordinates and does not require affine conductivity. It reconstructs

```text
T(a) = T_ref + sum_k a_k Phi_k
```

and evaluates analytic material laws directly before assembling `M_sigma(T(a))`.

## 5. Physical electromagnetic Riesz metric

The spatial field residual uses

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2 omega)) L_E^H M_sigma,ref L_E,
```

where `L_E=-j omega [R_A,G_c]`. This is magnetic energy plus Joule energy dissipated per electrical radian.

If

```text
H_em = L L^H,
```

then

```text
||r||_(H^-1) = ||L^-1 r||_2,
H^-1 r       = L^-H L^-1 r.
```

Riesz lifting and basis orthogonalization are therefore performed in physical energy coordinates without explicitly inverting `H_em`.

## 6. Regional losses and multiport outputs

For any cell mask,

```text
P_region = 1/2 E^H M_sigma,region E.
```

Disjoint copper and seawater masks produce `P_Cu` and `P_sea` from the same field state.

For divergence-free one-ampere impressed-current ports collected in `B`,

```text
A X = B,
Psi = B^T X,
Z = j omega Psi,
R = Re(Z),
L = Im(Z)/omega.
```

The reciprocal `A-psi` field matrix makes `Z^T=Z` structural for reciprocal materials. The code also checks passivity and port-power/Joule-power consistency. Joint multi-RHS reduction grows one field basis over all declared port excitations.

## 7. Shared thermal finite-volume grid

`RectilinearThermalFV3D` uses the same x/y/z cell geometry. The mass matrix is

```text
M_T,c = (rho c_p)_c V_c.
```

Across an internal material interface,

```text
G_f = A_f / (d_L/(2 k_L) + d_R/(2 k_R)).
```

This enforces heat-flux continuity through the two physical half-cell thermal resistances. The current outer-boundary implementation supports exact half-cell homogeneous Dirichlet resistance or leaves outer treatment external.

## 8. Thermal spectrum and retained-rank certificate

The verification core solves

```text
K_T Phi = M_T Phi Lambda.
```

For retained rank `r`, first omitted eigenvalue `lambda_(r+1)`, omitted initial M-norm `E0`, and certified source dual bound `Q`, the spatial projection tail satisfies

```text
||T_tail(t)||_M
<= exp(-lambda_(r+1)t) E0
 + (1-exp(-lambda_(r+1)t)) Q/lambda_(r+1).
```

The smallest rank satisfying the requested state/output tolerance can therefore be selected without a fixed empirical mode count. The present verification implementation still computes the full spectrum before truncation.

## 9. Continuous-domain status

For affine thermal-state dependence, the electromagnetic reduced residual can now be certified over a continuous rectangular state domain by branch-and-bound. The proof uses reduced-operator singular-value lower bounds and explicit residual derivative bounds. A box can be `certified`, `violated`, or `indeterminate`; a computational budget cannot create a false certificate.

This continuous certificate has not yet been generalized to nonlinear constitutive, geometry, frequency, or source parameters.

## 10. Current scope and next obligations

Already implemented:

- real 3-D node-edge-face topology and exact `C G = 0`;
- tree-cotree magnetic gauge and conducting-component scalar gauge;
- conductor/package/seawater material Hodge assembly;
- reciprocal `A-psi` field coordinates;
- physical Riesz coordinates and snapshot-free single/multi-RHS reduction;
- direct nonlinear temperature-dependent conductivity;
- region-separated Joule diagnostics;
- reciprocal multiport outputs and residual-to-output bounds;
- shared heterogeneous thermal finite-volume assembly;
- thermal projection-tail rank certification;
- continuous affine thermal-state EM residual certification.

Remaining work:

1. curved/unstructured compatible meshes or rigorously mapped complexes for real coil/package boundaries;
2. sparse end-to-end storage and a high-contrast iterative/direct solver with verified linear-solve error;
3. validated seawater constitutive laws and uncertainty propagation;
4. continuous certification for nonlinear constitutive, geometry, frequency, and source parameters;
5. scalable low-spectrum extraction with a verified first-omitted eigenvalue lower bound;
6. certified electromagnetic and thermal outer/open seawater-domain treatment;
7. terminal-current constrained solid-conductor ports when required by the physical excitation model.

The rectilinear core is a real spatial correctness reference, not the final curved-geometry engine.
