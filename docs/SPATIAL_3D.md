# Shared 3-D electromagnetic–thermal spatial core

The executable core constructs electromagnetic and thermal operators from one orthogonal rectilinear 3-D material grid. Conductor, package, seawater, and passive regions are represented as spatial cell fields rather than hand-written small matrices.

## 1. Exact discrete topology

`RectilinearComplex3D` creates oriented nodes, edges, and faces. The edge-to-node gradient incidence matrix `G_full` and face-to-edge curl incidence matrix `C` satisfy

```text
C @ G_full = 0
```

exactly in sparse integer arithmetic.

## 2. Deterministic tree-cotree magnetic gauge

The default magnetic-vector-potential gauge no longer uses a dense numerical null-space factorization.

A deterministic spanning tree is generated from the ordered edge graph. Tree-edge vector-potential coordinates are fixed to zero and cotree edges are retained as the independent magnetic coordinates,

```text
A = R_A alpha,
R_A = cotree edge selector.
```

For a connected graph,

```text
#tree   = #nodes - 1,
#cotree = #edges - #nodes + 1.
```

The cotree coordinates together with one-reference-node gradient coordinates form a direct-sum coordinate system on edge space. Thus every gauge equivalence class has one tree-gauge representative, with no gauge penalty, SVD rank threshold, or iterative gauge solve.

For every connected conducting component, one scalar-potential reference node is also removed deterministically, so `G_c` has no constant-potential nullspace.

## 3. Orthogonal material Hodge matrices

Material fields are cellwise. For an x-directed primal edge of length `dx`,

```text
M_sigma,e = (1/dx) sum_adjacent_cells sigma_c (dy_c/2)(dz_c/2).
```

For an x-normal primal face of area `dy dz`,

```text
M_nu,f = (1/(dy dz)) sum_adjacent_cells nu_c (dx_c/2).
```

The y/z formulas are analogous. Discontinuous conductor/package/seawater coefficients are therefore retained through their separate primal-dual cell contributions rather than replaced by an arithmetic interface average.

## 4. Spatial A-phi assembly

The affine verification path `build_compatible_aphi_from_cells(...)` constructs

```text
C, G_c, R_A, M_nu, M_sigma,0, M_sigma,k
```

from cell fields and passes them into the compatible gauge-eliminated `A-phi` core.

The newer `NonlinearSpatialAphiProblem` does not require affine conductivity. It reconstructs

```text
T(a) = T_ref + sum_k a_k Phi_k
```

on cells and evaluates the material conductivity directly through analytic constitutive laws before assembling the Hodge operator.

The current nonlinear material library includes an exact reciprocal conductivity induced by a linear copper resistivity law, plus constant and physically specified affine conductivity laws. See [`NONLINEAR_CONSTITUTIVE.md`](NONLINEAR_CONSTITUTIVE.md).

## 5. Physical electromagnetic Riesz metric

The 3-D builder uses

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2 omega)) L_E^H M_sigma,ref L_E.
```

This is magnetic energy plus Joule energy dissipated per electrical radian. It contains no tunable balancing coefficient.

If

```text
H_em = L L^H,
```

then residual operations are performed in Cholesky Riesz coordinates,

```text
||r||_(H^-1) = ||L^-1 r||_2,
H^-1 r       = L^-H L^-1 r.
```

The greedy basis is orthogonalized in these normalized coordinates, avoiding explicit formation of `H^-1` and avoiding repeated ill-conditioned Hermitian solves.

## 6. Region-separated losses

`build_region_loss_projector(...)` builds

```text
P_region(a) = 1/2 E^H M_sigma,region(a) E
```

from arbitrary cell masks. Disjoint copper and seawater masks therefore yield `P_Cu` and `P_sea` from the same electromagnetic state used by the coupled thermal source.

No separate copper-resistance formula or seawater correction is introduced.

## 7. Shared thermal finite-volume grid

`RectilinearThermalFV3D` uses the same x/y/z cell geometry. The mass matrix is

```text
M_T,c = (rho c_p)_c V_c.
```

Across an internal material interface,

```text
G_f = A_f / (d_L/(2 k_L) + d_R/(2 k_R)).
```

This enforces finite-volume heat-flux continuity through the two physical half-cell thermal resistances.

The current homogeneous outer-boundary implementation supports exact half-cell Dirichlet resistance or leaves outer treatment external.

## 8. Current scope and next obligations

Already implemented:

- real 3-D node-edge-face topology;
- exact `C G = 0`;
- tree-cotree magnetic gauge;
- conducting-component scalar gauge;
- conductor/package/seawater Hodge assembly;
- physical Riesz coordinates;
- nonlinear temperature-dependent conductivity interface;
- region-separated Joule diagnostics;
- shared heterogeneous thermal finite-volume assembly.

Remaining work:

1. generalize from orthogonal rectilinear complexes to curved/unstructured compatible meshes for real coil/package boundaries;
2. store the tree-cotree selector and compatible operators sparsely end-to-end on large meshes rather than densifying them in the current verification assembler;
3. add physically validated seawater conductivity laws and uncertainty/error propagation for material constants;
4. add a production high-contrast sparse `A-phi` solver/preconditioner with a verified linear-solve residual bound;
5. implement certified thermal spectral-tail rank selection;
6. replace finite candidate-state electromagnetic verification by continuous parameter-domain certification;
7. certify the physical outer seawater/thermal domain treatment.

The rectilinear core is a real spatial discretization and correctness reference, not the final curved-geometry engine.