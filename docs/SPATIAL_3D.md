# Shared 3-D electromagnetic–thermal spatial core

The executable core now constructs the electromagnetic and thermal operators from one orthogonal rectilinear 3-D material grid. This is the first stage in which conductor, package, seawater, and passive regions are represented as spatial cell fields rather than as hand-written small matrices.

## 1. Exact discrete topology

`RectilinearComplex3D` creates oriented nodes, edges, and faces. The edge-to-node gradient incidence matrix `G_full` and face-to-edge curl incidence matrix `C` are generated from topology alone and satisfy

```text
C @ G_full = 0
```

exactly in sparse integer arithmetic.

The magnetic-vector-potential gauge is removed by

```text
range(R_A) = ker(G_full^T),
A = R_A alpha.
```

For every connected conducting component, one scalar-potential reference node is removed deterministically, so the conductor gradient `G_c` has no constant-potential nullspace. No gauge penalty or small regularization parameter is introduced.

## 2. Orthogonal material Hodge matrices

Material fields are cellwise. For an x-directed primal edge of length `dx`, the conductivity Hodge coefficient is the integral of conductivity over the orthogonal dual face divided by `dx`. Each adjacent cell contributes its exact quarter dual area,

```text
M_sigma,e = (1/dx) sum_adjacent_cells sigma_c (dy_c/2)(dz_c/2).
```

The analogous y/z formulas are used automatically.

For an x-normal primal face of area `dy dz`, the reluctivity Hodge coefficient is the integral of reluctivity along the orthogonal dual edge divided by the primal area,

```text
M_nu,f = (1/(dy dz)) sum_adjacent_cells nu_c (dx_c/2).
```

Thus discontinuous conductor/package/seawater coefficients are not replaced by an arithmetic interface average.

## 3. Spatial A-phi assembly

`build_compatible_aphi_from_cells(...)` accepts

- cell reluctivity;
- reference cell conductivity;
- certified affine conductivity coefficients in reduced thermal coordinates;
- cell values of the retained thermal test modes;
- a divergence-free edge current source.

It constructs

```text
C, G_c, R_A, M_nu, M_sigma,0, M_sigma,k
```

and passes them into the compatible gauge-eliminated `A-phi` core.

The support of conductivity is fixed by the reference material field. Thermal coefficients may change conductivity inside an already-conducting material region but are rejected if they would create a new conducting region from a reference insulator. This prevents a parameter expansion from silently changing the topology of the scalar-potential space.

## 4. Physical electromagnetic Riesz metric

The 3-D builder no longer uses a purely topological residual norm. It constructs

```text
H_em = H_mag + H_diss/radian
```

with

```text
H_mag = 1/2 alpha^H (R_A^H C^H M_nu C R_A) alpha
```

and

```text
H_diss/radian = (1/(2 omega)) E^H M_sigma,0 E.
```

Equivalently, in the full `[alpha; phi]` coordinates,

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2 omega)) L_E^H M_sigma,0 L_E.
```

This is derived from magnetic energy plus Joule energy dissipated per electrical radian. It contains no tunable balancing coefficient. Positive definiteness is checked after all gauge eliminations; failure is reported rather than regularized by an arbitrary diagonal shift.

## 5. Shared thermal finite-volume grid

`RectilinearThermalFV3D` uses the same x/y/z cell geometry. The thermal mass matrix is

```text
M_T,c = (rho c_p)_c V_c.
```

Across a material interface, the face conductance is obtained from the two half-cell thermal resistances,

```text
G_f = A_f / (d_L/(2 k_L) + d_R/(2 k_R)).
```

This enforces the finite-volume heat-flux continuity between conductor, package, and seawater without arithmetic averaging of thermal conductivity.

The current homogeneous outer-boundary implementation supports either:

- exact half-cell Dirichlet boundary resistance on the six outer surfaces; or
- no outer Dirichlet contribution when the caller is constructing a different boundary treatment externally.

The physical outer boundary itself remains a model input; its location is not justified by this discretizer.

## 6. Current scope and next obligations

This stage removes the earlier requirement that `C`, `G`, `R_A`, and material Hodge matrices be supplied manually. Remaining work is now narrower:

1. generalize from orthogonal rectilinear complexes to an unstructured compatible mesh for curved coil/package boundaries;
2. replace the dense null-space construction of `R_A` by a sparse tree/cotree or equivalent scalable exact gauge construction;
3. add copper/seawater region-separated Joule diagnostic operators in addition to the combined reduced thermal source;
4. add the certified non-affine constitutive-separation layer for actual `sigma_Cu(T)` and `sigma_sea(T)` rather than assuming the caller has already supplied affine coefficients;
5. implement the certified thermal spectral-tail estimator so the 3-D thermal basis need not retain the full spectrum;
6. replace finite candidate-state electromagnetic verification by a continuous parameter-domain certificate.

The rectilinear core is therefore a real spatial discretization, not the final geometry engine. It is intended to validate the exact-sequence, material-interface, gauge, reduction, coupling, and analytic-network pipeline before the same algebra is transferred to curved unstructured geometry.
