# Compatible magnetoquasistatic A-phi core

The executable electromagnetic core now has both an algebraic compatible-discretization layer and a first real 3-D orthogonal spatial assembler. It does not require a geometry-specific impedance solver and does not introduce a gauge penalty parameter.

## Discrete variables

Let

- `C` be the edge-to-face discrete curl matrix;
- `G` be the conducting-region scalar-potential gradient matrix after removal of the conductor-wise constant-potential nullspace;
- `R_A` be an explicit magnetic-vector-potential gauge-elimination basis, so the full edge potential is `A = R_A alpha`;
- `M_nu` be the face reluctivity Hodge matrix;
- `M_sigma(a)` be the edge conductivity Hodge matrix.

Compatibility requires

```text
C G = 0.
```

The present temperature dependence is represented algebraically as

```text
M_sigma(a) = M_sigma,0 + sum_k a_k M_sigma,k.
```

This is an executable affine core. The constitutive layer must either provide an exact separated representation or a certified expansion with a remainder bound; the spatial assembler does not fit constitutive laws itself.

## Gauge-eliminated A-phi system

With harmonic convention `exp(j omega t)`, the unknown is

```text
x = [alpha; phi]
```

and the assembled system is

```text
[ R_A^H C^H M_nu C R_A + j omega R_A^H M_sigma R_A,   R_A^H M_sigma G ] [alpha]   [R_A^H J_s]
[ j omega G^H M_sigma R_A,                              G^H M_sigma G       ] [ phi ] = [    0     ].
```

The magnetic-vector-potential gauge is removed by the coordinate basis `R_A`, not by adding an empirical penalty. The scalar-potential constant nullspace is likewise removed separately in every connected conducting component.

## Electric field and loss projection

The full edge electric field is

```text
E = -j omega R_A alpha - G phi = L_E x,
```

with

```text
L_E = [-j omega R_A, -G].
```

For each retained thermal coordinate `j`, a conductivity-weighted loss Hodge matrix `W_j(a)` produces the reduced heat-source component

```text
q_j(a) = 1/2 x^H L_E^H W_j(a) L_E x.
```

The executable core supports

```text
W_j(a) = W_j,0 + sum_k a_k W_j,k,
```

so the heat-source Jacobian contains both:

1. the implicit field-state derivative through the electromagnetic solve; and
2. the explicit temperature derivative of the loss operator itself.

## Real 3-D orthogonal spatial assembly

`RectilinearComplex3D` now generates the oriented node/edge/face complex directly from x/y/z coordinates. Its integer incidence matrices satisfy

```text
C @ G_full = 0
```

by construction.

`build_compatible_aphi_from_cells(...)` constructs `C`, the conducting-region `G`, the explicit gauge basis `R_A`, and the material Hodge matrices from cellwise reluctivity and conductivity fields. The 1-form and 2-form Hodge coefficients integrate the cell coefficients over the corresponding orthogonal dual portions, so conductor/package/seawater jumps are retained spatially.

The physical residual Riesz metric is assembled from magnetic energy plus Joule energy dissipated per electrical radian,

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2 omega)) L_E^H M_sigma,0 L_E.
```

No arbitrary diagonal shift is added if this metric is singular; the assembler instead reports a gauge/topology/material-support failure.

See [`SPATIAL_3D.md`](SPATIAL_3D.md) for the spatial formulas and present scope.

## Connection to snapshot-free reduction

`CompatibleAphiDiscretization.to_parametric_problem()` produces the existing `ParametricEMProblem`. The residual-Riesz reducer then builds the electromagnetic reduced space without full-order solution snapshots.

The current chain is therefore

```text
3-D material cells
       -> exact discrete topology and material Hodge operators
       -> compatible gauge-eliminated A-phi system
       -> snapshot-free residual-Riesz EM reduction
       -> reduced temperature-dependent loss map and exact Jacobian
       -> electro-thermal residual
       -> analytic neural evolution/growth.
```

## Remaining implementation obligations

- generalize the spatial core from an orthogonal rectilinear complex to curved/unstructured compatible meshes needed by real coil and package boundaries;
- replace the dense null-space gauge basis by a sparse scalable tree/cotree or equivalent exact construction;
- distinguish copper and seawater local loss contributions before projection when output diagnostics require them separately;
- replace finite candidate-state EM verification by continuous parameter-domain certification;
- add certified non-affine constitutive separation for actual temperature-dependent copper and seawater conductivity;
- add scalable certified thermal-rank selection for the shared 3-D thermal operator.
