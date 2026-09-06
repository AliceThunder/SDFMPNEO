# Compatible magnetoquasistatic A-phi core

The executable electromagnetic core now has an algebraic compatible-discretization layer. It does not require a geometry-specific impedance solver and does not introduce a gauge penalty parameter.

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

This is an executable affine core. The final constitutive layer may replace it by an exact separated representation or a certified expansion with remainder bounds.

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

The magnetic-vector-potential gauge is removed by the coordinate basis `R_A`, not by adding an empirical penalty. The scalar-potential constant nullspace is likewise required to be removed before `G` is passed to the assembler.

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

## Connection to snapshot-free reduction

`CompatibleAphiDiscretization.to_parametric_problem()` produces the existing `ParametricEMProblem`. The residual-Riesz reducer then builds the electromagnetic reduced space without full-order solution snapshots.

The current chain is therefore

```text
discrete topology/material Hodge operators
            -> compatible gauge-eliminated A-phi system
            -> snapshot-free residual-Riesz EM reduction
            -> reduced temperature-dependent loss map and exact Jacobian
            -> electro-thermal residual
            -> analytic neural evolution/growth.
```

## Remaining implementation obligations

- construct `C`, `G`, and `R_A` from an actual 3-D mesh/topology rather than supplying them directly;
- assemble copper, package, and seawater material Hodge matrices from geometry and constitutive laws;
- distinguish copper and seawater local loss contributions before projection when output diagnostics require them separately;
- replace finite candidate-state EM verification by continuous parameter-domain certification;
- add certified non-affine constitutive separation when conductivity is not exactly affine in the retained thermal coordinates.
