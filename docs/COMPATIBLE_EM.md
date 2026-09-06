# Compatible magnetoquasistatic A-psi core

The executable electromagnetic core is derived from the magnetoquasistatic field equations and uses compatible discrete topology. It has no geometry-specific impedance-solver dependency and introduces no gauge penalty parameter.

## 1. Discrete variables and gauges

Let

- `C` be the edge-to-face discrete curl matrix;
- `G_c` be the conducting-region scalar-potential gradient after removal of one constant-potential null mode per connected conducting component;
- `R_A` be the deterministic tree-cotree magnetic-vector-potential gauge basis, so `A = R_A alpha`;
- `M_nu` be the face reluctivity Hodge matrix;
- `M_sigma(a)` be the edge conductivity Hodge matrix.

Compatibility requires

```text
C G_c = 0.
```

The tree-cotree gauge fixes the vector-potential coordinates on a deterministic spanning tree and retains cotree-edge coordinates. No SVD rank threshold and no gauge-penalty coefficient is part of the method.

## 2. Reciprocal scaled scalar-potential coordinate

With harmonic convention `exp(j omega t)`, define

```text
psi = phi / (j omega).
```

The unknown is

```text
x = [alpha; psi]
```

and the electric field is

```text
E = -j omega (R_A alpha + G_c psi)
  = L_E x,
L_E = -j omega [R_A, G_c].
```

The gauge-eliminated system is

```text
[ K_A + j omega R_A^H M_sigma R_A,  j omega R_A^H M_sigma G_c ] [alpha]   [R_A^H J_s]
[ j omega G_c^H M_sigma R_A,         j omega G_c^H M_sigma G_c ] [ psi ] = [    0     ],
```

with

```text
K_A = R_A^H C^H M_nu C R_A.
```

For real reciprocal material Hodge matrices this operator is complex symmetric. Reciprocity is therefore a structural property of the field discretization rather than a post-processing symmetrization.

## 3. Physical Riesz metric

The residual norm is based on magnetic energy plus Joule energy dissipated per electrical radian:

```text
H_em = blockdiag(1/2 K_A, 0)
       + (1/(2 omega)) L_E^H M_sigma,ref L_E.
```

The implementation factors

```text
H_em = L L^H
```

and computes

```text
||r||_(H_em^-1) = ||L^-1 r||_2,
H_em^-1 r        = L^-H L^-1 r.
```

Thus Riesz lifting and basis orthogonalization are carried out in physical energy coordinates without explicitly forming an inverse.

## 4. Temperature-dependent constitutive laws

Two executable paths exist:

1. an affine verification path,
   ```text
   M_sigma(a) = M_sigma,0 + sum_k a_k M_sigma,k;
   ```
2. an exact nonlinear path in which
   ```text
   T(a) = T_ref + sum_k a_k Phi_k
   ```
   is reconstructed on material cells and the conductivity law is evaluated directly.

For copper the current core supports the analytic relation

```text
rho_Cu(T) = rho_ref [1 + alpha (T-T_ref)],
sigma_Cu(T) = sigma_ref / [1 + alpha (T-T_ref)],
```

with exact derivatives. Other analytic material laws can implement the same constitutive interface. No affine temperature approximation is required by the nonlinear field core.

## 5. Heat-source projection and regional losses

For retained thermal coordinate `j`,

```text
q_j(a) = 1/2 x^H L_E^H W_j(a) L_E x.
```

The heat-source Jacobian includes both the implicit derivative of the electromagnetic field and the explicit derivative of the conductivity/loss operator.

Separate copper and seawater diagnostics are formed from the same field solution:

```text
P_Cu  = 1/2 E^H M_sigma,Cu E,
P_sea = 1/2 E^H M_sigma,sea E.
```

These are not add-on resistance models. The thermal equation continues to use the unified spatially projected Joule source.

## 6. Snapshot-free electromagnetic reduction

The reduced space is constructed from residual Riesz lifts. For multiple ports, one common basis is grown over the joint set

```text
thermal state x port RHS.
```

At each enrichment step the worst residual over all supplied states and excitations is lifted and added. A basis certified only for one port is never assumed to represent another port.

The reduced electromagnetic model also supports arbitrary RHS vectors directly, so the operating excitation is not permanently tied to the nominal `problem.b`.

## 7. Multiport impedance from the field equation

For one-ampere divergence-free impressed-current source cochains collected in `B`, the unit-port field states satisfy

```text
A(a) X = B.
```

The flux-linkage and impedance matrices are

```text
Psi = B^T X,
Z   = j omega Psi,
R   = Re(Z),
L   = Im(Z)/omega.
```

For reciprocal media,

```text
Z^T = Z.
```

For passive positive conductivity,

```text
R >= 0
```

in the positive-semidefinite sense. For arbitrary complex port-current vector `I`, the average input real power obeys

```text
1/2 Re(I^H Z I)
= 1/2 E^H M_sigma E,
```

which is used as an executable power-consistency regression.

The off-diagonal entries of `L` are the mutual inductances for this impressed-current port definition.

## 8. Real 3-D orthogonal spatial assembly

`RectilinearComplex3D` constructs nodes, edges, faces, exact incidence matrices, tree-cotree gauge coordinates, and orthogonal material Hodge matrices from x/y/z cell coordinates. The electromagnetic and thermal cores can share the same cellwise conductor/package/seawater material description.

This rectilinear complex is the current real spatial implementation. Curved coil/package boundaries still require an unstructured compatible mesh or a rigorously structure-preserving mapped complex.

## 9. Current chain

```text
3-D material cells
  -> exact discrete topology C,G
  -> tree-cotree R_A and conductor G_c
  -> material Hodge operators
  -> reciprocal gauge-eliminated A-psi field system
  -> snapshot-free multi-RHS EM reduction
  -> q_em,r(a), dq_em,r/da
  -> P_Cu, P_sea and Z/R/L/M outputs
  -> electro-thermal residual
  -> analytic neural evolution and residual-grown topology.
```

## Remaining obligations

- curved/unstructured compatible geometry for realistic round and rounded-square coils and package interfaces;
- scalable sparse high-contrast field solution with a verified linear-solve error bound;
- continuous geometry/temperature/frequency-domain electromagnetic certification rather than a finite candidate set;
- rigorous thermal spectral-tail/output estimator for scalable thermal rank selection;
- certified treatment of the outer seawater thermal/electromagnetic truncation or open-domain boundary;
- solid-conductor terminal-current ports when terminal-driven conductor current, rather than impressed stranded-current excitation, is required.
