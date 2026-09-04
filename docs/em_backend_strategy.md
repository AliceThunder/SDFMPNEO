# Matrix-Free EM Backend Strategy

This note defines the SDF-MPNEO electromagnetic backend independently of any one implementation.

## 1. Core decomposition

For a geometry-fixed seawater domain, write the environmental operator schematically as

`A_T(V) = V + G[ D_gamma(T,S,f) P(V) ]`,

where

- `P` is the physical compatibility/solenoidal projection appropriate to the discretisation;
- `D_gamma` is a local diagonal/material action containing conductivity, volume and frequency factors;
- `G` is the geometry/environment Green action.

The key implementation rule is that `G` and geometry topology are cached by geometry signature, while `D_gamma` is updated inside the thermo-fluid Newton loop.

Spatially varying conductivity therefore does not require rebuilding the geometry Green kernel. A backend only changes local material weights before each Green action.

## 2. Reduced projection

For active shared EM basis `B_J`, online assembly requests only

`A_T(B_J)`

and forms

`A_r = B_J^H A_T(B_J)`.

The source kernel directly accumulates

`S_r = project_rhs(B_J)`

rather than forcing construction of a full source field. Port feedback similarly evaluates only

`H_r = port_feedback(B_J)`.

The resulting solve is

`A_r Y = S_r`,

`Z_env = H_r Y`.

No dense full-order Green or impedance operator is part of the core contract.

## 3. Geometry compression

The source backend may exploit repeated-turn/affine geometry, symmetry, analytic segment families, hierarchical quadrature, or other structure. SDF-MPNEO does not prescribe a specific rounded-square or finite-comb formula.

The preferred new abstraction is a `GeometryMomentCompiler` that converts the unified coil geometry into source moments required by

- physics-seed compilation;
- `project_rhs(B_J)`;
- port feedback;
- independent source certification.

The compiler must operate on the unified SDF-MPNEO geometry representation rather than importing BFZI family classes.

## 4. Thermal coupling

At every slow-state residual evaluation:

1. reconstruct seawater temperature at the material/operator points;
2. compute local `sigma_sea(T,S)`;
3. update `D_gamma` only;
4. apply the cached geometry Green operator to the active EM basis block;
5. solve the reduced multiport system;
6. evaluate Joule heat at thermal cubature/certification points.

This preserves strong EM-thermal feedback without rebuilding the global environment operator.

## 5. Flow coupling

Direct motional induction is activated only when the magnetic Reynolds-number criterion requires it. If included, velocity-dependent terms enter through additional matrix-free actions; they do not alter the neural basis contract.

## 6. Backend implementation options

Valid implementations include

- FFT convolution on a regular masked volume grid;
- FMM or H2 Green action;
- DSE/structured integral kernels;
- sparse compatible PDE discretisations;
- hybrid near-field exact + far-field accelerated operators.

These implementations may be benchmarked against BFZI, but none is part of the SDF-MPNEO model definition.

## 7. Cache signatures

Geometry-dependent cached artifacts must include every parameter that changes topology or Green geometry. Material-dependent quantities such as temperature-updated conductivity are not baked into the geometry cache. Frequency-dependent artifacts must include frequency in their signature unless the cached object is mathematically frequency independent.
