# SDF-MPNEO Forward Graph

This document fixes the implementation contract for one physical parameter sample unless a leading batch dimension `B` is shown.

## 1. Geometry and operating inputs

The shared encoder receives

- `coil_tokens`: `[B, Nc, Dc]`
- `package_tokens`: `[B, Np, Dp]`
- `global_features`: `[B, Dg]`

`global_features` contains geometry/environment quantities admissible for all physics heads, such as frequency, salinity/reference conductivity, ambient state and imposed inflow descriptors. It must not contain actual port-current amplitudes.

Slow operating information is separated as

- `slow_features`: `[B, Ds]`

and is visible only to thermal/flow heads. It may contain excitation invariants such as `|I|^2`, load descriptors and thermal operating quantities. This makes the EM trial space excitation-independent at the neural interface level.

Coordinate-conditioned basis decoders additionally receive local query tensors for EM potential/harmonic coordinates, thermal DOFs, velocity potential/harmonic coordinates, log-TKE and log-omega. Production hyper-reduced solves need basis values only at operator/cubature points; full-field reconstruction is optional.

## 2. Reference-domain topology

For each topology template the backend supplies gauge-reduced exact-space generators and harmonic complements

- electromagnetic `C_J`, `H_J`, `D_J`;
- velocity `C_v`, `H_v`, `D_v`.

The production path validates

`D_J @ C_J = 0`, `D_J @ H_J = 0`,

`D_v @ C_v = 0`, `D_v @ H_v = 0`,

and requires `[C,H]` to have full column rank. Gauge-null directions therefore cannot consume neural modes.

## 3. Neural outputs and physics seeds

The neural generator outputs coordinates of trial spaces, not physical solutions:

- `current_potential`: `[B, Naj + Nhj, rJmax]`
- `thermal`: `[B, Nt, rTmax]`
- `velocity_potential`: `[B, Nav + Nhv, rVmax]`
- `log_tke`: `[B, Nk, rKmax]`
- `log_omega`: `[B, Nw, rWmax]`

A deterministic solution-free physics seed is prepended to each space. The final basis is

`B_p = orth_P [B_p^phys, (I - Pi_phys) B_p^NN]`.

Typical seed compilers are source/operator + Krylov modes for EM, diffusion modes for thermal physics, divergence-free Stokes modes for velocity, and constant/wall-distance/operator modes for SST log states. No FEM/CFD solution snapshot is permitted in the seed compiler.

Default maximum ranks are `(32, 32, 36, 12, 12)` for `(EM, thermal, velocity, TKE, omega)`.

## 4. Hard admissibility and metric conditioning

Electromagnetic and velocity neural proposals first pass through the exact-sequence generators:

`B_J_raw = [C_J, H_J] @ A_J`,

`B_v_raw = [C_v, H_v] @ A_v`.

All seed and neural columns are checked for rank before any jitter is added. Metric whitening then enforces

`B_p^H P_p B_p = I`.

Jitter is only a Cholesky regulariser and cannot hide a collapsed trial space.

## 5. Nested rank schedule

The initial schedule is

| level | rJ | rT | rv | rk | rw |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 12 | 4 | 4 |
| 2 | 16 | 16 | 20 | 8 | 8 |
| 3 | 24 | 24 | 28 | 10 | 10 |
| 4 | 32 | 32 | 36 | 12 | 12 |

Physics-seed ranks may not exceed the first level, so every online rank retains the complete physical backbone. Neural enrichment occupies the remaining prefix columns.

## 6. Matrix-free electromagnetic inner solve

The production backend does not construct full `[Nj,Nj]` Green, resistance or inductance matrices. Instead it exposes four block actions on `V: [B,Nj,K]`:

- system action `A(V)`;
- multiport source fields `S: [B,Nj,P]`;
- port feedback functional `H(V): [B,P,K]`;
- dissipative action `D(V)`.

The implementation of these actions is backend-specific. It may use structured Green kernels, FFT convolution, FMM/H2, DSE, sparse PDE actions or another deterministic physics solver.

With shared EM basis `B_J: [B,Nj,rJ]`, the core constructs only small matrices:

`A_r = B_J^H A(B_J)` -> `[B,rJ,rJ]`,

`S_r = B_J^H S` -> `[B,rJ,P]`,

`H_r = H(B_J)` -> `[B,P,rJ]`,

`D_r = B_J^H D(B_J)` -> `[B,rJ,rJ]`.

All ports are solved together:

`Y_J = A_r^{-1} S_r`.

The environmental impedance and dissipative port quadratic form are

`Z_sea = H_r Y_J`,

`Q_sea,port = Y_J^H D_r Y_J`.

The backend contract requires these two constructions to be physically consistent; reciprocity, passivity and Joule consistency are verified independently. The dense `(R + j omega L)` Schur formulation remains only as a reference adapter/test and is not the production interface.

## 7. Unified thermal and flow state

The full thermal state is fixed as one mixed-dimensional field

`Theta = [T_c(s), T_package(x), T_seawater(x)]`.

A single thermal basis `B_T` spans the 1D conductor and 3D solid/fluid temperature DOFs. Conservative mortar exchange couples the conductor and package inside the same thermal operator.

The slow reduced state is therefore exactly

`z = [a_T, a_v, a_k, a_omega]`.

At every residual evaluation the backend

1. reconstructs only quantities required at active operator/cubature points;
2. updates temperature/salinity-dependent electrical and fluid properties;
3. builds state-dependent matrix-free EM actions;
4. executes the small EM solve;
5. evaluates seawater/copper heat sources at required points;
6. assembles the coupled mixed-dimensional heat and RANS-SST residual;
7. returns `F_r(z)`.

This is strong coupling: EM is algebraically eliminated but re-evaluated inside the nonlinear thermo-fluid equilibrium.

## 8. Equilibrium and implicit training gradient

Steady operation solves

`F_r(z) = 0`

with pseudo-transient damped Newton

`(P/dtau + J_F) Delta z = -F_r`.

Training never unrolls Newton. After a detached equilibrium solve, the basis is rebuilt once with autograd and the implicit adjoint solves

`F_z^H lambda = dL/dz`,

`dL/dtheta = partial_theta L - lambda^H F_theta`.

No FEM/CFD/full-order solution target is used. The objective is an independent nondimensionalised/Riesz-whitened physical residual supplied by the backend.

## 9. Solution-free hyper-reduction and certification

Basis-Operator Cubature is built from neural/physics basis operator moments only, with non-negative weights. It does not consume solution snapshots.

The online cubature set and certification set must be disjoint. A rank level is accepted only if

1. the reduced equilibrium converged; and
2. independent impedance, thermal and flow indicators satisfy their tolerances.

Failure at maximum rank sets `requires_fallback = True`; the model must not silently return a trusted prediction.

## 10. Backend boundaries

The high-performance backend owns

- geometry and reference-topology templates;
- coordinate query features and Piola/metric maps;
- deterministic physics-seed compilation;
- matrix-free EM actions;
- temperature/salinity-dependent materials;
- conservative 1D-3D conjugate heat operators;
- incompressible RANS-SST operators;
- solution-free Basis-Operator Cubature;
- independent certification actions;
- optional full-field reconstruction.

The neural model must remain unchanged when one deterministic EM implementation is replaced by another or when Python kernels are replaced by C++/CUDA implementations.
