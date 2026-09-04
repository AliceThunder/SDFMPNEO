# SDF-MPNEO Forward Graph

This document fixes the first implementation contract. Symbols below refer to **one physical parameter sample** unless a leading batch dimension `B` is shown.

## 1. Geometry inputs

The shared encoder receives

- `coil_tokens`: `[B, Nc, Dc]`
- `package_tokens`: `[B, Np, Dp]`
- `global_features`: `[B, Dg]`

`global_features` may contain frequency, seawater salinity/reference conductivity, ambient temperature, imposed inflow velocity and dimensionless operating descriptors. Port currents are **not** admissible EM-head inputs; the EM trial space is excitation-independent.

## 2. Reference-domain topology

For each topology template (single-package and dual-package in V1), the backend supplies the exact-sequence maps

- electromagnetic curl map `C_J`: `[Nj, Naj]`
- electromagnetic harmonic basis `H_J`: `[Nj, Nhj]`
- velocity curl map `C_v`: `[Nv, Nav]`
- velocity harmonic basis `H_v`: `[Nv, Nhv]`

The incidence structure must satisfy the discrete identity `D @ C == 0`. Harmonic columns must also lie in `ker(D)`.

## 3. Neural outputs

The neural generator outputs **coordinates of trial spaces**, not physical solutions:

- `current_potential`: `[B, Naj + Nhj, rJmax]`
- `thermal`: `[B, Nt, rTmax]`
- `velocity_potential`: `[B, Nav + Nhv, rVmax]`
- `log_tke`: `[B, Nk, rKmax]`
- `log_omega`: `[B, Nw, rWmax]`

Default maximum ranks are

- `rJmax = 32`
- `rTmax = 32`
- `rVmax = 36`
- `rKmax = 12`
- `rWmax = 12`

## 4. Hard admissibility maps

Electromagnetic and velocity bases are generated through the reference de Rham complex:

`B_J_raw = [C_J, H_J] @ A_J`

`B_v_raw = [C_v, H_v] @ A_v`

Hence `D @ B_J_raw == 0` and `D @ B_v_raw == 0` by construction. Boundary-normal DOFs excluded by the topology template enforce no-through-current/no-through-wall conditions.

All five bases are metric-orthogonalised:

`B_p^H P_p B_p = I`.

The metric is a fixed/reference physical metric for basis conditioning. State-dependent online operators are **not** assumed to remain identity after temperature/material updates.

## 5. Nested rank schedule

The initial schedule is

| level | rJ | rT | rv | rk | rw |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 12 | 4 | 4 |
| 2 | 16 | 16 | 20 | 8 | 8 |
| 3 | 24 | 24 | 28 | 10 | 10 |
| 4 | 32 | 32 | 36 | 12 | 12 |

Only prefix columns are activated. Training therefore has to randomise the active level so early columns become the dominant low-rank modes.

## 6. Electromagnetic inner solve

For each candidate thermal state, the backend updates material-dependent electromagnetic operators:

- `R_s(T,S)`: `[B, Nj, Nj]`, real symmetric dissipative operator
- `L_s(G)`: `[B, Nj, Nj]`, real symmetric inductive operator (matrix-free in the final backend)
- `C_c(G)`: `[B, Nj, P]`, coil-to-seawater coupling
- `Z_0(T_c,f,G)`: `[B, P, P]`, copper + air contribution
- `omega`: `[B]`

With shared real basis `B_J: [B, Nj, rJ]`,

`A_r = B_J^T (R_s + j omega L_s) B_J`  -> `[B, rJ, rJ]`

`C_r = B_J^T C_c` -> `[B, rJ, P]`

All unit-port responses are solved together:

`Y_J = -j omega A_r^{-1} C_r` -> `[B, rJ, P]`

`J_s = B_J Y_J` -> `[B, Nj, P]`

The seawater impedance is recovered by Schur elimination:

`Z_sea = omega^2 C_r^T A_r^{-1} C_r` -> `[B, P, P]`

`Z = Z_0 + Z_sea`.

The same trial space is used for all ports. With symmetric physical operators this preserves reciprocity. The dissipative part is structurally non-negative because it equals the reduced seawater Joule quadratic form.

## 7. Slow multiphysics state

The electromagnetic coefficients are not part of the slow Newton unknown. The backend defines a reduced slow state containing

`z = [a_T, a_v, a_k, a_omega]`.

Its exact flattened length depends on the active rank level and on whether the 1D conductor temperature is represented inside the thermal basis or as a coupled block. V1 backend must expose this layout consistently through `initial_slow_state` and `assemble_reduced`.

At every residual evaluation:

1. reconstruct only thermal/flow quantities required at active cubature/operator points;
2. update `sigma_sea(T,S)`, `rho_Cu(T_c)`, fluid `mu(T)`, `rho(T)`, `k(T)` and SST closure quantities;
3. assemble/update the reduced/full-action EM operators;
4. execute the EM Schur solve;
5. compute `Q_sea = rho_s |J_s|^2` and copper loss;
6. assemble the coupled 1D-3D conjugate-heat and RANS-SST reduced residual;
7. return the single slow residual vector `F_r(z)`.

This is strong coupling: the EM solve is eliminated algebraically but is re-evaluated inside the thermo-fluid nonlinear equilibrium.

## 8. Equilibrium solve

Steady operation solves

`F_r(z) = 0`.

The foundation solver uses pseudo-transient damped Newton:

`(P/dtau + J_F) Delta z = -F_r`.

Small `dtau` stabilises remote iterates; successful steps increase `dtau`, asymptotically recovering Newton. The nonlinear system is solved per physical parameter sample to avoid artificial cross-sample Jacobians.

Transient extension reuses the same reduced physics as

`M_r dz/dt + F_r(z) = 0`.

## 9. Certification and adaptive capacity

The reduced solve and the certifier must not use the same hyper-reduction sample set. The backend returns independent indicators

- impedance indicator `eta_Z`
- thermal indicator `eta_T`
- flow indicator `eta_v`

A rank level is accepted only if

1. the reduced equilibrium converged, and
2. all three independent indicators satisfy their tolerances.

Otherwise the next nested rank is activated. Failure at maximum rank sets `requires_fallback = True`; it must not silently return a trusted prediction.

For linear electromagnetic states, residual-based bounds may be rigorous under the dissipative norm. For nonlinear RANS-SST thermal-flow quantities, certification is initially treated as a residual/dual-weighted a-posteriori indicator unless stronger stability constants are established.

## 10. Backend boundaries

The final high-performance backend is expected to provide

- topology templates and geometry/Piola maps;
- BFZI/DSE matrix-free Green-operator actions for EM;
- temperature/salinity-dependent material models;
- 1D conductor to 3D package conservative mortar coupling;
- 3D package/seawater conjugate heat operators;
- incompressible RANS-SST operators;
- solution-free basis-operator cubature for local nonlinear terms;
- an independent certification sample/operator set;
- optional full-field reconstruction only when requested.

The neural model must remain unchanged when Python reference kernels are replaced by C++/CUDA implementations.
