# Continuous-domain residual certification

## Purpose

SDF-MPNEO is solution-data-free, but finite collocation does not prove uniform validity on a continuous parameter domain. The final target is therefore

\[
\sup_{(a_0,G,U,t)\in\Omega}\|R_v(a_0,G,U,t)\|_2\le\varepsilon,
\qquad \varepsilon=10^{-5},
\]

with

\[
R_v=\dot a_\theta-M_r(G)^{-1}\{-K_r(G)a_\theta+q_r(G,a_\theta,U)\}.
\]

The implementation is deterministic and fail-closed: unresolved proof obligations return `indeterminate`; random validation is never promoted to a continuous-domain proof.

## Mass-residual formulation

Define

\[
r_M=M_r(G)\dot a_\theta+K_r(G)a_\theta-q_r(G,a_\theta,U)=M_r(G)R_v.
\]

If a branch proof gives

\[
\lambda_{\min}(M_r(G))\ge m_-(B)>0,
\]

then

\[
\|R_v\|_2\le \|r_M\|_2/m_-(B).
\]

For a physical box

\[
B=\{x:|x_i-x_i^c|\le h_i\},\qquad x=(a_0,G,U,t),
\]

the mean-value theorem gives

\[
\|r_M(x)\|_2\le\|r_M(x^c)\|_2+
\sum_i h_i\sup_{x\in B}\|\partial_i r_M(x)\|_2.
\]

The analytic DAG encloses `a`, `a_dot` and every first parameter/time derivative exactly in its polynomial-exponential algebra. Geometry derivatives are converted from normalized graph coordinates back to the physical geometry coordinates by the exact affine chain rule.

For physical bounds

\[
\|M\|\le\bar M,\quad\|K\|\le\bar K,\quad
\|M_{,G_k}\|\le M'_k,\quad\|K_{,G_k}\|\le K'_k,
\]

and Joule bounds

\[
\|q_{,a}\|\le L_q,\qquad
\|q^{\rm explicit}_{,G_k}\|\le Q_{G_k},\qquad
\|q^{\rm explicit}_{,U_k}\|\le Q_{U_k},
\]

each coordinate obeys

\[
\|\partial_i r_M\|
\le
\bar M\|\partial_i\dot a\|
+(\bar K+L_q)\|\partial_i a\|+E_i,
\]

where

\[
E_i=
\begin{cases}
M'_k\|\dot a\|+K'_k\|a\|+Q_{G_k},&i=G_k,\\
Q_{U_k},&i=U_k,\\
0,&i\in\{a_0,t\}.
\end{cases}
\]

Hence

\[
\overline R_v(B)=
\frac{\|r_M(x^c)\|+\sum_i h_iL_i(B)}{m_-(B)}
\]

is the branch upper bound. The branch is permanently resolved when this is below tolerance and every physical ingredient is certified. Otherwise the coordinate with the largest contribution `h_i L_i(B)` is bisected.

## Production thermal-operator theorem provider

`certify_geometry_thermal_operator_bounds()` now closes the thermal geometry obligation directly from `AffineTetrahedralGeometryChart.certify_box()`.

For a branch-center affine tetrahedral deformation `F=I+E`, define

\[
\eta_k=\sup_B\|F^{-1}\partial_{G_k}F\|_2.
\]

The chart supplies a certified perturbation radius `rho<1` and directional center derivatives, giving

\[
\eta_k\le \frac{\|\partial_{G_k}F\|_2}{1-\rho}.
\]

P1 mass transforms only by `det(F)`, so

\[
|\partial_{G_k}M|\preceq3\eta_k M.
\]

P1 stiffness transforms as `det(F)F^{-1}F^{-T}`, giving

\[
|\partial_{G_k}K|\preceq5\eta_k K.
\]

The existing whole-branch P1 mass/stiffness form ratios then produce certified `m_-`, `||M||`, `||K||`, `||M_,G||` and `||K_,G||`. Projection to the fixed shared thermal basis preserves these quadratic-form inequalities.

## Production reduced-EM/Joule theorem provider

`certify_geometry_joule_derivative_bounds()` closes the nonlinear Joule obligation in the fixed shared reduced EM space used by the training residual itself.

At the branch center, let

\[
H_c=V^H H V,
\]

where `H=K_em+D_em` is the physical A-psi energy metric. The affine Nedelec chart supplies whole-branch curl and conductive-mass ratios. The constitutive theorem `_material_energy_ratios()` supplies certified thermal conductivity ratios and directional conductivity derivatives over the induced thermal-state box. Combining them gives

\[
\mu H_c\preceq H(G,a)\preceq\nu H_c.
\]

Physical A-psi coercivity remains

\[
|x^HAx|\ge\beta x^HHx,\qquad\beta\ge1/\sqrt2.
\]

Therefore source dual-energy bounds give a branch-wide reduced EM state bound without solution snapshots.

For each geometry direction, Nedelec curl-curl and covariant mass forms satisfy the conservative differential estimate

\[
\|A_{,G_k}\|_{H_c}\le5\eta_k\nu.
\]

The same affine factor controls each reduced loss form. Thermal derivatives use the certified constitutive directional bounds.

### Terminal-current geometry dependence

Solid-terminal RHS vectors are normalized P1 surface-area loads. For one transported terminal surface,

\[
|\partial_{G_k}A_f|\le2\eta_k A_f,
\]

and normalization gives

\[
|\partial_{G_k}w_i|\le4\eta_k w_i.
\]

A positive-minus-negative two-terminal port column therefore has Euclidean derivative norm at most `8 eta_k`. Projection through the fixed reduced basis gives a certified reduced RHS geometry derivative. Thus explicit `dq/dG` includes both operator/loss deformation and terminal-source deformation; no term is silently omitted.

The resulting provider returns certified bounds for

\[
\|dq/da\|_2,\qquad
\|\partial^{\rm explicit}_{G_k}q\|_2,\qquad
\|\partial^{\rm explicit}_{U_k}q\|_2.
\]

No finite difference or sampled Jacobian maximum is used by the proof.

## Public API

The low-level continuous certifier remains in `geometry_mass_residual_domain.py`:

- `GeometryThermalOperatorBounds`
- `GeometryJouleDerivativeBounds`
- `GeometryMassResidualPhysicalProof`
- `bound_geometry_mass_residual_on_box`
- `certify_geometry_mass_residual_domain`

Production theorem providers are in `geometry_mass_providers.py`:

- `certify_geometry_thermal_operator_bounds(model, box)`
- `certify_geometry_joule_derivative_bounds(model, box)`
- `make_uwpt_mass_residual_physical_proof_factory(model)`
- `certify_trained_geometry_model_finite_domain(model, work_budget=...)`

The final convenience function automatically reads the saved initial-condition range, physical geometry chart, current range, time horizon and residual tolerance from a trained `GeometryResearchModel`.

## Status outcomes

- `certified`: every continuous branch has a theorem-derived upper bound below tolerance;
- `violated`: an explicitly evaluated branch center already exceeds tolerance;
- `indeterminate`: the branch budget is exhausted or a theorem bound cannot be constructed.

`indeterminate` never falls back to a statistical claim.

## Finite time and steady state

The current continuous residual certifier covers

\[
0\le t\le T<\infty.
\]

It explicitly rejects `time_upper=inf`. Exact stationary certification remains a separate proof obligation; a large finite time is not silently identified with `t=+inf`.

## Relationship to training

The intended workflow is

\[
\text{residual-grown training}
\rightarrow
\text{continuous-domain certification}
\rightarrow
\text{refine only unresolved/worst proof boxes}.
\]

Worst-residual exchange remains useful as a counterexample finder and training accelerator, but it is no longer the theoretical validity criterion.

## Claim language

A suitable final positioning is:

> SDF-MPNEO replaces test-set statistical generalization as the final validity criterion with continuous physical-domain residual certification. Training remains numerical, while acceptance is based on a deterministic governing-residual bound over the declared parameter domain.

A full-domain UWPT claim is made only when `status == "certified"` and `unresolved_boxes == 0` for the declared finite-time domain.
