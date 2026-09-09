# Continuous-domain residual certification

## Purpose

SDF-MPNEO is solution-data-free, but finite collocation does not by itself prove uniform validity on a continuous parameter domain. The final reliability target is therefore not a statistical test-set statement. It is the deterministic physical-domain statement

\[
\sup_{(a_0,G,U,t)\in\Omega}\|R_v(a_0,G,U,t)\|_2\le \varepsilon,
\qquad \varepsilon=10^{-5},
\]

where

\[
R_v=\dot a_\theta-M_r(G)^{-1}\{-K_r(G)a_\theta+q_r(G,a_\theta,U)\}.
\]

The code now contains a fail-closed first-order branch-and-bound implementation for finite time intervals. It is intended to replace dependence on an ever-growing number of random validation points once all physical derivative proof obligations are available.

## Why certify the mass residual

Define

\[
r_M=M_r(G)\dot a_\theta+K_r(G)a_\theta-q_r(G,a_\theta,U).
\]

Then

\[
r_M=M_rR_v.
\]

If a branch proof supplies

\[
\lambda_{\min}(M_r(G))\ge m_-(B)>0
\quad\forall G\in B,
\]

then

\[
\|R_v\|_2\le \frac{\|r_M\|_2}{m_-(B)}.
\]

This formulation avoids differentiating an interval-valued matrix inverse. The physical proof obligations become bounds on the thermal matrices and on the Joule source itself.

## First-order branch bound

For a physical parameter box

\[
B=\{x:|x_i-x_i^c|\le h_i\},\qquad x=(a_0,G,U,t),
\]

the mean-value theorem gives

\[
\|r_M(x)\|_2
\le
\|r_M(x^c)\|_2+
\sum_i h_i\sup_{x\in B}\|\partial_i r_M(x)\|_2.
\]

The analytic DAG supplies rigorous bounds for

\[
a_\theta,\quad \dot a_\theta,\quad
\partial_i a_\theta,\quad
\partial_i\dot a_\theta
\]

because every compiled term has the form

\[
c\,p^\gamma t^m e^{-\rho t}.
\]

For physical geometry coordinates the graph derivative with respect to normalized geometry is converted exactly by the affine normalization chain rule.

Let a branch-local physical proof provide

\[
\|M\|\le \bar M,\quad
\|K\|\le \bar K,\quad
\|\partial_{G_k}M\|\le M_k',\quad
\|\partial_{G_k}K\|\le K_k',
\]

and Joule bounds

\[
\|\partial_a q\|\le L_q,\quad
\|\partial_{G_k}^{\rm explicit}q\|\le Q_{G_k},\quad
\|\partial_{U_k}^{\rm explicit}q\|\le Q_{U_k}.
\]

Then each coordinate derivative of the mass residual is bounded by

\[
\|\partial_i r_M\|
\le
\bar M\,\|\partial_i\dot a_\theta\|
+(\bar K+L_q)\,\|\partial_i a_\theta\|
+E_i,
\]

with

\[
E_i=
\begin{cases}
M_k'\,\|\dot a_\theta\|+K_k'\,\|a_\theta\|+Q_{G_k}, & i=G_k,\\
Q_{U_k}, & i=U_k,\\
0, & i\in\{a_0,t\}.
\end{cases}
\]

Therefore

\[
\overline r_M(B)=
\|r_M(x^c)\|+
\sum_i h_iL_i(B)
\]

is a rigorous branch upper bound, and

\[
\boxed{
\overline R_v(B)=\overline r_M(B)/m_-(B)
}
\]

is the corresponding vector-residual upper bound.

## Adaptive certification tree

A branch is permanently resolved when

\[
\overline R_v(B)\le\varepsilon
\]

and its physical proof is certified. Otherwise the implementation splits the coordinate with the largest certified contribution

\[
h_iL_i(B).
\]

Thus increasing proof resolution is not the same as increasing a validation sample count. The algorithm refines a mathematical enclosure only where the current proof is too loose.

Possible outcomes are:

- `certified`: every branch has a certified upper bound below tolerance;
- `violated`: an explicitly evaluated branch center already exceeds tolerance;
- `indeterminate`: the work budget is exhausted, a proof provider is uncertified, or a required bound cannot be constructed.

There is deliberately no fallback from `indeterminate` to random sampling.

## Current implementation

The public API is in `sdfmpneo.certification.geometry_mass_residual_domain`:

- `GeometryThermalOperatorBounds`;
- `GeometryJouleDerivativeBounds`;
- `GeometryMassResidualPhysicalProof`;
- `compose_geometry_mass_residual_physical_proof`;
- `bound_geometry_mass_residual_on_box`;
- `certify_geometry_mass_residual_domain`.

The public branch coordinates are physical `[a0, G, U, t]`. The implementation converts physical geometry to the normalized graph coordinates internally and applies the exact derivative scale factor.

## What is already rigorous and what remains

The branch-and-bound logic, analytic DAG derivative enclosure, mass-to-vector residual conversion, and fail-closed proof composition are implemented.

The production UWPT model still needs theorem-derived branch providers for two physical pieces before a full 10-D geometry certificate can close automatically:

1. spectral-norm bounds for `M_r(G)`, `K_r(G)` and their geometry derivatives on each branch;
2. continuous reduced-EM/Joule bounds for `dq/da`, explicit `dq/dG`, and explicit `dq/dU` on the induced thermal-state domain.

These must come from geometry/energy/constitutive analysis, not finite differences or sampled maxima. Until both providers mark their results certified, the domain result remains `indeterminate`.

The existing whole-box P1 mass/stiffness form ratios and physical-energy EM machinery are the intended ingredients for these providers.

## Finite time and steady state

The first implementation certifies finite intervals

\[
0\le t_{\min}\le t\le t_{\max}<\infty.
\]

It intentionally rejects `time_upper=inf`. The exact steady-state query `t=+inf` is a separate proof obligation and should receive its own stationary branch certificate rather than being silently identified with a very large finite time.

## Relationship to training

The desired long-term workflow is

\[
\text{residual-grown training}
\rightarrow
\text{continuous-domain certification}
\rightarrow
\text{refine only an unresolved/worst proof box if necessary}.
\]

The current worst-residual exchange search remains useful as a numerical counterexample finder while the complete physical proof providers are being finished. It is evidence and training guidance, not the final certification mechanism.

## Claim language

A suitable final positioning is:

> SDF-MPNEO replaces test-set statistical generalization as the final validity criterion with continuous physical-domain residual certification. Training remains numerical, while the accepted surrogate is required to satisfy a deterministic governing-residual bound over the declared parameter domain.

Do not claim a full-domain UWPT residual certificate until all unresolved boxes are zero and every physical proof component is certified.
