# Residual-driven automatic analytic-state construction

This experimental branch replaces the one-source/one-neuron interpretation of the parametric analytic DAG with an explicit dynamic-state interpretation. It does **not** introduce a fixed source count `K`, a hidden width, or a layer count.

For analytic state `nu`,

\[
(\partial_t+\lambda_{j_\nu})h_\nu
=\sum_{s\in\mathcal A_\nu}\omega_{\nu s}\,\psi_{\nu s},
\qquad h_\nu(0)=0.
\]

`A_nu` is an active source set constructed during training. Its cardinality is not configured. The fixed physical decay `lambda_j`, zero response initial condition, governing EM/thermal field and residual tolerance remain unchanged.

## Dynamic family

A state has one target thermal mode and at most one response-state parent shared by all active source columns. Initial-coordinate decays and declared static geometry/current factors may differ between columns. Thus a dynamic family is identified by

\[
(j_\nu,p_\nu),
\]

where `p_nu=None` denotes a direct response with no response-state parent.

The existing `max_degree` and `max_realization_dimension` still describe admissible candidate columns. `max_nodes` remains a budget on **independent dynamic states**. None of these quantities acts as a source-count limit inside one state.

## The three structural actions

### Enrich

If the best residual column has the same `(target mode, response-state parent)` family as an existing state, add the column to that state's active source set and jointly re-optimize all active source coefficients.

There is no `K_max`. Enrichment stops only because no admissible inactive column produces an accepted max-residual decrease, because numerical convergence is reached, or because another structural action wins the nonlinear residual comparison.

### Grow

If no existing state has the required dynamic family, create a new state. Growth therefore represents a genuinely new analytic dynamic role rather than a missing coefficient inside an existing role.

### Split

An aggregated state can later need independent downstream addressability. Split selects one active source and exposes it as a new state. A naive local split would change all existing descendants, so the implementation recursively duplicates every affected downstream state along the split branch.

Under the production `max_parent_responses <= 1` rule this is exactly the distributive identity

\[
m(h_a+h_b)=mh_a+mh_b,
\]

followed through the response DAG. Therefore Split itself is a function-preserving reparameterization. Only the subsequently added residual-driven column changes the represented map.

Direct and split candidate branches receive their **own** residual-tangent coefficient before nonlinear line search. Split is not scored using the pre-split aggregate tangent.

## Continuous parameters

Every active source coefficient `omega_(nu,s)` is a continuous optimization variable. Max-residual-aligned IRLS Gauss--Newton operates on the flattened active source-coefficient vector,

\[
\omega=(\omega_{1,1},\omega_{1,2},\ldots).
\]

Exact realization automatic differentiation computes `J_a` and `J_da` with respect to every active source coefficient. No finite-difference training Jacobian is used.

The existing native C++ DAG/GN path remains enabled while the graph is still one-source-per-state. Once a true multi-source state appears, native code fails closed to the exact source-aware realization path until a future native multi-source kernel is added.

## Candidate selection

Hard-point residual weights from the max-residual trainer are shared by source-column scoring and coefficient refinement. A candidate tangent `T_c` is ranked by the weighted projected decrease

\[
\Delta_c = \frac{\langle R,T_c\rangle_W^2}{\langle T_c,T_c\rangle_W}.
\]

For split branches, the graph is first split exactly, then that branch's own tangent and optimal initial coefficient are recomputed. Enrich/Grow/Split alternatives are finally compared using the true nonlinear collocation residual with the existing max-first acceptance rule.

## Persistence and compatibility

The public `ParametricAnalyticEvolutionGraph` type and `add_product_response()` API remain valid. A legacy scalar response is represented internally as an analytic state with one active source.

Fixed-geometry model files containing multi-source states use research model format version 2 for graph metadata. Version 1 files remain loadable. Geometry-family checkpoints keep their outer format and embed the upgraded reference checkpoint, so old geometry checkpoints also remain loadable.

## What is intentionally unchanged

This branch does not change:

- residual tolerance (`1e-5` in the current UWPT configuration);
- physical EM or thermal equations;
- collocation and independent residual-search domains;
- `max_nodes`, `max_degree`, `max_parent_responses` or `max_realization_dimension`;
- long-time / stationary semantics;
- continuous-domain certification target;
- inference inputs or the physical thermal spectrum.

The architectural change is solely that **source columns and independent analytic dynamic states are no longer forced to be the same object**.
