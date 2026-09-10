# Residual-driven automatic analytic-state construction

The parametric analytic DAG uses an explicit dynamic-state interpretation. It does **not** introduce a fixed source count `K`, a hidden width, or a layer count.

For analytic state `nu`,

\[
(\partial_t+\lambda_{j_\nu})h_\nu
=\sum_{s\in\mathcal A_\nu}\omega_{\nu s}\,\psi_{\nu s},
\qquad h_\nu(0)=0.
\]

`A_nu` is an active source set constructed during training. Its cardinality is not configured. The fixed physical decay `lambda_j`, zero response initial condition, governing EM/thermal field and residual tolerance remain unchanged.

## Dynamic family

A newly aggregated state has one target thermal mode and at most one response-state parent shared by all active source columns. Initial-coordinate decays and declared static geometry/current factors may differ between columns. Thus a dynamic family is identified by

\[
(j_\nu,p_\nu),
\]

where `p_nu=None` denotes a direct response with no response-state parent.

The normal dictionary uses the configured `max_degree`, `max_parent_responses` and `max_realization_dimension`. `max_nodes` remains a budget on **independent dynamic states**. None of these quantities acts as a source-count limit inside one state.

For backward compatibility, a historical **single-source** node created under `max_parent_responses > 1` may still contain several response parents. Such a source remains evaluable, cloneable and loadable. The same representation is used by the selective convergence-rescue path for two-response interaction states; it is deliberately not aggregated into a multi-source dynamic family.

## The three structural actions

### Enrich

If the residual solve activates a source in the same `(target mode, response-state parent)` family as an existing state, add the column to that state's active source set and jointly re-optimize all active source coefficients.

There is no `K_max`. Enrichment stops only because no admissible inactive source produces an accepted max-residual decrease, because numerical convergence is reached, or because another structural action is required.

### Grow

If an activated source belongs to a dynamic family that does not yet exist, create that state. Several selected source columns in the same new family are aggregated into one state, so source cardinality does not consume independent-state budget.

### Split

An aggregated state can later need independent downstream addressability. Split selects one active source and exposes it as a new state. A naive local split would change all existing descendants, so the implementation recursively duplicates every affected downstream state along the split branch.

Under the production one-dynamic-parent family rule this is exactly the distributive identity

\[
m(h_a+h_b)=mh_a+mh_b,
\]

followed through the response DAG. Therefore Split itself is a function-preserving reparameterization. Only the subsequently added residual-driven column changes the represented map.

Split is now a low-frequency topology repair: normal Enrich/Grow selection is handled by the block sparse solve. Exact screened Split search is invoked only when the block proposal cannot pass the true nonlinear residual check.

## Continuous parameters

Every active source coefficient `omega_(nu,s)` is a continuous optimization variable. Max-residual-aligned IRLS Gauss--Newton operates on the flattened active source-coefficient vector,

\[
\omega=(\omega_{1,1},\omega_{1,2},\ldots).
\]

Exact realization automatic differentiation computes `J_a` and `J_da` with respect to every active source coefficient. No finite-difference training Jacobian is used.

The existing native C++ DAG/GN path remains enabled while the graph is still one-source-per-state. A coalesced independent equation seed is transiently scalarized to an exactly equivalent graph for native GN. Once true multi-source dynamic descendants appear, evaluation falls back to the exact source-aware realization path until a native multi-source kernel is added.

## Matrix-free block sparse structure optimization

The training hot path no longer performs normal structure growth by greedily sending one candidate after another through the full nonlinear EM-thermal model.

At the current collocation states the physical field and its Jacobian are already known. For a candidate source column, the residual linearization is

\[
\delta R = \delta\dot a-J_F\,\delta a.
\]

Candidates sharing one dynamic response key reuse the same analytic unit-response tangent. Static initial/geometry/current monomials enter only as amplitude scale vectors. The implementation therefore represents the whole candidate dictionary as a matrix-free operator: it stores compact scale matrices and shared dynamic tangent tables rather than materializing one dense `(points*modes) x candidates` matrix.

Source selection is performed jointly with a sparse-group proximal solve. In normalized coordinates `z`, the local problem is a hard-point-weighted least-squares model with both source sparsity and dynamic-family group sparsity,

\[
\min_z
\frac12\|W^{1/2}(R+Tz)\|_2^2
+\lambda_1\|z\|_1
+\lambda_g\sum_f\sqrt{|f|}\,\|z_f\|_2.
\]

A short decreasing regularization path is solved by matrix-free FISTA. Candidate columns are normalized before the sparse solve, so regularization is not biased toward arbitrary column scale. `max_nodes` is enforced on newly activated dynamic families, not on the number of active source columns.

The resulting joint direction is then rescaled by the existing hard-point linearized `L-infinity` minimax rule. Only source coefficients that remain nonzero after sparse optimization are materialized as one Enrich/Grow block.

## True nonlinear acceptance and fail-closed fallback

The sparse problem is only a structure proposal mechanism. It does **not** replace the governing physics or relax convergence.

The complete proposed block is evaluated against the true nonlinear collocation residual with the same max-first acceptance rule and the same configured `residual_tolerance`. The block line search performs only a small bounded number of full physical evaluations. If the predicted block fails, training falls back to the previous exact max-aligned scalar candidate path, including screened function-preserving Split proposals. Thus the fast path can fail closed without removing legacy capability.

Candidates whose direct analytic realization exceeds the normal `max_realization_dimension` are excluded from the normal block solve. They remain reachable through Split when that makes the downstream realization admissible, or through the bounded convergence-rescue allowance described below.

This changes the dominant cost from

\[
N_{candidate}\times N_{nonlinear\ validation}
\]

toward

\[
N_{shared\ tangent\ keys}+N_{matrix\!\!-\!vector\ iterations}
+N_{block\ nonlinear\ validation},
\]

so a larger admissible dictionary mainly widens cheap matrix-vector work instead of multiplying full EM-thermal solves.

## Automatic convergence rescue

A structural stall above the requested residual tolerance is no longer treated as a completed training result immediately.

The normal path first exhausts the configured one-response-parent block dictionary and its exact scalar/Split fallback. If no accepted structure remains, a small rescue dictionary is formed from at most eight representative response states. It contains only square and cross products of two response states,

\[
h_i^2,\qquad h_i h_j,
\]

which supply the second-order thermal-state interactions that the normal one-response-parent production dictionary cannot represent directly. These rescue candidates are again solved jointly as a sparse block, at most four sources are proposed, and the same exact nonlinear EM-thermal residual decides acceptance.

The rescue path permits an exact realization dimension up to 512 so the product of two aggregated seed states is not discarded merely because the normal fast-path budget is 64. It does not raise `max_nodes`; independent-state capacity remains a hard budget.

If the complete normal-plus-interaction search still returns `stalled`, training automatically continues twice. Each continuation adds one static polynomial degree and keeps the rescue realization allowance, so a default degree-3 run attempts degree 4 and then degree 5 before returning a genuine structural stall. The physical equations, collocation domain and requested residual tolerance are unchanged throughout these continuation passes.

This policy intentionally distinguishes **search budgets** from the **success criterion**: search complexity may be expanded automatically after a proven stall, but numerical success is reported only when both training and independent validation residuals satisfy the configured tolerance.

## Geometry seed

Geometry equation seeding predates multi-source analytic states and internally emits scalar source columns. The runtime temporarily decouples this legacy scalar source capacity from `max_nodes`, preserves the existing residual-based seed selection, and then function-preservingly coalesces all same-target independent seed columns into their dynamic families before residual-driven training continues.

## Persistence and compatibility

The public `ParametricAnalyticEvolutionGraph` type and `add_product_response()` API remain valid. A legacy scalar response is represented internally as an analytic state with one active source.

Fixed-geometry model files containing multi-source states use research model format version 2 for graph metadata. Version 1 files remain loadable. Geometry-family checkpoints keep their outer format and embed the upgraded reference checkpoint, so old geometry checkpoints also remain loadable. Regression coverage exercises both fixed-geometry and geometry-family save/load paths with true multi-source states.

## What is intentionally unchanged

This architecture does not change:

- residual tolerance (`1e-5` in the current UWPT configuration);
- physical EM or thermal equations;
- collocation and independent residual-search domains;
- `max_nodes` as the hard independent-dynamic-state budget;
- long-time / stationary semantics;
- continuous-domain certification target;
- inference inputs or the physical thermal spectrum.

The configured `max_degree`, `max_parent_responses` and `max_realization_dimension` define the **normal** search dictionary. After a proven stall, the convergence-rescue path may temporarily use two response parents, realization dimension 512, and up to two additional static degrees. Those allowances affect search expressivity only; they do not relax the governing residual or the success criterion.

The structural change is that admissible source columns are selected jointly by a sparse continuous optimization, while the true nonlinear governing residual remains the final acceptance criterion.
