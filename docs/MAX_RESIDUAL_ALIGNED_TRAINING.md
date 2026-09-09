# Max-residual-aligned training

SDF-MPNEO declares numerical convergence only when both the persistent training set and the independent residual search satisfy

\[
\max_i \|R_i\|_2 \le \varepsilon,
\qquad \varepsilon = 10^{-5}\ \text{by default}.
\]

Historically, however, joint weight refinement and candidate scoring minimized the unweighted mean-square residual.  Late in training this can spend most wall-clock time reducing already-small residuals while a few hard points remain several times above the maximum-residual tolerance.

The max-aligned runtime keeps the same physics, collocation points, analytic response dictionary and final tolerance, but changes the numerical optimization merit so it is consistent with the stopping criterion.

## Hard-point IRLS

For point residual norms \(r_i=\|R_i\|_2\), define the tolerance excess

\[
e_i = \max(r_i-\varepsilon,0),\qquad e_{\max}=\max_i e_i.
\]

If \(e_{\max}>0\), the frozen IRLS weight is

\[
w_i=1+24\left(\frac{e_i}{e_{\max}}\right)^2.
\]

Thus points already below tolerance keep weight one, while the worst currently violating point receives weight 25.  The Gauss--Newton subproblem is

\[
\min_\delta \sum_i w_i\|R_i+J_i\delta\|_2^2.
\]

The same weights are used in analytic candidate correlation and tangent-norm scoring, so topology growth and joint weight refinement optimize the same hard-point-focused merit.

## Max-first line search

A trial update is not accepted merely because the average MSE decreases.  The maximum residual is the primary merit: a strict decrease in the current maximum is accepted (subject to the residual-validity checks), while a numerically unchanged maximum requires a decrease in the frozen weighted least-squares merit.  The implementation also prevents a point that was already below the final tolerance at the beginning of the step from being pushed back above it.

The reported MSE/RMS metrics are unchanged and remain useful diagnostics.  They are no longer allowed to hide a stalled worst residual.

## Early handoff to topology growth

Joint weight refinement tracks the actual maximum residual.  If two consecutive accepted Gauss--Newton steps each improve that maximum by less than \(10^{-3}\) relatively, weight refinement stops immediately and returns control to candidate search.  This does not terminate training and does not reduce the candidate dictionary.  It only avoids spending the remaining Gauss--Newton iterations on a topology whose worst residual is effectively stationary.

## Numerical scope

This policy does not change:

- the governing EM/thermal equations;
- the residual tolerance;
- the collocation/search domain;
- `max_nodes`, `max_degree`, `max_parent_responses`, or `max_realization_dimension`;
- the response-node dictionary or long-time/infinity semantics;
- checkpoint/model format;
- independent residual-search convergence checks;
- continuous-domain certification or inference behavior.

The native C++ DAG/Jacobian/Gauss--Newton kernels are still used when admissible; the weighting is applied to their exact batched residual/Jacobian output before the unchanged dense least-squares solve.
