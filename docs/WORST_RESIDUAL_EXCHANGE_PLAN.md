# Worst-Residual Exchange Training Plan

## Motivation

Randomly replacing the validation set changes the observed maximum residual even when the network itself is unchanged. In a high-dimensional continuous parameter box, repeatedly drawing fresh validation batches can therefore lead to many train/validate/add cycles without directly targeting the worst unresolved region.

## Target training loop

Use a residual-exchange loop:

1. Train the current analytic response DAG on a persistent collocation set until its maximum training residual is below the configured tolerance.
2. Search the same admissible parameter box for current residual hotspots.
3. If the strongest discovered residual is below tolerance, declare numerical convergence for this finite search oracle.
4. Otherwise add only the strongest few unique hotspot points to the persistent training set and continue training.

The governing residual tolerance, physics, candidate dictionary and model capacity are unchanged.

## Search oracle

The search is numerical rather than a continuous-domain certificate. It should combine:

- deterministic low-discrepancy exploration over the full normalized parameter box;
- explicit boundary/centre anchors;
- both finite-time and exact steady-state (`t=+inf`) probes when steady-state training is enabled;
- local derivative-free refinement of the best finite-time seeds inside the same box;
- a small fixed exchange batch (for example 4 points) so the training set grows slowly and only with high-value counterexamples.

Finite time should be parameterized in a mixed linear/log coordinate so early and late transients are both searchable.

## Simplicity requirement

The public training state machine remains:

`train -> worst-residual search -> add hotspots -> train`

No guard/fresh-validation hierarchy is required for the exchange path. Existing continuation/checkpoint machinery may keep compatibility fields internally, but the user-facing convergence rule remains one tolerance and one search oracle.

## Claim boundary

Passing the search oracle is a strong numerical worst-case check, not a proof that

`sup_x ||R(x)|| <= tolerance`

over the full continuous parameter box. A future continuous residual/Lipschitz/interval certificate would be required for that claim.
