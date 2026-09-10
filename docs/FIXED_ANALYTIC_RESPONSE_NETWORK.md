# Fixed analytic response network

Fresh SDF-MPNEO training now uses a **fixed-depth low-rank analytic response network**. The normal training path no longer grows a DAG and no longer performs candidate enumeration, Grow, Enrich or Split.

The physical residual is unchanged:

\[
R(a,\dot a,U)=\dot a-F(a,U).
\]

A run is successful only when the configured maximum residual tolerance is met on both the training set and the independent validation set.

## Fixed topology

Let

\[
z(t)=[1,\;\hat a_{0,1}e^{-\lambda_1t},\ldots,\hat a_{0,n}e^{-\lambda_nt},\;\hat U_1,\ldots,\hat U_d].
\]

The hats denote affine normalization to the configured training box. Each hidden channel is assigned one physical thermal decay and obeys

\[
(\partial_t+\lambda_{j_c})h_c^{(\ell)}=s_c^{(\ell)},
\qquad h_c^{(\ell)}(0)=0.
\]

The first layer uses an affine source plus a shared low-rank quadratic map,

\[
s_c^{(0)}=w_c^Tz+
\sum_{r=1}^{R_q}q_{cr}(u_r^Tz)(v_r^Tz).
\]

Later layers use fixed dense response mixing together with low-rank input-response and response-response interactions,

\[
\begin{aligned}
s_c^{(\ell)}={}&w_c^Th^{(\ell-1)}\\
&+\sum_{r=1}^{R_x}b_{cr}(p_r^Tz)(q_r^Th^{(\ell-1)})\\
&+\sum_{r=1}^{R_h}d_{cr}(s_r^Th^{(\ell-1)})(t_r^Th^{(\ell-1)}).
\end{aligned}
\]

The predicted thermal state is the exact initial decay plus all fixed response channels assigned to each thermal mode.

All coefficients and low-rank factors exist from the beginning of training. Training therefore optimizes one continuous parameter vector; it does not decide which neuron or parent tuple should exist.

## Analytic time representation

The implementation does not turn the fixed network into a large dense state-space matrix. Every signal is represented directly in the closed basis

\[
t^k e^{-\mu t},
\]

where \(\mu\) is an integer combination of the physical thermal decay rates. Addition, multiplication and the response operator \((\partial_t+\lambda)^{-1}\) are closed in this basis. Equal-pole cases use the exact confluent polynomial branch.

Consequences:

- arbitrary finite times are evaluated directly, without time stepping;
- `t=inf` is evaluated as the stationary limit;
- the thermal decay spectrum remains the physical reduced thermal spectrum;
- there is no large matrix exponential created merely because many source terms are present.

## Continuous residual optimization

At each collocation point the network computes

\[
a,\quad \dot a,\quad J_a=\frac{\partial a}{\partial\theta},\quad
J_{\dot a}=\frac{\partial\dot a}{\partial\theta}
\]

analytically. With the physical state Jacobian \(J_F\),

\[
J_R=J_{\dot a}-J_FJ_a.
\]

Training uses hard-point-weighted Levenberg-Marquardt / Gauss-Newton steps with a trust radius and max-residual-first nonlinear acceptance. The expensive electromagnetic-thermal model is evaluated for parameter updates, not for thousands of discrete structural candidates.

If an independent validation point violates the tolerance, it is promoted into the training set and continuous optimization resumes. No candidate-search stage is introduced by this adaptive collocation step.

## Default capacity

For a fresh model the default fixed architecture is:

- depth: 4 response layers;
- channels per thermal mode: 2;
- shared input quadratic rank: 8;
- input-response rank per later layer: 4;
- response-response rank per later layer: 3;
- maximum continuous optimization iterations per validation epoch: 36;
- maximum adaptive validation epochs: 5.

For the current two-mode UWPT configuration with 12 static geometry/current inputs this is about 764 continuous parameters and 16 logical response channels.

These are capacity settings, not searched quantities. Advanced experiments can override them with environment variables:

```text
SDFMPNEO_FIXED_NETWORK_DEPTH
SDFMPNEO_FIXED_NETWORK_CHANNELS_PER_MODE
SDFMPNEO_FIXED_NETWORK_QUADRATIC_RANK
SDFMPNEO_FIXED_NETWORK_CROSS_RANK
SDFMPNEO_FIXED_NETWORK_STATE_RANK
SDFMPNEO_FIXED_NETWORK_MAX_ITERATIONS
SDFMPNEO_FIXED_NETWORK_VALIDATION_EPOCHS
```

The default `run.py` needs no new configuration entries.

## Persistence

Fixed-network research checkpoints use model format version 3 and store the architecture metadata plus the continuous parameter vector. Geometry-family checkpoints embed that upgraded reference checkpoint as before.

Version 1 and version 2 analytic-DAG checkpoints remain loadable. A **non-empty legacy DAG** is deliberately resumed with its historical residual-driven DAG trainer so an old checkpoint is never silently reinterpreted as a different model. Start a fresh training run to use the fixed network.

## Expected training log

For a fresh model the normal phase sequence is now approximately

```text
initial_residual
weight_refinement
weight_refinement
...
validation
saving
```

`candidate_search` should not appear in a fresh fixed-network run. If it appears, the loaded model is a legacy non-empty DAG and the compatibility trainer is active.
