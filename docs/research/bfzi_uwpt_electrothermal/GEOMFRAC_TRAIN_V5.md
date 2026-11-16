# GeomFracET V5 ? heat-conditioned seawater impedance

## Motivation and frozen reference

The completed V4 neural surrogate attained 16.315% mean autonomous
dynamic-Z error on 15 development traces (T04/D01?D04), and 18.464%
on the 15 E01?E05 final-test histories. E01?E05 are a *historical V4
test cohort*, not a fresh independent V5 test. V4 low-projection
8g checkpoint `runs/geomfrac_et_v4_8g_quality.pt` is the warm start.

The 32-mode thermal model and SOE fractional impedance predictor
are retained. The original seawater-Z predictor has no direct input
from thermal-state coordinates; it sees only geometry and dissipative
excitation history. A teacher-forced T04/D01 probe still yielded a
16.660% dynamic-Z error, motivating direct state conditioning.

## New model

`GeomFracET(heat_z=True)` includes a six-output residual head
that receives the SOE/geometry latent features and the 32 canonical
thermal coordinates divided by their fixed training scales.
The head predicts a *correction* to the existing six-component
dynamic seawater impedance:

```
delta_z = head(latent, thermal_state) - head(latent, zero_state)
z_sea = z_sea_V4 + delta_z
```

The correction is exactly zero at ambient temperature. Its output
layer starts at zero, reproducing the original V4 checkpoint before
V5 fitting. The correction is evaluated after the neural heat
substep, so both teacher-forced training and autonomous inference
use their interval-end temperature estimates with the same causal
SOE integration. Existing checkpoints retain `heat_z=False`.

`--reuse-scales` preserves the original source/output/thermal
normalization, and `--head-only` freezes all pretrained SOE/heat
weights for a controlled test of the added thermal-Z path.
The first V5 trial uses the V4 low-projection eight-geometry subset
(24 histories) for training and all 5 development geometries
(15 histories) for autonomous model selection, with learning rate
3e-4, dynamic-Z loss weight 2, checkpoint criterion mean autonomous
dynamic-Z relative L2 and a preserved epoch-zero V4 baseline.

## Acceptance

Compare V4 and V5 on all 15 T04/D01?D04 histories. Evaluate on
a newly frozen, geometry-disjoint V5 test cohort once the network
is chosen; do not reuse E01?E05 as a fresh hold-out after having
examined their V4 error details. Report port-Z, maximum spatial
temperature error and inference cost. The primary dynamic-Z target
is below 10% across the new independent cohort.

## Geometry-conditioned sea thermal tangent and closed-loop results

A second V5 path learns a geometry-conditioned six-component
seawater thermal susceptibility from the BFZI cold-anchor geometry
and material features. The learned 6x14 matrix acts on the
canonical seawater temperature mean and 13 seawater spatial
modes. The prediction is exactly zero at ambient temperature.
Copper effects remain analytic in the original circuit model.

The tangent was trained against 30 independent BFZI V150 S5
teacher geometries with 90 histories and spatial-projection-quality
weights. It achieves 2.554% mean dynamic-Z relative L2 on
the 15 T04/D01-D04 validation histories when supplied the
physical teacher temperature (teacher-forced diagnostic).

A fully autonomous test attaches this tangent to the best V4
neural heat model. The tangent is evaluated inside every
five-material energy-balance correction, coupling sea impedance
to temperature, circuit current and Joule heating self-consistently.
The 15-history autonomous mean dynamic-Z error drops from
16.315% (V4) to 10.572% (V5 tangent).
Maximum material budget residual 9.75e-09 W; mean CPU 0.367 s per trajectory.
Tangent teacher-forced and autonomous results: `runs/geomfrac_et_v5_tangent.json`, `runs/geomfrac_et_v5_tangent_dev.json`.
Main remaining error: D04 600s = 26.25%; next improve neural thermal evolution.

## Next optimization: neural heat evolution with frozen learned tangent

The V5 training CLI now accepts `--heat-only --tangent-model <path>`.
It freezes the original SOE impedance trunk and updates the 32-state
neural thermal head against physical temperature/heat residuals.
Validation checkpoint selection uses actual autonomous impedance
with the trained, frozen geometry-conditioned thermal tangent,
rather than the less accurate old SOE seawater-Z head.
The first thermal refinement uses 30 training geometries with
projection-aware history sampling and all 5 development geometries
for selection. It retains the 10.572% pre-training checkpoint
unless the new network improves the autonomous metric.

Standalone inference and independent evaluation can load both
checkpoints:
```powershell
python -m examples.bfzi_geom_eval --model runs/geomfrac_et_v4_8g_quality.pt --tangent-model runs/geomfrac_et_v5_tangent.pt --manifest runs/geomfrac_s5_v4_30g_manifest.json --split val --out runs/geomfrac_et_v5_tangent_dev.json
```

## Completed thermal-conditioned V5 impedance head

The controlled V5 head-only ablation completed 220 training
epochs, starting from the best V4 8g model. The five-development-
geometry, 15-history mean autonomous dynamic-Z relative L2
decreased from 16.315% to 12.7635%. The underlying SOE trunk
and neural 32-mode heat head were frozen, so the gain came
from making impedance directly depend on thermal state.

The geometry-conditioned physical tangent is the preferred
electromagnetic branch on the same development cohort:
10.572% autonomous dynamic-Z mean and 2.554% when given true
teacher thermal states. Subsequent work focuses on refining
the neural thermal head while freezing the learned tangent.

## Autonomous-state-aware impedance tangent improves below ten percent

A second geometry-conditioned tangent was initialized from the
physical-thermal-state tangent, then fine-tuned against a blend of
the physical teacher and the actual autonomous neural heat states
generated using the V4 best neural heat checkpoint with the first
tangent. Training used all 30 geometries, 90 physical histories,
and a physical-teacher mixing weight of 0.25, retaining canonical
seawater modes and the cold-zero thermal response. The model
was selected using only T04/D01-D04 development geometry data.

The teacher-forced, fixed-state development score for this tangent
is 5.458%. The full 15-history autonomous electrothermal feedback
evaluation achieves 8.3667% mean dynamic-Z relative L2; V4
baseline was 16.315%, and the original tangent-coupled V5 was
10.572%. All three comparisons use the same geometry-disjoint
development traces. Material budgets are conserved inside the
self-consistent circuit/heat correction.
Mean CPU inference per history 0.332 s; maximum material budget residual 9.75e-09 W.
The 600s T04 and D04 histories remain largest (22.39% and 18.41%).
Complete development results: `runs/geomfrac_et_v5_tangent_aware_dev.json`.
The new Q01-Q05 geometry test cohort was frozen separately; it has not influenced the model.
