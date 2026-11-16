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

## Frozen V5 tangent blending check

The initial physical-temperature tangent and the autonomous-aware tangent were compared at fixed convex mixing weights.

| Initial tangent weight | Mean autonomous dynamic-Z error | Worst history |
|---:|---:|---:|
| 0.00 | 8.367% | 22.392% |
| 0.25 | 8.630% | 20.236% |
| 0.50 | 9.097% | 22.168% |
| 0.75 | 9.741% | 24.181% |
| 1.00 | 10.572% | 26.255% |

The autonomous-aware tangent alone retains the lowest mean development error and is frozen for the new Q01-Q05 acceptance.

## V5 model frozen before independent testing

A second tangent self-consistency fitting iteration retained its starting checkpoint. Its 15-history development error remained 8.367%. The selected V5 checkpoint is the original autonomous-state-aware tangent; models and acceptance geometry IDs are frozen in `runs/geomfrac_et_v5_selection.json` before any Q01-Q05 evaluation.

## Independent Q01-Q05 geometry coverage (before test evaluation)

The five frozen final cases vary circular/rounded-square shape
parameters, coil separation and orientation, package dimensions,
insulation and encapsulation thermal conductivities and package/sea
heat transfer. Four of the five contain at least one scalar input
outside the observed minimum/maximum of the 30 training geometries:
Q01 package/sea convection coefficient, Q02 receiver package size,
Q03 package conductivity, and Q05 yaw. Q04 is in the observed
scalar coordinate ranges. All five belong to the original validated
BFZI design domain and have passed native geometry preflight.
This coverage statement uses the frozen geometry/thermal plan
only; no final-test prediction errors were consulted.

## Independent frozen Q01-Q05 physical acceptance

The neural thermal checkpoint and autonomous-state-conditioned thermal tangent were frozen based on T04/D01-D04 development data before evaluating Q01-Q05.
Five disjoint BFZI S5 geometries, 15 unseen 60/300/600s trajectories.

Mean autonomous dynamic-Z relative L2 8.322%; maximum 14.438%.

| History | Dynamic Z rel L2 | Max 3D T error |
|---|---:|---:|
| bfzi_s5_Q01_ultra_v5_drive1 | 8.224% | 1.383 K |
| bfzi_s5_Q01_ultra_v5_drive2 | 11.494% | 6.329 K |
| bfzi_s5_Q01_ultra_v5_drive3 | 14.438% | 12.883 K |
| bfzi_s5_Q02_ultra_v5_drive1 | 6.323% | 0.242 K |
| bfzi_s5_Q02_ultra_v5_drive2 | 6.179% | 0.551 K |
| bfzi_s5_Q02_ultra_v5_drive3 | 6.382% | 1.198 K |
| bfzi_s5_Q03_ultra_v5_drive1 | 7.791% | 3.979 K |
| bfzi_s5_Q03_ultra_v5_drive2 | 10.454% | 10.019 K |
| bfzi_s5_Q03_ultra_v5_drive3 | 12.532% | 22.866 K |
| bfzi_s5_Q04_ultra_v5_drive1 | 5.081% | 0.146 K |
| bfzi_s5_Q04_ultra_v5_drive2 | 3.419% | 0.777 K |
| bfzi_s5_Q04_ultra_v5_drive3 | 3.691% | 2.505 K |
| bfzi_s5_Q05_ultra_v5_drive1 | 6.564% | 0.177 K |
| bfzi_s5_Q05_ultra_v5_drive2 | 10.072% | 0.807 K |
| bfzi_s5_Q05_ultra_v5_drive3 | 12.186% | 1.464 K |

## V4 versus frozen V5 on the same unseen Q01-Q05 trajectories

V4 independent Q test average dynamic-Z relative L2 = 18.035%; frozen V5 = 8.322%; relative reduction = 53.86%.

| Geometry | V4 mean dynamic-Z rel L2 | V5 mean dynamic-Z rel L2 |
|---|---:|---:|
| Q01 | 22.734% | 11.385% |
| Q02 | 15.872% | 6.295% |
| Q03 | 21.540% | 10.259% |
| Q04 | 13.348% | 4.063% |
| Q05 | 16.679% | 9.608% |

## V5 thermal basis extension: 32 to 48 shared modes

The largest spatial temperature reconstruction errors in the frozen
Q01 and Q03 independent V5 acceptance occur in the TX package's
adjacent seawater skin. A 48-mode *optional experimental* canonical
basis preserves the first 32 modes, adding eight package-local
seawater-skin shapes around each TX/RX enclosure. The added shapes
depend on each enclosure's normalized local coordinates; their
meaning is consistent across package sizes and coil positions.
All 48 modes remain heat-capacity orthonormal. The original V5
production checkpoint and its frozen Q01-Q05 acceptance are unchanged.

Projected BFZI teacher temperature fields were compared at 600 s
for the five development geometries:
mean maximum abs reconstruction error 2.541 K (32 modes) to
1.151 K (48 modes). The previously examined Q01-Q05 cases yield
6.119 K to 2.787 K retrospectively, with Q01 11.089 -> 4.630 K and
Q03 16.569 -> 8.157 K. These values measure direct *projection of
physical truth*; they are not neural temperature-prediction results.
The Q dataset has now been examined for representation design and
will not serve as a new blind test for an eventual 48-mode model.

A minimal training path expands the existing neural heat state from
32 to 48 modes, copying the original 32-state heat dynamics and
zero-initializing the 16 new thermal outputs. It freezes pretrained
SOE impedance weights and selects a new checkpoint by the mean
autonomous 3-D temperature maximum error on T04/D01-D04. The
geometry-conditioned neural impedance tangent is reused without
retraining. The 48-mode training outcome must be compared against
the frozen V5 32-mode checkpoint before any new final acceptance.

Projection evidence: `runs/geomfrac_s5_v5_basis48_dev.json`,
`runs/geomfrac_s5_v5_basis48_qdiag.json`.

## V5 48-mode neural thermal trial and optional sea-skin lifting

A warm-started rank-48 GeomFracET neural heat head was trained for
70 epochs on all 30 training geometries. Autonomous maximum 3-D
temperature error on the 15 development histories selected the
epoch-zero checkpoint, with 1.898 K mean error. The extra 16
thermal states therefore need a training method better aligned to
the autonomous heat trajectory; simply increasing the rank of the
existing recurrent heat head did not improve generalization.

A small auxiliary 14x16 ridge map now learns the additional local
sea-skin mode coefficients from the **autonomous** 32-dimensional
neural temperature coordinates. It uses 30 training geometries,
90 BFZI teacher histories and the frozen autonomous thermal-Z
tangent. This map changes only spatial seawater reconstruction:
all five heat-capacity-weighted material means remain unchanged,
and copper/package temperatures are preserved.

The optional temperature output `temp_lift_K` is deliberately
separate from the original `temp_K` and is not fed back into the
Joule/circuit/time-evolution solver. Consequently the original V5
energy-budget physics and independently verified 8.322% dynamic-Z
test performance remain unchanged. This is a *thermal field
reconstruction* enhancement, not a newly validated coupled
physical model.

Across 15 development trajectories, the mean maximum 3-D
temperature absolute error decreases from 1.898 to 1.607 K.
On the previously examined Q01-Q05 cohort (retrospective
diagnostic), it decreases from 4.355 to 3.775 K; the Q03
600-second value decreases from 22.866 to 18.19 K. Since these
Q results were already examined during model development,
a further independently frozen cohort will be required to
claim fresh generalization for the lifting correction.

Fit the lifting map with:

```powershell
python -m examples.bfzi_geom_lift --manifest runs/geomfrac_s5_v4_30g_manifest.json --model runs/geomfrac_et_v4_8g_quality.pt --tangent-model runs/geomfrac_et_v5_tangent_aware.pt --out runs/geomfrac_et_v5_lift.npz --report runs/geomfrac_et_v5_lift_dev.json
```

Add `--lift-model runs/geomfrac_et_v5_lift.npz` to
`examples.bfzi_geom_neural_infer` to save both `temp_K`
and `temp_lift_K` for the same autonomous inference.
The output retains original physically coupled fields. Input
normalization and neural SOE network are unchanged.

Evidence: `runs/geomfrac_et_v5_heat48.json`,
`runs/geomfrac_et_v5_lift_dev.json`,
`runs/geomfrac_et_v5_lift_qdiag.json`.
