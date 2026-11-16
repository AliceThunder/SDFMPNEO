# GeomFracET V6 ? conservative package and seawater thermal feedback

## Status and primary objective

The frozen V5 physical acceptance used five unseen Q01-Q05 BFZI V150
S5 geometries (15 complete histories) and achieved 8.322% mean
autonomous dynamic port-Z relative L2. Its 3-D temperature field
still exhibits package and localized seawater errors on high-power
long-duration cases. Q01 TX package 600s has 12.88 K max error but
1.26 K material-mean error; Q03 TX package has 16.17 K max error,
7.69 K material-mean error and 12.05 K TX copper error.

## Package spatial-state correction

A geometry-conditioned 14-output small MLP learns corrections to
the 7+7 *zero-material-mean package thermal shape modes*. Its
inputs are the frozen 32-mode neural thermal state and the 45
BFZI geometry/material features, and it is exactly zero at ambient.
Training uses only 30 original V4 geometries with 90 teacher
histories, with five T04/D01-D04 geometries for model selection.

The package correction is an *absolute target relative to the
uncorrected neural reference trajectory*, not an increment to
accumulate each time step. This distinction prevents recurrent
temperature drift. The corrected package modes are used in the
thermal state at each time step, followed by the same five-material
nonlinear energy-budget closure. Source, current, seawater impedance
and heat evolution are recomputed on the corrected state.

After 300 optimization epochs, the selected package model's
autonomous mean package maximum error on development trajectories
improves from 1.601 K to 0.858 K. The mean *total 3-D* maximum
temperature error decreases from 1.898 K to 1.683 K and mean dynamic
port-Z error from 8.367% to 6.913%. Package correction incurs one
additional cheap neural reference rollout per input history,
without online BFZI EM solves or implicit thermal PDE solves.

## Self-consistent seawater skin lifting

The 16 optional sea-skin modes are an additive spatial thermal
correction, linear in the base neural thermal state. A corrected
effective spatial basis is constructed once per new geometry:
`Phi_eff = Phi_32 + Phi_skin S W^T / norm`.
The new modes have zero capacity-weighted mean in each material.
The effective basis enters the spatial temperature reconstruction,
diffusion flux, thermal source, physical circuit and energy-budget
Jacobian. It preserves the original five material-energy balances.

The previously trained neural sea-skin decoder was optimized for
temperature reconstruction. A fractional feedback gain avoids
over-correcting the nonlinear thermal evolution. On the same 15
development histories, the following metrics were measured:

| Configuration | Dynamic port-Z relative L2 | Mean maximum 3-D temperature error |
|---|---:|---:|
| Frozen V5 | 8.367% | 1.898 K |
| Package correction only | 6.913% | 1.683 K |
| Package + 20% sea-skin feedback | 7.516% | 1.551 K |
| Package + 40% sea-skin feedback | 8.003% | 1.466 K |
| Package + 60% sea-skin feedback | 8.374% | 1.414 K |
| Package + 100% sea-skin feedback | 9.201% | 1.319 K |

All tested configurations retained maximum five-material energy
budget residuals of order 1e-8 W. The **20% feedback configuration**
is the V6 primary coupled model because it improves both targets
relative to V5 with a balanced result. Package-only is the
impedance-accuracy-first alternative. The configuration is frozen
in `runs/geomfrac_et_v6_selection.json`, before any R01-R03
result is used.

## Independent assessment and usage

V6 results on the previously examined Q01-Q05 set (package-only)
are retrospective diagnostics: dynamic-Z 8.322% -> 7.566% and
mean 3-D temperature max 4.355 -> 3.169 K on 15 trajectories.
They do not constitute a new blind V6 acceptance.

Three new R01-R03 geometries are frozen in
`runs/geomfrac_s5_v6_plan.json`, with the same 60/300/600-second
physical teacher design. Their final physical-teacher evaluation
is in progress. Do not use R cases to choose the V6 architecture,
training epochs or feedback gain.

V6 autonomous inference uses the V5 SOE/heat checkpoint plus the
learned thermal-impedance tangent, 14-mode package MLP, and optional
sea-skin decoder, all without online full BFZI Green solves or
implicit FV heat solves. Standalone inference example:

```powershell
python -m examples.bfzi_geom_neural_infer --anchor runs/bfzi_s5_Q03_ultra_v5_anchor.npz --model runs/geomfrac_et_v4_8g_quality.pt --tangent-model runs/geomfrac_et_v5_tangent_aware.pt --pkg-model runs/geomfrac_et_v6_pkg.pt --lift-model runs/geomfrac_et_v5_lift.npz --lift-feedback --lift-gain 0.2 --horizon 600 --heat 240 --cool 180 --step 15 --voltage 75 --out runs/geomfrac_et_v6_infer.npz
```

For package-only mode, omit the two lift options. Model fitting:
`python -m examples.bfzi_geom_pkg --help`.

## References

- `runs/geomfrac_et_v6_pkg_dev.json`: learned package MLP result.
- `runs/geomfrac_et_v6_pkg_dev_trial.json`: coupled package-only
  autonomous developer metrics.
- `runs/geomfrac_et_v6_coupled_g02.json`: primary V6 developer metrics.
- `runs/geomfrac_et_v6_selection.json`: frozen configuration.
- `runs/geomfrac_s5_v6_plan.json`: frozen new-test geometry cohort.

## Frozen R01-R03 independent physics acceptance

Three geometry-disjoint BFZI V150 S5 teachers, nine trajectories; frozen package and 20% sea-feedback weights chosen on T04/D01-D04, without using R cases.

| Surrogate | Mean dynamic Z rel L2 | Mean max 3D T error | Mean CPU per trajectory | Max energy budget residual |
|---|---:|---:|---:|---:|
| V5 | 10.137% | 1.664 K | 0.285 s | 9.22e-09 W |
| V6_package | 7.725% | 1.653 K | 0.672 s | 9.88e-09 W |
| V6_coupled | 7.738% | 1.388 K | 0.658 s | 9.91e-09 W |

All values are independent of the V4 training geometries and T04/D01-D04 developer geometry set. Full trajectory results: `runs/geomfrac_et_v6_final.json`.

### Independent closure and remaining error by physical region

The 9-history R01-R03 frozen independent acceptance is complete and
committed (R01-R03 all 60/300/600s physical traces; each physics
status is candidate). For each geometry the three time histories'
mean errors are:

| Geometry | V5 dynamic-Z error | V6 package-only | V6 coupled (20%) | V5 mean max 3-D temp | V6 coupled |
|---|---:|---:|---:|---:|---:|
| R01 | 14.636% | 10.453% | 10.403% | 1.382 K | 1.017 K |
| R02 | 9.172% | 7.268% | 7.244% | 3.096 K | 2.818 K |
| R03 | 6.604% | 5.455% | 5.568% | 0.513 K | 0.328 K |

The coupled V6 independent mean dynamic-Z relative L2 is
7.738% versus 10.137% V5 (23.67% relative improvement), and
the mean per-history maximum 3-D temperature error is 1.388 K
versus 1.664 K (16.58% relative improvement). The 9-history
worst dynamic-Z error decreases from 19.244% to 12.917%.
The maximum material energy-budget residual is 9.91e-9 W.
Average measured CPU inference time per history is 0.658 s
versus 0.285 s for the original V5; the V6 package baseline
trajectory accounts for much of this extra work.

The 600-second R02 case is the dominant remaining thermal
accuracy limitation. Its transmitting-package adjacent seawater
skin maximum errors are 5.533 K for V5 and 5.642 K for
the coupled V6, while the original 32-mode capacity-weighted
projection of the true field gives 2.808 K in that region.
Meanwhile R02 TX package maximum error decreases from
3.214 to 1.777 K and RX package 2.785 to 0.687 K.
The V6 package correction is successful; the sea-skin state
evolution and feedback remain the next high-value improvement.
Future V7 experiments should learn geometry/drive/state-aware
sea-skin modes on the existing 30 training geometries, select
on T04/D01-D04, and evaluate generalization on an additional
frozen test geometry set after architecture selection. The
previously evaluated R01-R03 outcomes must remain evidence
for V6, not become training or model-selection labels.

Per-material diagnostic: runs/geomfrac_et_v6_thermal_breakdown.json.
