# GeomFracET V9: nonlinear electromagnetic temperature-susceptibility correction

## Goal and unchanged physical backbone

The frozen V7 BFZI V150 S5 independent acceptance on S01?S04
(12 time histories) has 7.890% mean dynamic port-Z relative L2
and 2.597 K mean maximum 3-D temperature error. A challenging
S04 600-second weak-response history has 23.128% dynamic-Z
relative error, while its *full* port-Z relative L2 is only 0.0028%.

V9 retains the trained GeomFracET fractional SOE memory,
the 32-state thermal evolution, V6 package correction,
the five-material conservative coupled physical solver,
the 20% sea-skin feedback, and the V7 seawater temperature
output decoder. It adds a lightweight **nonlinear residual
sea-impedance tangent** on the existing geometry-conditioned
linear sea-Z thermal susceptibility. This is the most direct
modification of the weak dynamic-Z prediction mechanism.

## Model

Baseline: geometry-conditioned linear seawater tangent on
14 canonical seawater thermal coordinates.

V9 addition: 45 geometry/material features, 14 normalized
seawater thermal coordinates and their quadratic features
enter a two-hidden-layer 64-unit neural residual output head,
producing six real/imaginary dynamic seawater port-Z terms.
Subtracting the network output at zero thermal state enforces
an exact zero-temperature-rise correction for every geometry.
The baseline tangent remains frozen.

The resulting sea-Z susceptibility is used at every time
step in the existing coupled nonlinear circuit, heat-source
and five-material energy-balance calculation. It is not a
postprocessed impedance adjustment: updated sea-Z affects
circuit current, Joule loss and the next thermal state.
The frozen V6 package and V7 output modules are unchanged.

Training uses 30 original BFZI V150 S5 geometries and 90
histories with the *autonomous V6 neural thermal states* as
inputs and the physical BFZI sea impedance as targets.
The dynamic-Z error loss normalizes each complete teacher
history by its physical dynamic impedance variation,
with a small residual-amplitude penalty. Checkpoint
selection uses autonomous V6/V9 closed-loop dynamic-Z mean
relative L2 on the untouched original development set:
five geometries T04/D01?D04, 15 time histories.

## Developer results and frozen selection

| Metric | V7 | V9 |
|---|---:|---:|
| Mean dynamic port-Z relative L2, 15 developer histories | 7.516% | 6.559% |
| Mean maximum 3-D temperature error | 1.551 K | 1.551 K |

Best checkpoint selected at epoch 240. The improvement
on the development cohort is 12.7% relative for dynamic Z.
Production V7 checkpoint and all prior independent
acceptance results remain intact.

After checkpoint freeze, a retrospective diagnostic on
the *previously examined* S04 case yields:

| Duration | V7 dynamic-Z error | V9 dynamic-Z error |
|---|---:|---:|
| 60 s | 5.400% | 5.529% |
| 300 s | 12.639% | 9.540% |
| 600 s | 23.128% | 21.506% |

This S04 comparison is diagnostic and was not used for
training or selection. V9's status remains experimental
until fresh independent acceptance completes.

## Fresh independent test cohort

The new W01?W04 geometries were frozen in
runs/geomfrac_s5_v9_plan.json, with distinct geometry
combinations not used in the original 30 training geometries
or previously tested Q/R/S geometries.
W01 and W02 use 55?60% of the normal generator's
excitation voltage and W03?W04 retain normal excitation.
Each geometry receives separate 60/300/600-second BFZI
V150 S5 full-physics teacher trajectories (12 total).

The selected V9 configuration is frozen in
runs/geomfrac_et_v9_selection.json **before running
new W physical teachers**. Fresh comparison must include
V7 and V9 on exactly the same W histories, dynamic-Z
relative L2, full port-Z relative L2, 3-D temperature
error, CPU time and material-energy residual.
The acceptance script is
examples/bfzi_geom_v9_accept.py.

## Inference

Use the original V7 inference command, adding:
  --z-fix-model runs/geomfrac_et_v9_ztune.pt

The nonlinear correction replaces the linear baseline
tangent at inference, retaining all other options.
The old linear tangent is still separately available
via --tangent-model.

Current evidence:
- runs/geomfrac_et_v9_ztune_dev.json
- runs/geomfrac_et_v9_eval_dev.json
- runs/geomfrac_et_v9_s04_retro.json
- runs/geomfrac_et_v9_selection.json

## W01?W04 frozen independent acceptance

Four previously unseen BFZI V150 S5 geometries, twelve 60/300/600-second physical histories, including two low-voltage excitation geometries, have been completed without changing the frozen V9 checkpoint.

| Model | Mean dynamic-Z rel L2 | Worst history dynamic-Z rel L2 | Mean maximum 3-D temperature error | Mean CPU/history | Max material energy residual |
|---|---:|---:|---:|---:|---:|
| V7 | 8.669% | 19.836% | 1.062 K | 0.689 s | 9.31e-09 W |
| V9 | 6.994% | 17.444% | 1.062 K | 0.769 s | 9.36e-09 W |

| Geometry | Drive setting | V7 mean dynamic-Z | V9 mean dynamic-Z | V7 600s | V9 600s |
|---|---|---:|---:|---:|---:|
| W01 | weak | 13.988% | 12.019% | 19.836% | 17.444% |
| W02 | weak | 9.715% | 6.002% | 13.642% | 6.864% |
| W03 | full | 3.271% | 2.870% | 3.740% | 3.698% |
| W04 | full | 7.702% | 7.085% | 8.004% | 6.246% |

The frozen V9 nonlinear tangent yields a relative 19.32% change in mean dynamic Z versus V7 on the new cohort. Full teacher and trajectory-level results are in runs/geomfrac_et_v9_final.json.
