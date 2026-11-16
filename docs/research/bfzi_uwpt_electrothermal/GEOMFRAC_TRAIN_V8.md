# GeomFracET V8 ? optional 64-mode seawater temperature decoder

## Objective and implementation

V7 has completed its independent S01?S04 BFZI V150 S5
acceptance: 12 physical histories, 7.890% mean dynamic-Z
relative L2 and 2.597 K mean maximum 3-D temperature error.
The S03 600 s local hotspot error remained 7.298 K.

V8 is an optional spatial temperature-output experiment.
The canonical heat-capacity-orthonormal basis is extended
from 48 to 64 modes, preserving all first 48 columns and
all five material-mean coordinates. The 16 extra modes
are bulk-seawater-centered, geometry-scaled functions near
the two packages, Gram?Schmidt orthogonalized against
the existing modes. Their orthogonalization yields small
components on the adjacent seawater skin.

A new 16-output small MLP learns the additional projected
physical temperature residual relative to frozen V7.
Inputs: 45 geometry/material features, 32 autonomous neural
thermal coordinates, time and applied voltage. Training uses
the 30 original geometries and 90 histories, with T04/D01?D04
(5 geometries, 15 histories) only for checkpoint selection.
V6 five-material physical feedback and V7 sea-skin output
remain unchanged. New optional field: temp_bulk_K.

## Results and selection

| Metric | 32 modes | 48 modes | 64 modes |
|---|---:|---:|---:|
| Development average maximum temperature projection error | 2.541 K | 1.151 K | 1.071 K |
| S03 600s projected true-temperature maximum error, retrospective | 10.552 K | 6.425 K | 5.327 K |

Checkpoint selected at epoch 40. Mean maximum 3-D
temperature error on all developer histories decreases
from V7 1.1823 K to V8 1.1791 K: only 0.0032 K gained.

Previously evaluated S01?S04 tests are retrospective
diagnostics, not independent V8 acceptance:

| Previously tested geometry | V7 600s error | V8 600s error |
|---|---:|---:|
| S01 | 3.751 K | 3.751 K |
| S02 | 6.039 K | 6.037 K |
| S03 | 7.298 K | 6.510 K |
| S04 | 1.378 K | 1.387 K |

S03 improves 7.298 to 6.510 K, about 10.8%.
S04 changes 1.378 to 1.387 K, a slight regression.
Adding local bulk-water modes has limited global
developer benefit; V7 remains the accepted production
model. V8 is an optional experimental output decoder.

All outputs share BFZI FV cell ordering. The new
temperature field does not feed the EM/thermal solver.
Its added modes have zero capacity-weighted material
means, and the physical five-material energy balances,
copper/package temperatures and dynamic port Z remain
unchanged. Inference smoke checks passed and 17
relevant unit tests passed.

## Reproduce

Projection: python -m examples.bfzi_geom_basis64
  --manifest runs/geomfrac_s5_v4_30g_manifest.json
  --split val --out runs/geomfrac_et_v8_basis64_dev.json

The decoder is saved as runs/geomfrac_et_v8_bulk.pt.
Append the argument
--bulk-output-model runs/geomfrac_et_v8_bulk.pt
to the V7 inference command with
--skin-output-model runs/geomfrac_et_v7_skin_resid.pt.
The additional output appears as temp_bulk_K.

Evidence: runs/geomfrac_et_v8_bulk_dev.json,
runs/geomfrac_et_v8_bulk_output_dev.json,
runs/geomfrac_et_v8_bulk_v7a_retro.json through
runs/geomfrac_et_v8_bulk_v7d_retro.json.

## Next research priority

S04 has a 23.128% relative dynamic-Z error on its
600-second history. The corresponding full port-Z
relative L2 error is only 0.0028% because its physical
dynamic impedance change is small. Improve
weak-thermal-signal dynamic impedance generalization
using the original train and developer splits, followed
by new independent samples. Previously examined S
geometries must not be used for training or model selection.
