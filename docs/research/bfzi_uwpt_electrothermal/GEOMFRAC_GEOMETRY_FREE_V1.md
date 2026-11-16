# GeomFracET: toward geometry-only cold EM and shared thermal operators

## Objective and dependency being removed

V9 is the accepted electromagnetic-thermal surrogate after its frozen
W01-W04 12-history independent acceptance. It still needs a BFZI V150
cold electromagnetic anchor and the per-geometry FV heat operator.
The 45-dimensional V9 input also embeds BFZI-calculated cold impedance,
so merely replacing the cold solver while reusing its old geometry
feature is insufficient for full geometry-only prediction.

The next architecture has three connected goals:

1. Learn cold BFZI response from user-specified geometry, pose, package
   dimensions and material properties, without any BFZI solve at query time.
2. Learn a common thermal-state operator on geometry-shared coordinates,
   without constructing a new FV heat grid or basis for each query.
   Support 32, 48 and 64 ranks, with five material-mean states
   used as the physical energy-conservation coordinates.
3. Retrain and jointly optimize cold, transient impedance and
   thermal-field accuracy on separate geometry-level train/dev/test
   groups, preserving the causal SOE fractional memory and
   five-material conservation correction.

## Implemented prototype: geometry-only physical initialization

New module: sdfmpneo_bfzi/geom_init.py
Training: examples/bfzi_geom_init.py
Query: examples/bfzi_geom_init_query.py

A single 18-feature representation uses only user-controlled:
two rounded-square/circular shape parameters, two corner-core sizes,
relative xyz displacement, tilt/yaw/pitch, two package widths,
five thermal/material settings and the selected known
external-circuit family. No BFZI-calculated cold Z, mutual
inductance or thermal field is present in these inputs.

The training atlas learns and stores the following geometry
conditioned quantities:

- six real components of cold two-port Z0;
- three independent cold air self/mutual inductances;
- positive capacities for five material regions;
- capacity-orthonormal, common-coordinate thermal operator
  K_r (r=32, 48 or 64);
- r-by-3 projected cold weak Joule source coefficients;
- known external-circuit impedance for each circuit family.

A smooth regularized geometry kernel predicts cold scalar
parameters; a pooled TX/RX self-coil model improves the
cold impedance predictions. The shared thermal operator and
cold source projection are interpolated as convex combinations
of training-geometry matrices. This preserves the thermal
operator's symmetry and positive diffusion spectrum without
a new geometry's H or a new POD basis. The first five
canonical coordinates remain material temperature means.
The existing geometry-defined spatial shapes provide the
common interpretation of the 32/48/64 indices.

**Current code implements the operator/port initialization only.**
V9's physical circuit and neural time evolution have not
yet been reconnected to these approximate predictions for
a fully anchor-free autonomous rollout. The full 3-D field
query decoder also still needs to be adapted to geometry-only
coordinates. Therefore the existing V9 inference remains
the accepted operational version until end-to-end
physical/accuracy acceptance.

## Data and first accuracy results

Training: original 30 S5 geometries, cold anchors only.
Model selection: T04 and D01-D04, five geometry-disjoint cold anchors.
The previous W01-W04 V9 independent cohort is used only
retrospectively for characterization. All training and
selection precede inspection of W cold prediction errors.

| Observable | Original five-geometry developer | Previously evaluated W01-W04 |
|---|---:|---:|
| Cold complex six-vector Z0 mean relative L2 | 2.375% | 1.378% |
| Cold port maximum-entry absolute error (mean) | 0.514 ohm | 0.352 ohm |
| Shared thermal operator mean relative Frobenius error (r=32) | 17.719% | 17.233% |
| Five material capacities mean relative L2 | 0.0103% | 0.0063% |
| Cold projected heat-source mean relative Frobenius | 8.936% | 9.446% |

The initial one-kernel cold predictor had 4.405% developer
cold-Z error; adding the shared two-coil self-response
model reduces it to 2.375%. Training the same shared
operator representation at 48 or 64 modes keeps developer
operator relative Frobenius error near 17.71%.
The number of modes is now a model choice rather than
a requirement to build a fresh geometry-specific POD basis.

The geometry-only atlas query (after loading a model) took
0.0004 s in a representative W01 test. Measured original
cold-anchor setup and overhead from the independent BFZI
teachers was approximately 19-20 seconds per new geometry.
This is **a partial initialization-cost comparison**;
a true full-query comparison must include the anchor-free
dynamic loop, full 3-D reconstruction, model loading and
file output, all still pending.

The current cold-Z accuracy is too coarse for immediate
substitution into the independently accepted V9 dynamic
impedance workflow. It is evidence of technical feasibility
of cold/thermal operator prediction with zero BFZI calls
at query time, not final autonomous surrogate accuracy.

## Next implementation sequence and gates

**G1 ? Accurate cold response**:
Enlarge the geometry-only cold training atlas without
generating full 600s thermal teachers. Train a physically
structured coil-inductance/cold-seawater decomposition,
using larger shape/pose/package coverage. Developer
target: mean cold port Z0 below 0.5% relative L2,
and worst geometry below 1%, with independent
geometry holdout validation. The pretrained V9 and
frozen W01-W04 tests remain untouched.

**G2 ? Anchor-free conservation-compatible dynamics**:
Implement the reduced circuit/source model using predicted
cold Z0, air inductance, five material capacities, common
K_r and projected weak Joule shapes. Couple the SOE neural
heat state to the five material integrated budgets directly
in the learned shared-coordinate operator. Do not load a
BFZI head or build a new-geometry mesh. Add a continuous
temperature reconstruction for arbitrary points in seawater,
packages and copper, with zero capacity-mean local spatial
corrections. Compare r=32/48/64 at the same error and
computational cost.

**G3 ? Joint quality/efficiency**:
Train geometry-only cold EM, thermodynamic operator and
SOE dynamic neural components with physical cold targets,
thermal response and dynamic complex-Z losses. Compare
complete unseen-geometry cold-start prediction with
the existing anchor-prepared V9 on fresh independent
BFZI V150 S5 geometries. Track complete first-inference
wall time, repeated-query cost, dynamic-Z relative L2,
full-port-Z error, 3-D max temperature error, and
five-material energy residual. Promote only a model
that meets the intended accuracy and runtime requirements.

## Commands

Train the 32-mode initial atlas:

    python -m examples.bfzi_geom_init --rank 32

Train additional common-coordinate operators:

    python -m examples.bfzi_geom_init --rank 48 --out runs/geomfrac_init_v1_48.npz --report runs/geomfrac_init_v1_48_dev.json
    python -m examples.bfzi_geom_init --rank 64 --out runs/geomfrac_init_v1_64.npz --report runs/geomfrac_init_v1_64_dev.json

Query a geometry with no BFZI installation:

    python -m examples.bfzi_geom_init_query --atlas runs/geomfrac_init_v1.npz --case-file runs/geomfrac_s5_v9_plan.json --case W01 --out runs/geomfrac_init_query.npz

Result files: runs/geomfrac_init_v1_dev.json,
runs/geomfrac_init_v1_W_retro.json, and the r=48/64
checkpoints and reports.

Next milestone: train the larger cold-only dataset,
reduce cold error, and implement an operational
fully anchor-free 32-state reduced-time inference path.
