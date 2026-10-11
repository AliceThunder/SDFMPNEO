# SDF-MPNEO vNext Generation-2 architecture

## 1. Status and purpose

This document freezes the Generation-2 FAST surrogate architecture before code migration. It is an engineering contract between the existing mesh-free REFERENCE/CERTIFIED stack and the new learned FAST stack.

Generation-2 is not a replacement for the physical teacher. It changes how the FAST surrogate factors the problem so that the learned part has a better inductive bias, clearer error decomposition, and a training/validation protocol that matches the actual Port -> Spatial inference chain.

The existing `energy-v3` tensor teacher remains authoritative and reusable. Generation-2 re-encodes each cached scene at training time; the cached `encoded` field from the old network is not the architecture contract.

The design is consistent with the repository theory in `theory/sections/08_neural_operator.tex`, `08b_physics_factored_scene_operator.tex`, `09_training_and_active_learning.tex`, and `09b_scene_sampling_and_coverage.tex`, but makes several previously implicit decisions explicit enough to implement.

---

## 2. Problems in the current FAST architecture

The migration is driven by structural issues rather than by another round of hyperparameter tuning.

### 2.1 Port validation plateaus while training loss keeps falling

The current tensor Port model has enough capacity to reduce training loss substantially after validation has stopped improving. This is a representation/generalization problem, not a lack-of-capacity problem.

Important causes are:

1. the analytic baseline contains the coils and background, but finite package physics is largely left to the neural residual;
2. the graph performs only one effective many-body interaction round;
3. current sampling is broad IID sampling in a high-dimensional mixed discrete/continuous scene space;
4. the current dissipative decoder imposes a stronger condition than passivity actually requires.

### 2.2 The current dissipative correction is over-constrained

The current decoder uses the form

```text
R_pred = R_baseline + F F^T
```

where `R_baseline` is the conductor/background analytic baseline. This guarantees passivity, but also assumes

```text
R_pred - R_baseline >= 0
```

in Loewner order for every passive package configuration.

There is no general physical theorem that a passive dielectric or magnetic inclusion can only increase the port resistance matrix relative to the package-free scene. Passive bodies can redistribute fields and conductor currents; some current-space directions may show reduced conductor loss even though the final total dissipation remains positive semidefinite.

Generation-2 therefore constrains the **final total** dissipative operator to be PSD, while allowing a signed correction relative to the baseline.

### 2.3 One graph pass is not a many-body electromagnetic model

The current graph computes one round of coil-coil, package-package, coil->package and package->coil messages, then decodes the output. Increasing MLP depth inside an edge function does not increase physical interaction order.

Repeated polarization/scattering is closer to

```text
p = alpha (E0 + G p)
```

or

```text
(I - alpha G) p = alpha E0
```

so the network needs explicit repeated interaction updates. Generation-2 uses a small fixed number of shared interaction rounds. For the present 1-2 package domain, three rounds are the default; the architecture remains valid for larger variable object counts.

### 2.4 Spatial training is currently entangled with Port prediction error

The current Spatial trainer uses Port-predicted dissipation channels to transform both the predicted field and the teacher target. Therefore the Spatial label itself depends on upstream Port error.

This makes it difficult to distinguish:

- Port amplitude/channel error;
- Spatial shape error;
- final end-to-end field error.

Generation-2 trains Spatial against a **teacher-channel canonical shape**. At inference time the predicted shape is normalized to the predicted Port channels. Port error and Spatial shape error become separately measurable.

### 2.5 Spatial uses a frozen Port latent as its only global context

A latent optimized for port observables is not guaranteed to retain local interface, boundary, anisotropy and field-concentration information needed for continuous local loss.

Generation-2 keeps the useful Port latent but adds a Spatial-specific context refinement stage that also sees the normalized scene features and relative interactions.

### 2.6 Port and Spatial do not currently share one held-out scene partition

A cascaded Port -> Spatial model must be evaluated on scenes unseen by both stages. Generation-2 defines one immutable scene partition and reuses it for every trainable stage.

---

## 3. Non-negotiable invariants

Generation-2 is accepted only if all of these remain structural properties rather than soft penalties.

### 3.1 Reciprocity

For reciprocal scenes,

```text
Z = Z^T
```

The reactive correction is decoded as a symmetric matrix. Dissipative channel matrices are Hermitian PSD and their sum is Hermitian.

### 3.2 Passivity

The final resistance/dissipation operator must satisfy

```text
R >= 0
```

for every inference call, independent of training quality.

Generation-2 must not enforce an unnecessary sign restriction on `R - R_baseline`.

### 3.3 Exact power closure between Port and Spatial

Let `D_k` be the Port dissipation channels and `H_k(x)` the corresponding continuous spatial operators. Then

```text
integral H_k(x) dmu = D_k
```

must hold up to the numerical tolerance of the low-dimensional congruence normalization.

### 3.4 Local non-negativity

Every decoded local loss matrix remains Hermitian PSD. Local Joule density is therefore non-negative for every port current.

### 3.5 SE(3) and object-order behavior

Global rigid transformations in a symmetry-preserving environment must not change Port observables. Relative poses are the interaction variables. Object relabeling must produce only the corresponding permutation of ports/channels/features.

### 3.6 Variable topology

The network operates on variable coil/package counts and is bucketed only for batching efficiency. The model schema must not assume a fixed ordered parameter vector.

### 3.7 REFERENCE remains authoritative

FAST may be rejected outside its declared support. CERTIFIED/REFERENCE remains the correctness path and is not silently replaced by the learned model.

---

## 4. Generation-2 data contract

### 4.1 Teacher reuse

The existing tensor `energy-v3` teacher provides:

- scene and frequency;
- reciprocal target impedance;
- power-closing dissipation channels;
- continuous conductor/package/background spatial samples;
- numerical diagnostics.

These labels remain valid for Generation-2. No teacher regeneration is required merely because the learned representation changes.

The new trainer **re-encodes `sample.scene` at runtime** instead of trusting the cached old `sample.encoded` tensor representation. This is what makes old teacher caches reusable across neural feature-schema migrations.

### 4.2 One immutable scene partition

Generation-2 introduces one prefix-stable split based on `(split_seed, teacher_index)`:

```text
TRAIN      80%
VALIDATION 10%
TEST       10%
```

The exact fractions are configuration values, but the default is 80/10/10.

The same membership is used by:

- Port training;
- Port checkpoint selection;
- Spatial training;
- Spatial checkpoint selection;
- final bundle quality reporting.

The TEST subset is never used for checkpoint selection or active-learning acquisition.

### 4.3 Stable dataset growth

Adding teacher indices must not move an existing scene between partitions. A sample is assigned by a stable 64-bit hash threshold, not by permuting `range(N)`.

### 4.4 Regime reporting

The initial migration continues to reuse the existing 4096-sample cache. Generation-2 quality reports, however, should stratify at least by:

- package count/topology;
- nested/free inclusion regime;
- tensor vs scalar background;
- lossy/lossless environment;
- frequency band;
- coil separation/near-field band;
- anisotropy/contrast band.

Low-discrepancy and active-learning sampling are a subsequent data-generation phase, not a prerequisite for reusing the existing teacher cache.

---

## 5. Generation-2 scene representation

### 5.1 Base tensor features remain useful

The existing tensor feature schema already carries:

- intrinsic coil geometry/material data;
- intrinsic package geometry/material tensors;
- relative coil-coil, coil-package and package-package poses;
- frequency/material scale information.

Generation-2 keeps these quantities but adds cheap physics-derived interaction descriptors rather than asking edge MLPs to rediscover all asymptotic scalings.

### 5.2 Physics-derived descriptors

The new encoder derives bounded, dimensionless descriptors from the raw scene, including:

- normalized center distance;
- bounded inverse-distance and inverse-cube interaction factors;
- package volume divided by scene length scale cubed;
- electric contrast invariants relative to the local background;
- conductivity/loss contrast invariants;
- magnetic contrast `mu_r / mu_bg - 1`;
- orientation projections already implied by the relative rotation matrix.

These are **interaction priors**, not claimed exact package impedances. They are intentionally cheap, analytic, smooth and well-defined for the current tensor-material domain.

### 5.3 Why not hard-code an inaccurate full package impedance baseline

A superquadric, anisotropic, finite-size package near finite-section coils does not in general have an exact closed-form port correction. A crude dipole impedance added directly to `Z` could introduce a biased baseline that the neural residual must undo.

Generation-2 therefore uses analytic physics chiefly in two safe places:

1. the existing coil/background port baseline;
2. dimensionless package/pair interaction features that seed repeated graph interactions.

A more accurate explicit self/pair package operator can later replace these descriptors without changing the Gen2 decoder or training contract.

---

## 6. Generation-2 Port surrogate

### 6.1 Encoder and interaction core

Let `c_i^0` be coil latent states and `p_a^0` package latent states. Generation-2 performs `K` shared interaction rounds, default `K = 3`.

At round `r`:

```text
m_cc(i) = aggregate_j M_cc(c_i^r, c_j^r, e_ij)
m_pp(a) = aggregate_b M_pp(p_a^r, p_b^r, e_ab)
m_cp(a) = aggregate_i M_cp(p_a^r, c_i^r, e_ia)
m_pc(i) = aggregate_a M_pc(c_i^r, p_a^r, e_ia)
```

and residual node updates are

```text
c_i^(r+1) = LN(c_i^r + U_c(c_i^r, m_cc(i), m_pc(i)))
p_a^(r+1) = LN(p_a^r + U_p(p_a^r, m_pp(a), m_cp(a)))
```

where `LN` denotes a stable feature normalization layer. Message/update weights are shared across rounds by default so interaction order increases without multiplying the parameter count unnecessarily.

### 6.2 Baseline-conditioned total PSD resistance

Let `R0` be the analytic conductor/background baseline resistance. Define a stable PSD square root

```text
B = sqrt_psd(R0)
```

with only scale-relative numerical regularization.

The network predicts a **real symmetric log-correction** `S = S^T` from final coil latents. The total resistance is

```text
M = exp_sym(S)
R = B M B^T
```

where `exp_sym` is the matrix exponential evaluated through a symmetric eigendecomposition with a bounded eigenvalue range for numerical stability.

Properties:

- `R` is PSD by construction;
- `S = 0` reproduces `R0` exactly;
- eigenvalues of `M` may be below or above one, so `R - R0` may have either sign;
- the decoder remains smooth and very cheap for the small port counts targeted by FAST.

This replaces the current `R0 + F F^T` restriction.

### 6.3 Reactive decoder

The network predicts a signed symmetric correction

```text
X = X0 + DeltaX
DeltaX = DeltaX^T
```

using per-port diagonal and symmetric pair heads. The exact DC/reactance gate remains enforced by the physical runtime contract.

### 6.4 Dissipation channels

Raw channel operators are decoded as PSD factors. The present migration keeps the established channel family:

```text
coil:0 ... coil:P-1
electric_environment:aggregate
```

because the existing `energy-v3` cached Port labels use this exact channel definition.

Let raw PSD channels be `C_k^raw`. Generation-2 computes one congruence `T` such that

```text
sum_k T C_k^raw T^H = R
```

and returns

```text
C_k = T C_k^raw T^H
```

Therefore channel passivity and exact total power closure are structural.

A later teacher-schema revision may split the aggregate environment channel into per-package/background channels, but that is intentionally not required for the first Gen2 migration because doing so would discard the reusable 4096 teacher cache.

### 6.5 Port objective and metrics

The three observable families are treated as separate first-class metrics:

```text
L_R       symmetric relative resistance error
L_X       symmetric relative reactance error
L_channel symmetric relative dissipation-channel error
```

The optimization loss is

```text
L_port = w_R L_R + w_X L_X + w_C L_channel
```

with defaults

```text
w_R = 1
w_X = 1
w_C = 2
```

because downstream Spatial inference depends directly on channel accuracy.

Every epoch must report train and validation values for all three components as well as the composite loss. Checkpoint selection uses the composite validation objective, while the channel metric remains visible so a low aggregate score cannot hide a channel plateau.

---

## 7. Generation-2 Spatial surrogate

### 7.1 Canonical-shape factorization

For each physical loss channel with integrated operator `D`, write the continuous local field conceptually as

```text
H(x) = D^(1/2) S(x) D^(1/2)
```

where `S(x)` is a normalized PSD shape field whose weighted integral is the identity in the corresponding channel subspace.

The implementation need not explicitly materialize a unique matrix square-root gauge for every training point. The existing congruence-normalization machinery can enforce the same contract numerically.

The key Generation-2 rule is:

**training normalization uses teacher channels, not Port-predicted channels.**

For a teacher sample:

1. decode raw PSD spatial matrices;
2. compute normalization transforms against `sample.target_dissipation_channels`;
3. compare the normalized prediction directly with the physical teacher spatial matrices.

The teacher label no longer changes when the upstream Port model makes an error.

### 7.2 Inference normalization

At inference time:

1. Port Gen2 predicts runtime channels `D_pred`;
2. Spatial Gen2 predicts a raw PSD shape field;
3. the same normalization rule maps the raw field to `D_pred`;
4. the resulting continuous field integrates exactly to the Port prediction.

Thus the deployed Port -> Spatial energy identity is preserved.

### 7.3 Spatial-specific context refinement

Spatial no longer treats frozen Port latents as sufficient statistics.

The Spatial network receives:

- frozen Gen2 Port latents;
- normalized intrinsic scene features;
- normalized pair/cross interaction features;
- local query coordinates.

A small Spatial context refiner performs its own residual interaction update before the point decoder. This preserves useful Port knowledge while restoring local information needed for boundary/interface and anisotropy-sensitive loss shapes.

The Port parameters remain frozen during Spatial training in the first migration. Joint fine-tuning can be considered later only after a reliable test protocol exists.

### 7.4 Spatial metrics

Generation-2 reports three different quantities and never labels them ambiguously:

1. `train_shape_loss`: canonical shape loss on TRAIN;
2. `validation_shape_loss`: the same canonical shape loss on VALIDATION using teacher channels;
3. `validation_end_to_end_loss`: deployed Port-predicted channels + Spatial-predicted shape against teacher field.

Checkpoint selection is based on `validation_shape_loss`, because the Spatial optimizer cannot change the frozen Port error. End-to-end validation remains the user-facing pipeline diagnostic.

An optional `validation_port_floor` may be reported by applying predicted Port channels to the teacher spatial shape. It estimates how much of the final field error is imposed upstream by Port alone.

---

## 8. Training protocol

### 8.1 Stage A: Port

Train Port Gen2 on the global TRAIN subset. Fit all input normalizers on TRAIN only. Select the best checkpoint on the global VALIDATION subset.

### 8.2 Stage B: Spatial

Freeze the selected Port Gen2 artifact. Train Spatial Gen2 on the same TRAIN scene indices using teacher-channel canonicalization. Select by `validation_shape_loss` on the same VALIDATION indices.

### 8.3 Stage C: Final test

After both checkpoints are frozen, report on TEST:

- Port `R`, `X`, channel and composite metrics;
- Spatial canonical shape metric;
- full Port -> Spatial end-to-end metric;
- power closure and structural invariants.

TEST is not used for early stopping.

### 8.4 Early stopping

Early stopping remains useful, but a long patience does not turn late-epoch training improvements into generalization. Defaults should be modest once the metrics are correctly separated.

Generation-2 should prefer restoring the best validation checkpoint over consuming all configured epochs.

---

## 9. Sampling and active learning

The first migration deliberately reuses the current teacher cache so the architecture change can be evaluated without simultaneously changing the data distribution.

After that baseline is established, the intended Generation-2 data path is:

1. stratify discrete regimes (topology, tensor/scalar background, lossy/lossless, nested/free inclusion);
2. use deterministic low-discrepancy continuous proposals within a regime;
3. preserve immutable VALIDATION and TEST cells;
4. run active learning only into TRAIN;
5. select candidates using physical error/uncertainty/coverage rather than Euclidean parameter distance alone.

This order lets us attribute gains to architecture vs. data coverage instead of changing both at once.

---

## 10. Heterogeneous-media compatibility

Generation-2 is designed so compiled smooth heterogeneous media become variable package/interface graphs rather than a voxel field. The iterative interaction core naturally supports more material objects.

However FAST capability `heterogeneous_media=true` must not be published until:

- the heterogeneous sampler is part of the training distribution;
- object-count/material-support metadata covers the compiled scenes;
- held-out heterogeneous validation/test regimes meet the declared error budget.

REFERENCE/CERTIFIED may support more heterogeneous scenes earlier than FAST.

---

## 11. Artifact and schema policy

Generation-2 is intentionally incompatible with Generation-1 model checkpoints.

Teacher cache compatibility:

```text
energy-v3 teacher cache: reusable
```

Model/artifact compatibility:

```text
old Port checkpoint: invalid
old Spatial checkpoint: invalid
old Port artifact: load only through legacy loader, never resume Gen2 training
old Spatial artifact: bound to old Port fingerprint, never reuse with Gen2 Port
```

New artifacts store at least:

- `model_generation = 2`;
- Gen2 feature schema;
- interaction round count;
- split contract/fingerprint;
- training support/domain summary;
- model dtype/device-independent state;
- Port fingerprint bound into the Spatial artifact.

---

## 12. Acceptance criteria for the migration

Generation-2 migration is not complete merely when the code runs. The implementation must satisfy all of the following.

### 12.1 Structural tests

For representative scenes:

- `R` is PSD;
- `Z` is reciprocal;
- every dissipation channel is Hermitian PSD;
- channel sum equals `R`;
- Spatial local matrices are PSD;
- Spatial numerical integration closes to runtime Port channels;
- common rigid transformations preserve invariant observables;
- object relabeling gives the expected permutation.

### 12.2 Training-contract tests

- Port and Spatial receive identical TRAIN/VALIDATION/TEST memberships;
- extending the teacher cache does not move old indices between subsets;
- input normalizers fit TRAIN only;
- TEST is never used in checkpoint selection;
- Spatial training target does not call the Port prediction to define teacher canonicalization.

### 12.3 Regression quality checks

The first Gen2 run should reuse the existing teacher distribution. Improvement is judged on the new frozen VALIDATION/TEST protocol, not by comparing incompatible historical random splits.

The main questions are:

1. does `validation_channel_loss` keep improving after the old plateau region?
2. does canonical `validation_shape_loss` track `train_shape_loss` more closely?
3. how much of end-to-end Spatial error is explained by Port channel error?
4. does the gap between train and validation stop widening strongly after early epochs?

### 12.4 Performance

Generation-2 may add a few small graph interaction rounds and small matrix eigendecompositions. For current port counts and 1-2 package scenes these costs are negligible compared with teacher generation and should remain compatible with fast batched GPU inference.

---

## 13. Frozen Generation-2 decisions

The following choices are frozen for the migration implementation unless a concrete contradiction is found during coding:

1. reuse `energy-v3` teacher truth;
2. runtime re-encoding of cached scenes;
3. one global 80/10/10 prefix-stable partition;
4. three shared many-body interaction rounds by default;
5. baseline-conditioned matrix-exponential total resistance decoder;
6. PSD raw dissipation channels followed by exact congruence closure to total resistance;
7. explicit Port `R/X/channel` metrics with channel weight 2 by default;
8. teacher-channel canonical Spatial training;
9. Spatial-specific context refinement on top of frozen Port latents;
10. Spatial checkpoint selection by canonical validation shape loss;
11. end-to-end Spatial loss reported separately;
12. no claim of heterogeneous FAST support until trained and validated;
13. no mandatory teacher regeneration for the architecture migration.

These decisions resolve the main correctness issues found in the Generation-1 training behavior while preserving the strongest parts of the existing mesh-free, structure-preserving runtime.