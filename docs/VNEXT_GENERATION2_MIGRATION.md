# SDF-MPNEO vNext Generation-1 -> Generation-2 migration

## 1. Goal

This document defines the code migration from the current Tensor FAST implementation to the Generation-2 architecture frozen in `docs/VNEXT_GENERATION2_ARCHITECTURE.md`.

The migration deliberately separates three concerns:

1. **model/training migration** -- must happen now;
2. **teacher-data migration** -- not required for the first Gen2 run because `energy-v3` truth remains valid;
3. **new distribution expansion** -- heterogeneous sampling, low-discrepancy design and active learning come only after the Gen2 baseline is measured on the current distribution.

No GitHub Actions workflow is part of this migration. Validation is performed locally with the repository test suite and controlled training workflow.

---

## 2. What is retained unchanged

The following code and data are retained as the physical foundation:

- continuous scene/geometry types;
- tensor-electric material types;
- MQS/mixed/dielectric REFERENCE teacher;
- `energy-v3` reciprocal energy projection;
- conductor/package/background spatial teacher fields;
- tensor scene topology checks;
- teacher cache format and current cached samples;
- continuous Spatial PSD field decoders where their local-coordinate contract remains valid;
- inference-time exact Port -> Spatial power normalization concept;
- thermal subsystem and electrothermal envelope interfaces.

The migration therefore does **not** restart from a new solver.

---

## 3. What is replaced

### 3.1 Port learned model

Replace the current one-pass `HybridPhysicsFactoredResidualNet` usage in the tensor workflow with a Generation-2 model that provides:

- runtime Gen2 scene re-encoding;
- repeated shared message-passing rounds;
- baseline-conditioned total PSD resistance;
- signed symmetric reactance correction;
- PSD dissipation channels closed to the final total resistance;
- access to final coil/package latents for the Spatial stage.

The old model class remains loadable only for legacy artifacts/tests where necessary; the default workflow must no longer instantiate it.

### 3.2 Controlled Port trainer

Replace the opaque aggregate-only metric flow with component metrics:

- train/validation resistance loss;
- train/validation reactance loss;
- train/validation channel loss;
- train/validation composite Port loss.

The trainer receives explicit TRAIN and VALIDATION subsets rather than choosing an independent split internally.

### 3.3 Spatial training target

Replace runtime-Port-channel target construction with teacher-channel canonicalization.

Old behavior:

```text
Port predicted channels
  -> transform teacher field
  -> train Spatial on a Port-error-dependent target
```

New behavior:

```text
teacher channels
  -> normalize Spatial raw field
  -> compare directly to teacher spatial field
```

Inference still uses runtime predicted Port channels for exact deployed power closure.

### 3.4 Spatial context

Replace "frozen Port latent is the complete scene context" with:

```text
frozen Port latent
+ normalized Gen2 scene features
+ one Spatial-specific interaction refinement
```

The local boundary-aware conductor coordinate representation is retained.

### 3.5 Split protocol

Replace separate Port and Spatial split seeds with one global stable partition.

Default:

```text
80% TRAIN
10% VALIDATION
10% TEST
```

All stages share the same teacher indices.

### 3.6 GUI metric semantics

The GUI must stop plotting unlike quantities under the generic names `train` and `validation` for Spatial.

Generation-2 plots/reporting should distinguish:

- Spatial train shape;
- Spatial validation shape;
- Spatial validation end-to-end.

Port should expose the composite plus component metrics in the console/metrics stream even if the main plot keeps only the composite curves.

---

## 4. New modules

The migration should introduce narrow modules rather than continuing the current monkey-patching pattern.

### 4.1 `generation2_split.py`

Responsibilities:

- stable hash assignment of teacher indices;
- `Generation2Partition` data structure;
- TRAIN/VALIDATION/TEST subsets;
- partition fingerprint stored in checkpoints/artifacts.

It supersedes the use of independent `deterministic_split` calls in the Gen2 workflow.

### 4.2 `generation2_features.py`

Responsibilities:

- re-encode `sample.scene` at training time;
- reuse the existing tensor invariant feature encoder;
- append/derive bounded physics interaction descriptors;
- publish one explicit feature schema/version.

Cached Generation-1 `sample.encoded` values remain ignored by Gen2 training.

### 4.3 `generation2_port.py`

Responsibilities:

- `Generation2PortNet`;
- shared multi-round interaction core;
- batched latent computation;
- baseline-conditioned PSD resistance decoder;
- symmetric reactance decoder;
- PSD channel decoder and exact congruence closure;
- `Generation2PortArtifact` save/load/predict contract;
- final latent extraction used by Spatial.

### 4.4 `generation2_objectives.py`

Responsibilities:

- separate `R`, `X`, channel relative losses;
- composite Port objective;
- canonical Spatial shape loss;
- end-to-end Spatial diagnostic;
- optional Port-induced Spatial floor diagnostic.

This removes the need for objective monkey-patching through module globals in the Gen2 path.

### 4.5 `generation2_spatial.py`

Responsibilities:

- Spatial context refiner;
- Generation-2 Spatial shape model;
- teacher-channel normalization during training;
- runtime Port-channel normalization during inference;
- Gen2 spatial artifact schema/fingerprint.

The existing local continuous conductor/package/background decoders may be reused internally.

### 4.6 `generation2_training.py`

Responsibilities:

- explicit-subset Port controlled training;
- explicit-subset Spatial controlled training;
- checkpoint signatures containing Gen2 model/split contracts;
- component metrics;
- restore-best behavior;
- no independent split logic.

### 4.7 `generation2_bundle.py` or compatible tensor-bundle extension

Responsibilities:

- publish/load Gen2 artifact family;
- reject accidental mixing of Gen1 Port and Gen2 Spatial or vice versa;
- store generation number and partition fingerprint.

If the existing tensor-bundle wrapper is extended instead, the manifest must still distinguish the artifact family unambiguously.

---

## 5. Existing files to modify

### `run.py`

Add/freeze Gen2 configuration:

```text
TRAINING.generation = 2
TRAINING.split_seed
TRAINING.validation_fraction
TRAINING.test_fraction
PORT_TRAINING.interaction_rounds = 3
PORT_TRAINING.resistance_weight = 1
PORT_TRAINING.reactance_weight = 1
PORT_TRAINING.channel_loss_weight = 2
SPATIAL_TRAINING.context_rounds = 1
```

Remove separate Port/Spatial validation split semantics from the active Gen2 configuration. Legacy keys may remain accepted only for old workflows.

### `workflow.py`

For Generation-2:

1. prepare/reuse teacher cache;
2. derive one global partition by teacher index;
3. build Port TRAIN/VALIDATION/TEST subsets;
4. train Gen2 Port;
5. build Spatial subsets using the same indices;
6. train Gen2 Spatial;
7. evaluate final TEST metrics once;
8. publish Gen2 bundle and training summary.

### `training_gui.py`

Update Spatial plotting semantics and optionally expose component metric labels in status/console output.

### public API / artifact IO

Expose Gen2 artifact classes and loaders without silently treating them as old schema objects.

---

## 6. Checkpoint and cache compatibility

### 6.1 Teacher cache

**Reusable.**

Reason: the physical scene and `energy-v3` labels do not depend on the learned representation. Gen2 derives its input features from `sample.scene` during training.

### 6.2 Generation-1 Port checkpoint

**Not resumable.**

Reasons:

- different model class;
- different dissipative decoder;
- different interaction depth semantics;
- different split contract;
- different objective reporting/weights.

### 6.3 Generation-1 Spatial checkpoint

**Not resumable.**

Reasons:

- different Port fingerprint;
- different target canonicalization;
- Spatial-specific context refiner;
- different checkpoint-selection metric;
- shared global split.

### 6.4 Published old bundle

Remains a legacy inference artifact. It is not upgraded in-place. A Gen2 training run publishes a new artifact family/version.

---

## 7. Migration implementation order

The migration must be performed in this order so each step has a clear contract.

### Phase 1 -- global split and Gen2 feature contract

Implement:

- stable TRAIN/VALIDATION/TEST partition;
- feature-schema wrapper that re-encodes scenes;
- physics-derived interaction descriptors;
- tests for prefix stability and SE(3)/permutation behavior of the added descriptors.

No model training change is enabled until this phase is complete.

### Phase 2 -- Gen2 Port model and objective

Implement:

- repeated interaction core;
- baseline-conditioned matrix-exponential resistance decoder;
- signed symmetric reactance decoder;
- channel congruence closure;
- component losses/metrics;
- artifact save/load/fingerprint.

Required focused tests:

- `S=0` reproduces baseline resistance;
- decoded resistance is PSD for arbitrary network output;
- signed correction can produce both `R < R0` and `R > R0` directions while remaining PSD;
- channel sum equals total resistance;
- reciprocity;
- batch and single-scene inference agree.

### Phase 3 -- Gen2 controlled Port training

Implement explicit TRAIN/VALIDATION subsets and component metrics. Checkpoint selection uses validation composite loss. TEST remains untouched.

At this point the Port path can be trained independently against existing teacher cache.

### Phase 4 -- canonical Spatial model

Implement:

- teacher-channel target normalization;
- Spatial context refiner;
- train and validation canonical shape loss;
- separate end-to-end diagnostic;
- Gen2 spatial artifact/fingerprint.

Required focused tests:

- changing Port prediction while teacher channels are fixed does not change the canonical Spatial target;
- inference still closes exactly to runtime Port channels;
- canonical shape validation uses the same metric as training;
- local PSD is preserved.

### Phase 5 -- workflow/bundle/GUI switch

Make Gen2 the default in `run.py`/`workflow.py` on the feature branch.

The training summary must include:

```text
partition counts/fingerprint
Port best train/validation R loss
Port best train/validation X loss
Port best train/validation channel loss
Port composite loss
Spatial train/validation shape loss
Spatial validation end-to-end loss
final TEST metrics
```

### Phase 6 -- current-cache baseline experiment

Only after Phases 1-5 pass focused tests:

- reuse the existing 4096 teacher cache;
- train Gen2 from fresh model checkpoints;
- compare on the fixed Gen2 validation/test partition;
- do not regenerate teacher data yet.

This isolates architecture improvement from sampling improvement.

### Phase 7 -- sampling/active-learning upgrade

After the baseline is measured:

- add regime-stratified proposal logic;
- add low-discrepancy continuous samples;
- add active-learning additions only to TRAIN;
- preserve frozen VALIDATION/TEST.

This phase intentionally changes the teacher-cache identity/distribution and is therefore separate from the model migration.

### Phase 8 -- heterogeneous FAST expansion

After the Gen2 core demonstrates stable generalization:

- include compiled smooth heterogeneous scenes in TRAIN;
- define held-out heterogeneous validation/test cells;
- expand material/object-count domain metadata;
- only then publish `heterogeneous_media=true` for FAST.

---

## 8. Migration stop conditions

Migration must stop rather than silently continue if any of these occur:

1. the new resistance decoder cannot preserve PSD for all tested outputs;
2. channel congruence fails exact closure beyond numerical tolerance;
3. batch and single-scene Port paths disagree;
4. Spatial canonical target still depends on Port prediction;
5. Port and Spatial receive different global partition memberships;
6. old model checkpoints are accepted by a Gen2 trainer;
7. the workflow silently regenerates the expensive teacher cache solely because the neural schema changed;
8. a Gen2 bundle can accidentally bind a Spatial artifact to a different Port fingerprint.

---

## 9. Rollback strategy

The work is performed on branch:

```text
feat/vnext-generation2
```

`main` remains the current Generation-1 baseline during migration.

The migration should use additive Gen2 modules first. Existing Generation-1 files are only switched once the new path has focused tests. This keeps rollback simple and avoids repeatedly rewriting working legacy code.

No teacher cache deletion is part of rollback.

---

## 10. First post-migration experiment

The first meaningful experiment after code migration is **not** another data-size scaling run.

Use the existing 4096 teacher scenes and the new fixed split. The questions are:

1. Does Port validation channel loss continue improving after ~10 epochs instead of immediately plateauing?
2. Is the Port train/validation component gap smaller than the old aggregate gap?
3. Does Spatial canonical validation shape loss track training shape loss?
4. What is the difference between canonical shape validation and full end-to-end validation?
5. Does the end-to-end error floor correspond quantitatively to Port channel error?

Only after these are answered should the data-generation distribution be changed.

---

## 11. Definition of migration completion

The Generation-2 migration is complete when:

- the default feature-branch workflow trains and publishes Gen2 Port and Spatial artifacts;
- all structural invariants are enforced by construction;
- one global TRAIN/VALIDATION/TEST partition is used by both stages;
- teacher cache reuse is demonstrated;
- the Spatial target is Port-error-independent;
- the summary exposes the decomposed metrics needed to diagnose future plateaus;
- legacy artifacts/checkpoints cannot be mixed into the Gen2 pipeline accidentally;
- the focused local test set passes;
- the user can run the same `python run.py --mode train` entry point after checking out the Gen2 branch.

After that point, model-quality work can proceed on a sound architecture rather than by patching ambiguous aggregate losses.