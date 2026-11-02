> **BFZI research branch (2026-11-01).** The active implementation is `sdfmpneo_bfzi/`: exact BFZI continuous-coil geometry, enclosed twin-port scenes, explicit return paths, direct air Neumann integrals, finite-domain seawater MQS Galerkin fields, shared Joule dissipation, and deterministic reduced subspaces. This is an **unconverged finite-domain prototype**, not yet a validated ocean/FEM surrogate. The vNext commands further below remain **legacy superellipse workflows**, not BFZI electrothermal training. See [new code instructions](docs/research/bfzi_uwpt_electrothermal/CODE_ENTRY.md) and [theory](theory/main.tex).

# SDF-MPNEO vNext

SDF-MPNEO vNext is the mesh-free-first electrothermal surrogate runtime. The
core model uses geometry-local conductor/surface/volume quadrature, dense or
matrix-free correctness backends, structure-preserving neural FAST artifacts,
and continuous thermal Green/interface evolution. It does **not** require a
fixed world voxel/FEM mesh or a finite thermal/world truncation box.

The current vNext branch supports arbitrary-pose finite-cross-section
superelliptic spiral coils, strictly nested or mutually disjoint superquadric
material regions, homogeneous isotropic or tensor-electric backgrounds,
piecewise-homogeneous isotropic or tensor-electric package materials,
continuous conductor/package/background loss fields, and time-dependent
thermal queries. REFERENCE, FAST, and CERTIFIED remain separate execution
paths so neural inference never silently replaces the correctness backend.

## BFZI-MQS prototype (current research branch)

Install with `pip install -e ".[dev]"`. Run the small example with:

```bash
python run.py --mode bfzi             # twin-port seawater + wire + thermal demo
python run.py --mode bfzi-convergence         # lighter wall/basis/domain/kernel probes
python run.py --mode bfzi-convergence --full   # additional high-cost refinements
python run.py --mode bfzi-convergence --basis  # one assembly: nested 15/27/59/63 modes
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
```

The new API imports `BFZICoil`, `RigidPose`, `InsulatingBox`, `BFZIAssembly`,
`BfziScene`, `build_bfzi_source`, `solve_seawater_reference`,
`compress_seawater_response`, `BFZICurrentControlledEnvelope`, and
`BFZIVoltageControlledEnvelope` from `sdfmpneo_bfzi`.
Both ports require explicit return conductor routes wholly inside the sealed
boxes, and the finite wire diameter must be smaller than the nominal turn pitch.
The BFZI branch does not rename or feed its path into the legacy superellipse.

Physical scope: two sealed nonconductive nonmagnetic boxes, static SE(3)
poses, finite spherical conducting seawater domain, fixed frequency and
a divergence-free Galerkin seawater current basis. The original positive
meshless spherical-ray cubature remains available with
`SeaMQSConfig(quadrature_scheme="ray")`, including exact box-wall ray cuts
and optional `wall_layers`. **The new default is
`quadrature_scheme="box_shell"`**: disjoint rotated-box Gauss shells cover
every insulating face. A smooth, nonnegative partition of unity blends
the localized shell integral with the **full** spherical integral: the
sphere receives zero weight inside insulation and smoothly regains unit
weight away from the box; the local shells carry the complementary weight.
The spherical radial rule is further split around package distances.
This avoids hard angular masking of the expanded boxes and resolves
thin-package wall modes without depending on a coarse ray intersecting
a particular face. `shell_normal_order`, `shell_tangential_order`, and
`shell_width_factor` control near-wall refinement independently of
`radial_order`, `polar_order`, `azimuthal_order` for the spherical part.
Tangential Gauss rules are now **composite**: the physical face center and
both box edges are explicit interval boundaries; positive u²-mapped
exterior nodes cluster towards each edge without introducing a world mesh.
This addresses the earlier ~85% sensitivity to changing the tangential
Gauss order from four to five, but must be checked numerically rather than
assumed fixed. Global seawater mode widths also depend on physical package
reach instead of the artificial sphere radius, making domain refinements
less entangled with the basis shape. This uses positive
geometry-local quadrature, not a world spatial solver mesh. The original
run output showed 52% mode sensitivity and 71% cubature sensitivity, so
angular, basis, outer-domain and singular-kernel convergence remain explicitly
**unproven** until the new runs and successive refinements agree. `L_air` uses a softened finite-radius
Neumann approximation. Its self entries now subdivide the *analytic*
continuous BFZI-plus-return path into short panels and integrate each
straight-panel diagonal analytically, avoiding the strong coincident-node
bias of low-order line Gauss integration. The panel size can be refined via
`build_bfzi_source(..., air_panel_length_factor=2.0)`; the default is 4.0
wire radii per panel. This is still not a fully resolved GMD or
cross-section magnetic solution.
The seawater heat source integrates to the seawater port channel **under the
same cubature**; Green-kernel assembly remains memory-bounded in blocks.

The **ordinary** convergence probes now use the same 27-mode/two-tangent
trial space as the BFZI demo; the one-tangent 15-mode space is reported only
as a deliberately lower-order comparison. The dedicated `--basis` probe
uses one enriched reference assembly and exact Galerkin restriction to
compare 15, 27, 59 and 63 nested curl modes with *identical* volume
cubature, softening, outer boundary and source quadrature. The 27-to-59
enrichment adds 32 localized modes on the broad faces of thin insulating
pads. The 59-to-63 enrichment adds four physically informed, real,
source-guided solenoidal modes, two per complete port, using
`curl(g A_port / mu0)` with a box- and sphere-vanishing smooth envelope.
The actual closed BFZI coil-plus-return vector potential supplies the
direction field, not an arbitrary constant vector. These modes retain
the zero normal trace at insulating surfaces and global solenoidality.
They are optional in `build_sea_basis(scene, sources=source,
include_source_guided=True)` and do not silently change the standard
27-mode demo. The full all-face 123-mode hierarchy remains optional.
This is a trial-space diagnostic, not neural surrogate training;
the 63-mode response must still be tested for independent convergence.

The two complete paths use `BFZIAssembly.wire_radius` and the existing
`sdfmpneo_vnext.scene.ConductorMaterial` temperature law to compute
the **finite circular-wire internal impedance** (DC + cylindrical AC skin
effect, Bessel-ratio solution); this includes the spiral and explicit return.
The structured response has the mutually exclusive PSD channels
`conductor:0`, `conductor:1` and `seawater`. The conductor helper methods
`conductor_cross_section_loss`, `conductor_heat_nodes`, and
`conductor_volume_heat` return the cylindrical skin-current radial loss,
conservative line-source heat nodes, and positive meshless volumetric Joule
nodes respectively, for **both the BFZI spiral and its return**.
The existing vNext lumped thermal `StableThermalModel` can be stepped
using `BFZICurrentControlledEnvelope` or `BFZIVoltageControlledEnvelope`
(with a passive external series/tuning impedance); conductivity, coil power
and, for voltage control, port currents update at every fixed-point iterate.
The seawater electromagnetic response stays at its reference conductivity.
For a thermal ROM with a third input column, the envelopes also inject the
seawater Joule power into a separate heat channel. A thermal model may then
have more than two states, provided its first two *decoded temperatures*
refer to the two copper conductors; additional states can represent a reduced
seawater/insulation temperature. `seawater_heat_nodes()` gives positive
spatial volume nodes whose total exactly matches the seawater port channel
under the shared Galerkin cubature. The default two-channel coil thermal
model is unchanged. A third **lumped** seawater node is illustrative, not
a calibrated infinite-ocean thermal solver.

Limitations: this is **not yet** a converged infinite-ocean reference.
It does not resolve current redistribution from proximity, arbitrary
noncircular conductor sections, heating within seawater or packages,
temperature-dependent seawater MQS, resonant feedback, or a trained BFZI
neural FAST model. The convergence script reports differences, not proof
of convergence or certified accuracy.

---

## Install

Python 3.10+ is required.

```bash
pip install -e ".[dev]"
```

For neural FAST artifacts:

```bash
pip install -e ".[dev,neural]"
```

For the one-click PyQt training console:

```bash
pip install -e ".[dev,neural,gui]"
```

The NumPy/SciPy REFERENCE and CERTIFIED paths do not require PyTorch.

## One-click cached / resumable training

The root `run.py` is the main vNext training configuration and launch entry.
Edit `CONFIG` in that file, then run:

```bash
python run.py
```

The default mode opens a nonblocking PyQt6/pyqtgraph console. Training runs in
an independent `QProcess` and metrics are tailed by a `QThread`, so correctness
teacher solves and CUDA training do not block the UI. The window provides
start/resume, pause, resume and safe stop controls together with port/spatial
loss curves and teacher-cache progress.

Headless training uses the same configuration:

```bash
python run.py --mode train
```

Teacher truth is stored in a content-addressed cache. Keeping the sampler,
teacher and truth parameter space unchanged reuses existing deterministic
sample shards; changing sample count only generates missing shards. Network,
optimizer, device and GUI changes do not invalidate teacher data. Port and
spatial model checkpoints store optimizer/model/RNG/batch position state for
safe continuation after a stop.

The legacy GUI path remains in the Git history; its configuration, cache identity and
resume semantics.

## Scene model

The main objects are:

- `SuperellipseSpiral` + `ConductorMaterial` + `CoilObject`;
- optional `SuperquadricPackageGeometry` + material + `PackageObject`;
- arbitrary independent `RigidPose` values for coils and packages;
- a homogeneous background material;
- optional strictly nested or mutually disjoint material regions.

Isotropic electric materials may use `HomogeneousMedium`, `IsotropicMaterial`,
`DebyeMaterial`, `MultiDebyeMaterial`, `TabulatedMaterial`, or another passive
frequency-response implementation. Tensor-electric regions use
`TensorElectricMaterial`, with an SPD relative-permittivity tensor and a PSD
conductivity tensor whose principal axes are expressed in the containing
region frame. Package tensors therefore rotate with package pose, while a
background tensor is expressed in world axes.

The tensor-electric backend is a piecewise-homogeneous electric transmission
model. It is not a general continuously varying tensor VIE solver.

## REFERENCE / FAST / CERTIFIED

`MeshfreeVNextSystem` is the common runtime:

```python
from sdfmpneo_vnext import MeshfreeVNextSystem

system = MeshfreeVNextSystem(
    port_artifact,
    spatial_artifact=spatial_artifact,
)

ports_fast = system.fast_ports(scene, frequency_hz)
loss_fast = system.fast_spatial(scene, frequency_hz)

ports_ref = system.reference_ports(scene, frequency_hz)
loss_ref = system.reference_spatial(scene, frequency_hz)

certified = system.certified_ports(scene, frequency_hz)
```

REFERENCE dispatches conductor-only scenes to the mixed finite-cross-section
backend and heterogeneous/tensor-electric scenes to the coupled electric /
magnetic interface backend. Homogeneous lossy-background power is represented
by an unbounded exterior-domain integral rather than a finite world box.

FAST artifacts fail closed when geometry or material values fall outside the
training domain. Tensor-electric scenes additionally require artifacts marked
`supports_tensor_electric`; scalar material features are never used as an
anisotropy proxy.

## Isotropic hybrid FAST training

The existing hybrid dataset/training path remains available for isotropic
package/background material domains:

```bash
python examples/vnext_generate_hybrid_dataset.py data/hybrid --count 128

python examples/vnext_train_hybrid_residual.py \
  data/hybrid \
  artifacts/hybrid-port.pt \
  --epochs 200

python examples/vnext_train_hybrid_spatial.py \
  data/hybrid \
  artifacts/hybrid-port.pt \
  artifacts/hybrid-spatial.pt \
  --epochs 120
```

The port decoder is reciprocal/passive by construction and returns PSD
conductor plus aggregate electric-environment dissipation channels whose sum
closes to the dissipative part of the predicted impedance. The spatial model
then resolves those channels into continuous conductor, package, and optional
unbounded-background loss fields.

## Tensor-electric FAST port and spatial training

Tensor-electric training uses tensor-invariant material features for both the
port model and the spatial latent state. The spatial decoder reuses the mature
PSD conductor/package/background shape networks but is normalized against the
tensor-aware port channels; it therefore does not collapse anisotropic
permittivity/conductivity to scalar surrogates.

The current tensor FAST family is deliberately the **hybrid package-aware**
family: its teacher/trainer requires at least one package region. A tensor
homogeneous background may be sampled and learned together with those package
regions. Background-only tensor scenes are supported by REFERENCE/CERTIFIED,
but are not silently accepted by a tensor hybrid FAST artifact trained on a
package domain.

A minimal in-memory research workflow is:

```python
import numpy as np

from sdfmpneo_vnext import HybridSceneSamplerConfig, MQSConfig
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from sdfmpneo_vnext.tensor_training_data import TensorHybridTeacherSample
from sdfmpneo_vnext.tensor_spatial_training_data import (
    TensorHybridSpatialTeacherSample,
)
from sdfmpneo_vnext.tensor_neural import (
    train_tensor_hybrid_residual_surrogate,
)
from sdfmpneo_vnext.tensor_spatial_neural import (
    train_tensor_hybrid_spatial_loss_surrogate,
)

sampler = TensorHybridSceneSamplerConfig(
    base=HybridSceneSamplerConfig(),
    tensor_package_probability=1.0,
    tensor_background_probability=0.5,
)

teacher_config = MQSConfig()
rng = np.random.default_rng(7)
spatial_samples = []
for _ in range(16):
    scene, frequency_hz = sample_tensor_hybrid_scene(rng, sampler)
    port_truth = TensorHybridTeacherSample.generate(
        scene,
        frequency_hz,
        teacher_config=teacher_config,
    )
    spatial_samples.append(
        TensorHybridSpatialTeacherSample.generate(
            port_truth,
            teacher_config=teacher_config,
        )
    )

port, port_report = train_tensor_hybrid_residual_surrogate(
    tuple(sample.port for sample in spatial_samples),
    epochs=200,
)
spatial, spatial_report = train_tensor_hybrid_spatial_loss_surrogate(
    port,
    spatial_samples,
    epochs=120,
)
```

`TensorHybridSpatialLossArtifact.prepare(...)` produces the same prepared
continuous-field interface used by the isotropic hybrid path, including:

- conductor-local `local_dissipation_matrix/matrices` queries;
- package-local/world `package_*dissipation*` queries;
- unbounded-background dissipation queries when the background is lossy;
- exact normalization to the tensor FAST port dissipation channels;
- compatibility with `MeshfreeVNextSystem.fast_spatial` and continuous thermal
  evolution.

## Tensor artifact persistence

Tensor port and spatial artifacts can be published as one self-contained
bundle:

```python
from sdfmpneo_vnext.tensor_bundle import (
    publish_tensor_bundle,
    load_tensor_bundle,
)

publish_tensor_bundle(
    "artifacts/tensor-fast",
    port,
    spatial_artifact=spatial,
    overwrite=True,
)

loaded = load_tensor_bundle("artifacts/tensor-fast")
system = loaded.system
ports = system.fast_ports(scene, frequency_hz)
field = system.fast_spatial(scene, frequency_hz)
```

The manifest contains checksums, runtime capability metadata, and a semantic
fingerprint binding the spatial artifact to the exact tensor port weights and
normalization state.

### Tensor bundle JSON inference

```bash
sdfmpneo-vnext-tensor artifacts/tensor-fast request.json --device auto
```

`--device auto` prefers CUDA, then MPS, then CPU. Tensor bundle loading preserves
the dtype stored in the port/spatial artifacts.

## Continuous thermal fields

A FAST spatial field can be coupled directly to continuous thermal evolution:

```python
thermal = system.fast_continuous_thermal_field(
    scene,
    frequency_hz,
    HomogeneousThermalMedium(...),
)
temperature = thermal.temperature_step(position, time_s, currents)
```

The thermal field remains continuous in space and time. Package thermal
interfaces and anisotropic background/package thermal tensors remain distinct
from the electromagnetic tensor-electric model.

## Scope boundaries

The vNext MVP intentionally fails closed outside its declared physics domain.
In particular:

- tensor electric support is piecewise homogeneous, not a continuously varying tensor VIE;
- magnetic permeability is isotropic in the tensor-electric stage;
- partially intersecting packages are rejected; packages must be nested or disjoint;
- conductor branching/junction networks are not part of the spiral-coil MVP;
- tensor FAST training is AC-only; exact tensor DC uses REFERENCE/CERTIFIED;
- object-local quadrature/interface discretization is still a numerical approximation even though no global world mesh is required;
- phase change and nonlinear radiation are not part of the current thermal MVP.

## Performance notes

Historical vNext GPU batching, CPU multiprocessing,
precision policy and cached batch inference details.
