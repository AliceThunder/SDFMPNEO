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
python run.py --mode bfzi-convergence --basis  # nested modes + independent R/K/Joule cubature
python run.py --mode bfzi-convergence --basis-green   # fixed R/K, reassemble Green L / vary softening
python run.py --mode bfzi-convergence --basis-domain  # 63-mode outer sphere radius sequence
python run.py --mode bfzi-convergence --basis-local   # radial/angle/wall/combined R/K ladder (fixed L)
python run.py --mode bfzi-convergence --basis-angular # polar/azimuth composite Gauss R/K tests
python run.py --mode bfzi-convergence --basis-geometry # enclosure-projected polar Gauss refinement
python run.py --mode bfzi-convergence --basis-verify  # optional costly refined R/L/K solve
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
The same `--basis` run now uses **fine primary cubature**
`(radial,polar,azimuthal,shell_normal,shell_tangent)=(6,10,16,4,5)`
to assemble `R`, `K` and positive spatial heat nodes, while its expensive
nonlocal `L` double integral uses the *same physical modes and whitening*
but the earlier `(5,8,12,3,4)` points. This avoids increasing quadratic
Green-matrix cost merely to resolve locally sharp resistive/source terms.
The standard single-quadrature reference solver remains unchanged unless
`SeaMQSConfig(inductance_quadrature=(5,8,12,3,4))` is supplied.
It is a controlled approximation: independent verification of the
nonlocal `L` is still required.

The `--basis` run then checks the 63-mode result on **two
independent finer cubatures**: one refining the near-wall normal/tangential
rules, another refining the outer-sphere radial/angular rules. In addition
to changes in `R`, `K`, and the **frozen-current** Joule integral,
`SeawaterResponse.reintegrate_local_response(points, weights)` now
re-solves the two-port system using held-out `R/K` and **the original
Green `L`**, without any new `O(Q²)` magnetic assembly. This reports
`dZ`, fixed-probe `dq`, and `dP` for the changed local operators,
which are more directly meaningful than a full matrix-norm difference.
These are same-L numerical sensitivity results, not a new truth solution.
The independent `--basis-green` sweep uses the exact original
`R/K` and heat cubature, but recomputes and re-solves the symmetric
Green `L` on alternative magnetic quadratures. At one fixed magnetic
cubature it also compares 25, 18 and 12 mm kernel softenings. The
underlying Green distances are shared between the alternate softenings.
This isolates `dL`, induced `dZ/dq/dP` and the original-current
equation residual. **Softening must not be extrapolated to zero at fixed
cubature:** point-collocation self terms scale like `w_i²/a` and may
produce an artificial divergence, even though the continuum Coulomb
double integral is finite. Refine the magnetic rule jointly with `a`.
**Outer-boundary basis fix (2026-11-02).** Earlier domain scans were
confounded by the factor `1-r²/R²` multiplying every current trial
function: moving the artificial sphere changed the basis throughout
the charging region. `build_sea_basis` now defaults to a compact
`boundary_layer` spherical cutoff with a fixed physical transition
width determined by enclosure reach (or explicit `outer_taper_width`).
Within the common physical interior both the cutoff and its gradient
are exactly 1 and 0 respectively, so raw near-pad trial modes are
invariant under outer radius changes. The legacy `quadratic` option
remains selectable for comparisons. The `box_shell` radial rule now
splits exactly at the cutoff transition onset; the R/K and independent
Green L cubatures use the same split. `--basis-domain` prints
`raw_mode_dW` at fixed near-pad probes to expose unwanted trial
space changes. This updated domain sequence **must be re-run**;
results cannot be numerically compared to prior outputs as if only
the outer radius had changed, because the basis construction changed.
The compact cutoff also imposes a stronger zero-current trace on the
artificial boundary, and still needs convergence against larger
domains and richer long-range modes.

The separate `--basis-domain` mode solves the same 63-mode *construction*
on outer radii 0.52, 0.70, 0.85 and 1.05 m. Those are distinct
boundary-conditioned trial spaces, so domain changes are not a
fixed-matrix L-only test. Both switches can be costly; neither certifies
the infinite-domain limit.

**Local R/K convergence extension (2026-11-02).** A local test on
the previous reference returned resolved port changes of **1.6657%**
for near-wall refinement and **2.8853%** for outer sphere quadrature
refinement, even though the same-L energy algebra closes. The new
`--basis-local` option therefore independently varies outer *radial*
order, outer *angular* orders, near-wall orders, then computes two
combined progressively finer R/K solutions with the **same 63 physical
trial modes, same Green L, and same source**. It reports port `dZ`,
probe `dq`, and seawater power `dP` against the original and between
successive combined stages. Only the combined stages form a
monotonically refined quadrature hierarchy; the single-axis rows are
diagnostic, not a sequential convergence curve. These refined responses
are R/K-only references, not certified continuum solutions.

**Composite angular sphere integration (2026-11-02).** On the
35-test-passing local `af9f56d` reference, changing just the outer
angles caused `base_dZ=0.0878657` and `base_dP=0.0998883`.
The combined R/K fine-grid change was `step_dZ=0.0109989`,
`step_dq=0.187754` and `step_dP=0.0129961`, so the original,
nonmonotonic single-axis comparisons do **not** establish convergence.

The sphere component of `box_shell_quadrature` now supports two
optional controls: `polar_panel_count` partitions the cosine
polar coordinate into equal Gauss panels (must divide the total
`polar_order` to preserve north-south symmetry), and
`azimuthal_panel_order` uses Gauss-Legendre points inside each fixed
periodic azimuth sector. Default `(1,1)` preserves the historical
global Gauss-mu / single midpoint-phi rule. Slab integration near the
box walls is unchanged. These parameters are available in
`SeaMQSConfig`. They apply to primary R/K and to Green L only when
Green L uses the same primary cubature. With explicitly independent
`inductance_quadrature`, Green L uses its own angular rule.

The latest local `aa958f5` run passed **36 tests** and reported
`polar_split2` `base_dZ=0.154019` and `base_dP=0.177437`.
The mixed refinement `polar2_phi3` to `polar2_phi3_fine` changed
`dZ=0.138306`, `dq=0.0818322`, `dP=0.16552`.
Thus just repartitioning ten total polar points is **not** equivalent
to refining a quadrature, and a polar/azimuth simultaneous change
does not isolate which direction still needs more accuracy.

The updated `--basis-angular` first checks midpoint-versus-Gauss
azimuth, then runs the *same* two-polar-panel, two-Gauss-points-per-
azimuth-sector rule at total polar orders **10, 14, 18 and 22**,
with radial order, sector count, shell orders, mode basis, source and
Green L unchanged. Finally it separately compares azimuth Gauss order
two versus three at the same 22 polar nodes. Each row reports
`dR/dK`, re-solved `dZ/dq/dP`, and four absolute
`probe_q` values; each following row additionally prints
`step_from_previous` and the maximum per-probe relative change.
The run additionally prints `K_only_re_solved` using the *same*
new coupling integral but the original R, Green L and spatial Joule
nodes. This isolates how much of each response change is attributable
to the source-coupling matrix rather than the resistive operator.
`SeawaterResponse.resolve_with_coupling(K)` exposes this same passive
re-solve for externally reintegrated two-port coupling matrices;
matching R and heat nodes still ensure exact discrete energy closure.
It does **not** make an arbitrarily supplied K physically accurate.

Only the polar-order 10/14/18/22 rows form a meaningful
single-axis order refinement. The Gauss nodes are not mathematically
nested, so small consecutive differences alone are not a rigorous
error bound. The old quadrature stays as the default.

Composite Gauss angular rules cost extra spherical field evaluations.
In particular, small total port-power errors can hide much larger
local heat-source changes; choose reference orders from verified
pointwise and port stability, not node count alone.

**Geometry-aligned polar R/K refinement (2026-11-02).** The
37-test-passing `10a615d` run showed that varying two fixed polar
panels' Gaussian degree from 14→18→22 changed the re-solved port
`dZ=0.0483211` and `0.0243709` respectively, while changing
azimuth Gauss order from 2→3 at 22 polar nodes gave only
`dZ=0.000767303`. Almost all of the polar-induced port change was
reproduced by K-only re-solves; the pressing problem is **geometry-
localized source coupling K angular integration**, not a
generalized lack of outer-sphere azimuthal resolution.

The new optional `polar_geometry_order` for
`BfziScene.box_shell_quadrature` and `SeaMQSConfig` partitions the
cosine-polar interval at the minimum/maximum projected polar angles
of the actual oriented insulation-box corners **and** the same
corners on their partition-of-unity blending shells. The panel
boundaries depend on the physical scene and box-shell blending width,
not on the chosen Gauss integration order. It uses exactly
`polar_geometry_order` positive Gauss nodes **per geometric
interval**; this is distinct from the global `polar_order`
parameter, which is ignored for this one optional polar rule.
Geometry panels cannot be combined with uniform `polar_panel_count>1`.
R/K honor this option. Green L inherits it only if no separate `inductance_quadrature` is specified; an independent Green L quadrature keeps its own lower-cost angular rule.
Default behavior remains unchanged.

`python run.py --mode bfzi-convergence --basis-geometry`
holds the original 63-mode basis, source, R whitening, Green L,
radial and box-wall cubature, and azimuth sectors fixed. It tests
per-panel polar Gauss orders 3, 4, 5, 6, 8 at azimuthal Gauss
order 2, then varies only azimuth Gauss 2→3 at polar order 8.
It reports `dR/dK`, baseline and consecutive-step
`dZ/dq/dP`, maximum individual probe `dq`, and separate
**K-only and R-only** re-solves. This is a hypothesis about
the dominant numerical error mechanism, **not** evidence
that the geometry-panel approach is converged until local results
verify it. More panels can increase field-evaluation cost.

**After 38 passing tests: K polar convergence and next error source.**
The user-run geometry-aligned integration sequence on `4d02f50`
showed `geometry_mu6→geometry_mu8` changes
`dZ=0.000242124`, `dq=0.000444726`,
`dP=0.000296029`; `geometry_mu8` with azimuth
Gauss order two versus three gave `dZ=0.000413271`,
`dq=0.000109597`, `dP=0.000500205`.
This strongly supports convergence of the **spherical angular
portion** of primary R/K at the fixed 63-mode basis, Green L,
radial and wall cubatures. It does **not** certify R/K, Joule
or full electromagnetic reference convergence. Relative to the
old global-angular reference, the converged-in-angle
`geometry_mu8` response still differs by ~6.38% in port Z:
the original reference had a substantial angular bias.

`--basis-geometry` now also re-integrates the same
`geometry_mu8` solution with radial order 6→7,
box-wall orders 4/5→5/6, and both together. These held-out
comparisons **all use geometry_mu8 as the baseline**, not
the old under-resolved global-angular reference.
They reveal which one-volume R/K errors remain after
the angular bottleneck is addressed.

**Important Green cost separation.** When
`SeaMQSConfig.inductance_quadrature=(radial,polar,azimuthal,
shell_normal,shell_tangential)` is explicitly given, the
independent Green L cubature now uses exactly its own
global Gauss-mu/midpoint-azimuth angular rule,
**not** the R/K `polar_geometry_order`,
`polar_panel_count` or `azimuthal_panel_order`.
Otherwise, enabling high-resolution R/K geometry panels
would accidentally increase L double-integration work from
thousands to tens of thousands of nodes, causing a
quadratic-cost explosion. The outer radial panel cap is
still shared, and the physical trial modes and R-derived
whitening remain identical. Without an independent L
configuration, L still uses the primary R/K nodes; specify
`inductance_quadrature` for a practical high-resolution
geometry-R/K reference. Green L convergence and softening
must be verified separately.

**Outer radial physical resolution.** The spherical portion of
`box_shell_quadrature` additionally supports
`max_radial_panel_width` (meters): all physical radial intervals
longer than the cap are split into positive Gauss panels, without
altering the box shells. The `--basis-domain` sequence enables a
0.18 m cap for both R/K and Green L at radii 0.52, 0.70, 0.85, 1.05
and 1.30 m. This prevents the outermost Gauss interval from becoming
arbitrarily broad at fixed order as the sphere grows. Enabling this
cap changes the discretization, and its convergence must be checked
independently: it does not by itself validate the infinite-seawater
limit. Larger-radius Green L integration becomes costlier.

The optional `--basis-verify` solves the same fine 63-mode
`R/K` system again, this time recomputing `L` on that full finer
cubature rather than reusing the coarse Green rule. It reports resulting
`dZ`, `dq` and `dP` and can be **substantially** slower. Small holdout discrepancies
alone do **not** certify `L`, basis completeness, or outer-domain convergence.
These are trial-space/reference diagnostics, not neural surrogate training;
the 63-mode response is not yet certified.

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
