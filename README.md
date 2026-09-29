# SDF-MPNEO vNext

SDF-MPNEO vNext is the mesh-free-first electrothermal surrogate branch.  The
runtime is built around geometry-local conductor/surface/volume quadrature,
matrix-free or dense correctness backends, structure-preserving neural
surrogates, and continuous thermal Green-function evolution.  It does **not**
introduce a fixed world voxel/FEM mesh or a finite thermal/world truncation
box.

The current implementation supports arbitrarily posed finite-cross-section
superelliptic spiral coils; arbitrary strictly nested or mutually disjoint
superquadric material regions, including both conductor-enclosing packages and
free inclusions; homogeneous passive isotropic background media with constant,
Debye, multi-Debye, tabulated, or custom frequency-response models; continuous
conductor/material/background loss fields; and time-dependent temperature
queries.  REFERENCE, FAST, and CERTIFIED paths are
kept separate so a neural artifact never silently replaces the correctness
backend.

## Install

Python 3.10+ is required.

```bash
pip install -e ".[dev]"
```

For neural FAST artifacts:

```bash
pip install -e ".[dev,neural]"
```

The NumPy/SciPy REFERENCE and CERTIFIED stack does not require PyTorch.

## Main physical objects

A scene is assembled from:

- `SuperellipseSpiral` + `ConductorMaterial` + `CoilObject`
- optional `SuperquadricPackageGeometry` + material + `PackageObject`
- a passive isotropic homogeneous background material: `HomogeneousMedium`,
  `DebyeMaterial`, `MultiDebyeMaterial`, `TabulatedMaterial`, or a custom
  `PassiveIsotropicMaterial` implementation
- arbitrary `RigidPose` values for coils and packages

The conductor representation has a finite superelliptic cross-section and uses
geometry-local longitudinal/section quadrature.  Dielectric packages use
surface/volume quadrature.  These are local numerical discretizations of the
objects, not a fixed global volume grid.

Measured passive material data can be used without fitting a Debye model:

```python
from sdfmpneo_vnext import TabulatedMaterial

medium = TabulatedMaterial(
    frequencies_hz=(20e3, 100e3, 500e3),
    relative_permittivity_real=(12.0, 7.0, 4.5),
    loss_conductivity_values=(2e-5, 3e-4, 1.5e-4),
)
```

Tabulated response is interpolated in log-frequency inside the supplied
frequency interval. Queries outside that interval are rejected rather than
extrapolated.

## REFERENCE / FAST / CERTIFIED

`MeshfreeVNextSystem` exposes the common runtime:

```python
from sdfmpneo_vnext import MeshfreeVNextSystem

system = MeshfreeVNextSystem(
    port_artifact,
    spatial_artifact=spatial_artifact,
)

z_fast = system.fast_ports(scene, frequency_hz)
loss_fast = system.fast_spatial(scene, frequency_hz)

z_ref = system.reference_ports(scene, frequency_hz)
loss_ref = system.reference_spatial(scene, frequency_hz)

certified = system.certified_ports(scene, frequency_hz)
```

Package-aware REFERENCE queries dispatch to the mixed conductor + dielectric
surface-integral backend.  Homogeneous lossy-background power is represented
as an unbounded exterior-domain integral rather than a finite world box.

Radially graded isotropic package media can be compiled into a convergent
strictly nested shell hierarchy. The profile may be the built-in
`RadialIsotropicMaterialProfile` or a callable that returns any
`PassiveIsotropicMaterial` response (including Debye, multi-Debye, tabulated,
or custom models). Supplying `enclosed_coils` automatically keeps every
internal material interface clear of the finite conductor volume:

```python
layers = compile_graded_superquadric_regions(
    outer_package_geometry,
    lambda rho: DebyeMaterial(
        relative_permittivity_static=4.0 + 8.0 * rho,
        relative_permittivity_infinite=2.0 + rho,
        relaxation_time=1e-6 * (1.0 + rho),
    ),
    shell_count=8,
    enclosed_coils=scene.coils,
)
graded_scene = Scene(scene.coils, scene.medium, layers)

graded_report = graded_material_convergence(
    Scene(scene.coils, scene.medium),
    frequency_hz,
    outer_package_geometry,
    profile,
    shell_counts=(4, 8, 16),
)

certified = certify_dielectric_ports(
    graded_scene,
    frequency_hz,
    fast_artifact,
    convergence_report=hybrid_report,
    graded_convergence_report=graded_report,
)
```

The graded report isolates material-profile shell error from conductor/SIE
quadrature error. Full discretization certification can therefore require both
the ordinary hybrid refinement report and the graded-shell report to converge.

## Hybrid dataset: packages + optional lossy background

The hybrid dataset stores:

- port impedance and PSD dissipation channels;
- continuous conductor spatial loss;
- continuous package spatial loss;
- when the homogeneous background is lossy, continuous unbounded-background
  spatial loss;
- the declared geometry and package/background material design domains in the
  immutable manifest.

Generate a dataset with the historical lossless-background behavior:

```bash
python examples/vnext_generate_hybrid_dataset.py data/hybrid \
  --count 128
```

Opt into a lossy homogeneous background domain:

```bash
python examples/vnext_generate_hybrid_dataset.py data/hybrid-lossy \
  --count 256 \
  --dc-probability 0.10 \
  --dc-conductive-probability 0.50 \
  --lossy-background-probability 0.4 \
  --debye-background-probability 0.35 \
  --multi-debye-background-probability 0.15 \
  --multi-debye-min-poles 2 \
  --multi-debye-max-poles 4 \
  --background-epsilon-min 1.0 \
  --background-epsilon-max 6.0 \
  --background-conductivity-min 1e-5 \
  --background-conductivity-max 5e-3 \
  --background-debye-epsilon-infinite-min 1.0 \
  --background-debye-epsilon-infinite-max 6.0 \
  --background-debye-delta-epsilon-min 0.5 \
  --background-debye-delta-epsilon-max 30.0 \
  --debye-package-probability 0.25 \
  --multi-debye-package-probability 0.15 \
  --package-offset-fraction-min 0.0 \
  --package-offset-fraction-max 0.35 \
  --package-count-min 1 \
  --package-count-max 4 \
  --nested-package-probability 0.25 \
  --graded-package-probability 0.50 \
  --free-inclusion-probability 0.35 \
  --free-inclusion-center-radius-min 0.65 \
  --free-inclusion-center-radius-max 1.8 \
  --free-inclusion-half-extent-min 0.12 \
  --free-inclusion-half-extent-max 0.45 \
  --background-radial-order 12 \
  --background-angular-order 48
```

The generator records the declared coil/material-region geometry domain and
the package/background material domains in `manifest.json`. Package
orientation is sampled from Haar SO(3). A region may enclose a finite
conductor, form a strict nested material shell, or be sampled as a free
inclusion at a declared scene-relative distance and size. Surfaces that cut a
finite conductor or partially intersect another material region are rejected
and resampled. Material metadata includes conservative frequency-effective
`Re(epsilon_r)` and loss-conductivity bounds for Debye and multi-Debye
responses. Appending with incompatible domain arguments is rejected instead of
mixing different design domains in one frozen dataset.

## Train the hybrid FAST port surrogate

```bash
python examples/vnext_train_hybrid_residual.py \
  data/hybrid-lossy \
  artifacts/hybrid-port.pt \
  --epochs 200 \
  --device cpu
```

The training script reads the declared geometry domain together with
frequency-effective package/background permittivity and loss-conductivity
support from the dataset manifest. It does not infer the intended design domain
from finite-sample minimum/maximum values. FAST therefore fails closed on
out-of-domain coil sizes, spacing, frequency, package pose/size, material
response, or unsupported magnetic package contrast instead of silently
extrapolating. Common global SE(3) motion remains an exact invariant.

The port decoder is structurally reciprocal/passive and produces PSD
dissipation channels whose sum closes to the dissipative part of the predicted
impedance.

## Train the hybrid continuous spatial-loss surrogate

```bash
python examples/vnext_train_hybrid_spatial.py \
  data/hybrid-lossy \
  artifacts/hybrid-port.pt \
  artifacts/hybrid-spatial.pt \
  --epochs 120 \
  --background-segments-per-turn 16 \
  --background-radial-order 12 \
  --background-angular-order 48
```

The spatial model predicts continuous PSD loss shapes for conductors, packages,
and the unbounded homogeneous background.  Package and background losses share
one electric-environment port channel and are normalized jointly, so they do
not double-count dissipation.

The background neural decoder uses bounded SE(3)-invariant coordinates and a
hard far-field envelope.  Its PSD loss matrix decays at least as
`O(r^-4)`, making the 3-D exterior loss integral structurally integrable.

## Load FAST artifacts and query electrothermal evolution

```python
import numpy as np

from sdfmpneo_vnext import (
    HomogeneousThermalMedium,
    MeshfreeVNextSystem,
)
from sdfmpneo_vnext.hybrid_neural import HybridNeuralResidualArtifact
from sdfmpneo_vnext.hybrid_spatial_neural import HybridSpatialLossArtifact

port = HybridNeuralResidualArtifact.load(
    "artifacts/hybrid-port.pt"
)
spatial = HybridSpatialLossArtifact.load(
    "artifacts/hybrid-spatial.pt",
    port,
)

system = MeshfreeVNextSystem(
    port,
    spatial_artifact=spatial,
)

ports = system.fast_ports(scene, frequency_hz)
field = system.fast_spatial(scene, frequency_hz)

thermal = system.fast_continuous_thermal_field(
    scene,
    frequency_hz,
    HomogeneousThermalMedium(
        conductivity=0.45,
        density=1100.0,
        heat_capacity=1300.0,
    ),
)

# If scene.medium is an Isotropic/Debye/MultiDebye material carrying
# thermal_conductivity, density, and heat_capacity, the third argument can be
# omitted and the homogeneous thermal background is built from the scene.
#
# Package materials may also carry thermal_conductivity, density, and
# heat_capacity. Thermally distinct, strictly nested or disjoint superquadric
# packages are coupled through a local mesh-free modified-Helmholtz/MFS
# transmission solve; the unbounded exterior remains an analytic thermal Green
# field.
#
# A homogeneous anisotropic thermal background is also supported with
# AnisotropicThermalMedium(K, density, heat_capacity). Its infinite-domain
# steady/transient Green function is analytic. Tensor-anisotropic package
# thermal interfaces are still fail-closed.

temperature = thermal.temperature_step(
    np.array([0.0, 0.0, 0.04]),
    5.0,
    np.array([1.0 + 0.0j] * len(scene.coils)),
)
```

`temperature_history(...)` supports piecewise-constant current histories and
arbitrary observation times without truncating queries to a neural training
time window.

For package thermal contrast, REFERENCE uses the same interface solver with
REFERENCE electromagnetic loss fields. FAST uses the trained port/spatial loss
artifacts and solves only the local package thermal-interface system online.
Multiple strictly nested or disjoint thermally distinct material regions are
solved in one coupled interface system; no finite world box is introduced.

## Important current scope limits

The current vNext code intentionally fails closed outside implemented physics:

- arbitrary piecewise-homogeneous isotropic media can be represented by
  strictly nested or disjoint superquadric material regions, including free
  inclusions; continuously graded radial isotropic profiles are supported by
  convergent conductor-safe nested-shell compilation, while general
  non-radial 3D electromagnetic heterogeneity and true tensor-anisotropic EM
  VIE media are not yet implemented; homogeneous tensor-anisotropic thermal
  backgrounds are supported analytically, while tensor-anisotropic package
  thermal interfaces are still not implemented;
- constant, Debye, multi-Debye, tabulated, and custom passive isotropic
  frequency responses share the same solver interface; electric and magnetic
  package contrast are both handled by local surface-integral transmission
  corrections without a global world mesh;
- exact DC is supported for electrostatic scenes and for mixed conductive /
  insulating material regions through a partial environment-current
  formulation: exposed conductors participate in the static-conduction
  transmission problem while conductors fully enclosed by insulating package
  regions remain galvanically isolated;
- FAST port/spatial inference requires artifacts trained for the corresponding
  declared geometry and material domains; package/conductor surface
  intersections are rejected;
- strictly nested or disjoint package material interfaces are supported for
  electromagnetic and thermal transmission; partially intersecting package
  volumes are still rejected because they require explicit Boolean material
  partitioning;
- object-local quadrature and conductor/surface discretization remain numerical
  approximations even though there is no fixed global world mesh.

These restrictions are explicit so unsupported physics cannot silently fall
back to a simpler model.

## Tests

When the repository is available locally:

```bash
pytest -q
```

Useful focused suites for the current hybrid path are:

```bash
pytest -q \
  tests/test_vnext_hybrid_dataset.py \
  tests/test_vnext_hybrid_neural.py \
  tests/test_vnext_hybrid_spatial_neural.py \
  tests/test_vnext_system_dielectric.py
```

No GitHub Actions workflow is required for the vNext development flow.
