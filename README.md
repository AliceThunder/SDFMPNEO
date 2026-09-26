# SDF-MPNEO vNext

SDF-MPNEO vNext is the mesh-free-first electrothermal surrogate branch.  The
runtime is built around geometry-local conductor/surface/volume quadrature,
matrix-free or dense correctness backends, structure-preserving neural
surrogates, and continuous thermal Green-function evolution.  It does **not**
introduce a fixed world voxel/FEM mesh or a finite thermal/world truncation
box.

The current implementation supports arbitrarily posed finite-cross-section
superelliptic spiral coils, superquadric dielectric packages, homogeneous
background media, continuous conductor/package/background loss fields, and
time-dependent temperature queries.  REFERENCE, FAST, and CERTIFIED paths are
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
- a `HomogeneousMedium`
- arbitrary `RigidPose` values for coils and packages

The conductor representation has a finite superelliptic cross-section and uses
geometry-local longitudinal/section quadrature.  Dielectric packages use
surface/volume quadrature.  These are local numerical discretizations of the
objects, not a fixed global volume grid.

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

## Hybrid dataset: packages + optional lossy background

The hybrid dataset stores:

- port impedance and PSD dissipation channels;
- continuous conductor spatial loss;
- continuous package spatial loss;
- when the homogeneous background is lossy, continuous unbounded-background
  spatial loss;
- the declared background design domain in the immutable manifest.

Generate a dataset with the historical lossless-background behavior:

```bash
python examples/vnext_generate_hybrid_dataset.py data/hybrid \
  --count 128
```

Opt into a lossy homogeneous background domain:

```bash
python examples/vnext_generate_hybrid_dataset.py data/hybrid-lossy \
  --count 256 \
  --lossy-background-probability 0.6 \
  --background-epsilon-min 1.0 \
  --background-epsilon-max 6.0 \
  --background-conductivity-min 1e-5 \
  --background-conductivity-max 5e-3 \
  --background-radial-order 12 \
  --background-angular-order 48
```

The generator records the declared background domain in `manifest.json`.
Appending with incompatible domain arguments is rejected instead of mixing
different design domains in one frozen dataset.

## Train the hybrid FAST port surrogate

```bash
python examples/vnext_train_hybrid_residual.py \
  data/hybrid-lossy \
  artifacts/hybrid-port.pt \
  --epochs 200 \
  --device cpu
```

The training script reads the background conductivity support from the dataset
manifest.  It does not infer the intended design domain from finite-sample
minimum/maximum values.

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

temperature = thermal.temperature_step(
    np.array([0.0, 0.0, 0.04]),
    5.0,
    np.array([1.0 + 0.0j] * len(scene.coils)),
)
```

`temperature_history(...)` supports piecewise-constant current histories and
arbitrary observation times without truncating queries to a neural training
time window.

## Important current scope limits

The current vNext code intentionally fails closed outside implemented physics:

- the background medium is currently homogeneous and nondispersive
  (`epsilon_r`, `mu_r`, constant `sigma`);
- dielectric package materials may use the implemented isotropic/Debye
  material path, but magnetic package contrast is not yet approximated by the
  dielectric SIE;
- conductive media at exactly DC require a separate static-conduction
  interface formulation;
- FAST package/background spatial inference requires artifacts trained for the
  corresponding declared media domain;
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
