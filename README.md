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

## Install

Python 3.10+ is required.

```bash
pip install -e ".[dev]"
```

For neural FAST artifacts:

```bash
pip install -e ".[dev,neural]"
```

The NumPy/SciPy REFERENCE and CERTIFIED paths do not require PyTorch.

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

After installing the package, a published tensor bundle can be queried without
writing a Python driver:

```bash
sdfmpneo-vnext-tensor \
  artifacts/tensor-fast \
  request.json \
  --output result.json
```

The request uses the same serialized scene contract as `scene_to_dict(...)` and
accepts `"mode": "fast"`, `"reference"`, or `"certified"`. Tensor-electric
package and homogeneous-background materials are represented with
`"model": "tensor_electric"`, for example:

```json
{
  "scene": {
    "coils": [
      {
        "name": "tx",
        "geometry": {
          "outer_a": 0.02,
          "outer_b": 0.018,
          "turns": 1.0,
          "pitch_a": 0.001,
          "pitch_b": 0.001,
          "conductor_width": 0.001,
          "conductor_thickness": 0.0008
        },
        "material": {
          "conductivity": 58000000.0
        }
      }
    ],
    "medium": {
      "model": "tensor_electric",
      "relative_permittivity_tensor": [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0]
      ],
      "conductivity_tensor": [
        [0.00005, 0.0, 0.0],
        [0.0, 0.0001, 0.0],
        [0.0, 0.0, 0.0002]
      ]
    }
  },
  "frequency_hz": 100000.0,
  "mode": "fast",
  "currents": [[1.0, 0.0]]
}
```

Spatial and continuous-thermal requests use the same `spatial_queries` and
`thermal` fields accepted by `run_system_inference(...)`. `certified` mode is
scene-aware: heterogeneous and tensor-electric scenes are routed to the
coupled electric/magnetic correctness backend rather than the conductor-only
certifier.

## Continuous electrothermal evolution

FAST and REFERENCE spatial fields feed the same continuous thermal interfaces:

```python
import numpy as np
from sdfmpneo_vnext import HomogeneousThermalMedium

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
    np.ones(len(scene.coils), dtype=complex),
)
```

If the scene background material carries thermal conductivity, density, and
heat capacity, the explicit thermal medium may be omitted. Homogeneous
anisotropic thermal backgrounds are supported through
`AnisotropicThermalMedium`. Thermally distinct package regions are coupled by
a local mesh-free modified-Helmholtz/interface solve; the unbounded exterior
remains analytic. `temperature_history(...)` accepts piecewise-constant current
histories and arbitrary observation times, so long-time inference is not
truncated to a neural training-time window.

For lumped closed-loop electrothermal evolution, heterogeneous scenes should
use `ChannelResolvedCurrentEnvelope` or `ChannelResolvedVoltageEnvelope` with
`build_lumped_channel_thermal_model`, preserving every electromagnetic loss
channel explicitly.

## Graded and nested media

Radially graded isotropic package media can be compiled to a convergent nested
shell hierarchy with `compile_graded_superquadric_regions`. Convergence of the
material-profile shell approximation is tracked separately from conductor and
surface quadrature convergence. Strictly nested or mutually disjoint package
regions are supported in electromagnetic and thermal transmission.

## Current scope limits

The runtime intentionally fails closed outside implemented physics:

- tensor-electric support is piecewise homogeneous; a general continuously
  varying/non-radial 3-D tensor electromagnetic VIE is not implemented;
- magnetic permeability is isotropic in the tensor-electric material stage;
- partially intersecting package volumes are rejected because they require an
  explicit Boolean material partition;
- internal conductor branching/junction networks and phase-change/radiative
  thermal nonlinearities remain outside the current MVP;
- object-local quadrature and interface discretization are numerical
  approximations even though no fixed global world mesh is introduced;
- FAST inference requires artifacts whose declared geometry/material domains
  include the query scene.

These restrictions are explicit so unsupported physics cannot silently fall
back to a simpler model.

## Tests

Run the repository locally with:

```bash
pytest -q
```

Focused vNext suites include:

```bash
pytest -q \
  tests/test_vnext_hybrid_neural.py \
  tests/test_vnext_hybrid_spatial_neural.py \
  tests/test_vnext_tensor_electric.py \
  tests/test_vnext_tensor_fast.py \
  tests/test_vnext_tensor_spatial_fast.py \
  tests/test_vnext_serialization.py \
  tests/test_vnext_inference.py \
  tests/test_vnext_system_dielectric.py
```

The vNext development flow does not require a GitHub Actions workflow.
