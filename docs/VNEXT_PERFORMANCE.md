# SDF-MPNEO vNext performance workflow

The performance path is designed around the work that actually dominates the
vNext pipeline rather than adding Python threads around everything:

- correctness/teacher samples are independent and use local worker processes;
- BLAS/OpenMP threads inside each worker are bounded to avoid oversubscription;
- tensor FAST port training uses true same-topology vectorized PyTorch batches;
- tensor spatial training vectorizes the expensive neural point evaluations
  across scenes while preserving per-scene PSD normalization/closure;
- tensor FAST inference can batch scenes of the same topology and caches scene
  encoding plus analytic baselines;
- `device=auto` selects CUDA first, then Apple MPS, then CPU;
- `precision=auto` uses FP32 on CUDA/MPS and FP64 on CPU.

REFERENCE and CERTIFIED remain CPU correctness paths. The GPU acceleration is
for neural FAST training/inference; it does not silently replace the physical
teacher.

## Install

```bash
pip install -e ".[dev,neural]"
```

For NVIDIA systems, install the PyTorch build appropriate for the installed
CUDA driver/runtime before installing this project.

## End-to-end accelerated tensor training

The simplest path generates teacher samples, trains the port surrogate,
optionally trains the continuous spatial surrogate, and publishes one tensor
bundle:

```bash
sdfmpneo-vnext-tensor-train \
  artifacts/tensor-fast \
  --count 256 \
  --workers 8 \
  --native-threads-per-worker 1 \
  --batch-size 32 \
  --device auto \
  --precision auto \
  --with-spatial \
  --overwrite
```

Important behavior:

- `--workers` controls independent teacher **processes**.
- `--native-threads-per-worker` limits NumPy/SciPy BLAS/OpenMP threads inside
  each process. With several workers, `1` is normally the safe starting point.
- `--with-spatial` does **not** run the coupled electromagnetic teacher twice.
  Port and spatial truth are extracted from the same correctness solve.
- `--batch-size` is a real neural batch for scenes with equal coil/package
  topology. Mixed topologies are automatically bucketed before batching.
- `--device auto` resolves to `cuda`, then `mps`, then `cpu`.
- `--precision auto` resolves to FP32 on accelerators and FP64 on CPU.
  Use `--precision float64 --device cuda` on GPUs with strong FP64 throughput
  when the additional precision is required.

For a large CPU server, start with approximately one teacher process per
physical core group and one native BLAS thread per process. Increasing both
worker count and BLAS thread count at the same time usually causes
oversubscription and can make the solve slower.

## CUDA tensor port training from Python

```python
from sdfmpneo_vnext.performance import (
    train_tensor_hybrid_residual_surrogate_accelerated,
)

port, report = train_tensor_hybrid_residual_surrogate_accelerated(
    train_samples,
    validation_samples=validation_samples,
    batch_size=32,
    device="auto",
    precision="auto",
    epochs=200,
)
```

Unlike the legacy per-scene optimizer loop, the accelerated trainer stacks
all tensors for same-topology scenes and evaluates latent states, impedance,
and PSD dissipation channels with an explicit batch dimension on the selected
PyTorch device.

## CUDA tensor spatial training

```python
from sdfmpneo_vnext.spatial_performance import (
    train_tensor_hybrid_spatial_loss_surrogate_accelerated,
)

spatial, report = train_tensor_hybrid_spatial_loss_surrogate_accelerated(
    port,
    train_spatial_samples,
    validation_samples=validation_spatial_samples,
    batch_size=16,
    device="auto",
    epochs=120,
)
```

The spatial path concatenates conductor/package/background query points across
same-topology scenes so the shape networks run as large GPU kernels. The final
PSD congruence normalization remains per scene because every scene has its own
physical port-loss target and quadrature weights.

## Parallel tensor teacher generation

For custom workflows, the one-solve generator can be used directly:

```python
from sdfmpneo_vnext.tensor_teacher_pipeline import (
    iter_tensor_teacher_samples_parallel_once,
)

samples = tuple(
    iter_tensor_teacher_samples_parallel_once(
        256,
        workers=8,
        native_threads_per_worker=1,
        include_spatial=True,
        seed=37,
    )
)
```

Each index gets a deterministic RNG stream derived from `(seed, index)`, so
changing the worker count does not change the sampled scenes.

## Batched FAST inference and caching

Repeated inference should avoid re-encoding identical scenes and recomputing
analytic baselines. Use `TensorAcceleratedRuntime`:

```python
from sdfmpneo_vnext.tensor_bundle import load_tensor_bundle
from sdfmpneo_vnext.performance import TensorAcceleratedRuntime

loaded = load_tensor_bundle("artifacts/tensor-fast", device="auto")
runtime = TensorAcceleratedRuntime(loaded.port_artifact, cache_size=512)

predictions = runtime.predict_structured_batch(
    scenes,
    frequencies_hz,
    batch_size=128,
)
```

Scenes are bucketed by `(number of coils, number of packages)` before the
network call. Feature encoding and analytic port baselines are cached with an
LRU-style bounded cache keyed by the serialized scene, frequency, and baseline
resolution.

For one tiny scene, CUDA can be slower because geometry encoding and analytic
baseline work still happen on CPU and device launch/transfer overhead is not
free. The batched runtime is intended for sweeps over many scenes/frequencies.

## Tensor bundle inference

The normal tensor inference command now defaults to automatic device
selection:

```bash
sdfmpneo-vnext-tensor \
  artifacts/tensor-fast \
  request.json \
  --device auto \
  --output result.json
```

Tensor bundle loading preserves the artifact's stored floating-point dtype.
An FP64 artifact is not silently down-cast to FP32 during load. MPS rejects an
FP64 tensor artifact explicitly because MPS does not support this model path in
FP64.

## What is and is not accelerated

| Stage | CPU processes | Native CPU threads | CUDA/MPS |
| --- | --- | --- | --- |
| tensor teacher generation | yes | bounded per worker | no |
| REFERENCE/CERTIFIED | caller-level parallelism | NumPy/SciPy | no |
| tensor FAST port training | not needed | PyTorch CPU if selected | yes, true batch |
| tensor FAST spatial training | not needed | PyTorch CPU if selected | yes, batched point evaluation |
| tensor FAST batch inference | not needed | PyTorch CPU if selected | yes |
| analytic baseline / scene encoding | cacheable CPU work | NumPy | no |
| continuous thermal interfaces | CPU | NumPy/SciPy | no |

The next GPU target, if profiling shows it is worthwhile, is the matrix-free
REFERENCE operator. It should only be moved after measuring the actual teacher
bottleneck; the current implementation intentionally keeps the correctness
backend independent from PyTorch and the learned FAST path.
