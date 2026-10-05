# vNext performance implementation status

Implemented on `perf/vnext-cuda-batch-parallel`:

- true same-topology batched tensor FAST port training;
- batched tensor spatial neural point evaluation;
- CUDA/MPS/CPU automatic device selection for tensor bundles and CLI;
- accelerator FP32 / CPU FP64 automatic training precision;
- dtype-preserving tensor port and spatial artifact loading;
- deterministic multi-process tensor teacher generation;
- per-worker native BLAS/OpenMP thread limiting;
- one coupled electromagnetic teacher solve for combined port + spatial truth;
- bounded feature/analytic-baseline cache for batched tensor FAST inference;
- accelerated end-to-end tensor training CLI;
- focused numerical-equivalence and training regression tests.

Validation performed in the current development environment:

- source-level API/import/serialization contract review;
- branch comparison against `main` (no commits behind);
- isolated PyTorch verification that loading FP64 state into a default FP32
  module silently casts parameters, motivating the dtype-preserving loader;
- static comparison of scalar and batched mathematical implementations.

Full repository `pytest` has not been executed in this environment because a
local repository worktree cannot be fetched through its network/DNS path.
GitHub Actions are intentionally not used. Run the focused tests locally before
merging:

```bash
pytest -q \
  tests/test_vnext_performance.py \
  tests/test_vnext_performance_training.py \
  tests/test_vnext_tensor_fast.py \
  tests/test_vnext_tensor_spatial_fast.py \
  tests/test_vnext_serialization.py \
  tests/test_vnext_inference.py
```
