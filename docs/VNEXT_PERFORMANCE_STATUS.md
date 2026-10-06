# vNext performance / training workflow status

Branch: `perf/vnext-cuda-batch-parallel`

The branch contains the vNext tensor performance pass plus the one-click
cached/resumable training workflow.

Implemented:

- true same-topology vectorized tensor FAST port batches;
- batched tensor spatial neural point evaluation;
- deterministic multi-process tensor teacher generation;
- per-worker BLAS/OpenMP thread limiting;
- one coupled electromagnetic teacher solve for combined port + spatial truth;
- automatic CUDA/MPS/CPU device selection;
- accelerator FP32 / CPU FP64 automatic training precision;
- dtype-preserving tensor port/spatial artifact loading;
- cached batched tensor FAST inference;
- content-addressed teacher sample cache with per-shard SHA-256;
- cache reuse when count/workers/device/network/optimizer change;
- automatic cache invalidation when sampler/teacher/truth parameter space changes;
- batch-level resumable port and spatial training checkpoints;
- cooperative pause/resume/stop control at safe numerical boundaries;
- nonblocking PyQt6 + pyqtgraph training GUI using QProcess + QThread;
- port/spatial training curves plus teacher-cache progress;
- root `run.py` as the one-click configuration/launch entrypoint;
- headless training with `python run.py --mode train`;
- focused cache identity, control protocol, batch-equivalence, dtype and training tests.

No GitHub Actions are added or used.

## Local validation command

The intended focused validation command is:

```bash
python -m pytest -q \
  tests/test_vnext_performance.py \
  tests/test_vnext_performance_training.py \
  tests/test_vnext_training_control.py \
  tests/test_vnext_workflow_cache.py \
  tests/test_vnext_workflow_identity.py \
  tests/test_vnext_run_config.py \
  tests/test_vnext_tensor_fast.py \
  tests/test_vnext_tensor_spatial_fast.py
```

For GUI validation on a machine with a display (or Qt offscreen platform), install:

```bash
pip install -e ".[dev,neural,gui]"
```

Then run:

```bash
python run.py
```

## Current environment limitation

The current development environment cannot obtain a local repository worktree
through its network/DNS path, and raw GitHub files cannot be downloaded into the
execution container. Therefore the complete repository pytest/py_compile run has
not been executed here. GitHub Actions are intentionally not used as a workaround.

The branch has instead been checked by direct source/interface inspection and the
focused regression tests above have been added for execution in a normal local
checkout.