# Native analytic DAG and Gauss–Newton runtime

The late-stage SDF-MPNEO trainer now has a second compiled layer dedicated to the analytic response DAG and joint Gauss–Newton refinement. It complements the tetrahedral/reduced-EM C++ backend; it does not replace the Python training algorithm, checkpoint format, physics, residual target, candidate dictionary or certification logic.

## Exact split of work

The existing stable semigroup code still constructs/caches the exact observation rows

\[
c^T e^{At},\qquad c^T A e^{At},
\]

including long-time and stationary handling. Those rows depend on topology, decay rates and time but not on the real response weights or point amplitudes. During repeated weight refinement they are therefore prepared once and reused.

For graphs satisfying the production dictionary constraint `max_parent_responses <= 1`, C++ then evaluates, in batches:

- response-node state vectors;
- actual-ancestry weight derivatives only;
- modal values `a` and derivatives `da/dt`;
- modal weight Jacobians `J_a` and `J_da`;
- the Gauss–Newton linearization `J_da - J_F J_a`.

The dense least-squares solve remains NumPy/LAPACK so the numerical optimization step and line search are unchanged. During that solve the runtime temporarily permits the configured native CPU thread count because the point-parallel layer is idle.

Historical/custom graphs with more than one response parent, or with a nonzero imaginary component in any response weight, fail closed to the previous exact Python implementation. No imaginary component is discarded.

## Native threading

`SDFMPNEO_NATIVE_THREADS` controls the OpenMP team used by the batched DAG/Jacobian/GN kernels. When unset it defaults to the logical CPU count. This native team is requested explicitly with the OpenMP `num_threads` clause, so the GUI's historical `OMP_NUM_THREADS=1` setting does not force the DAG/GN kernels to one thread.

The physical point layer still uses `SDFMPNEO_POINT_WORKERS`; the low-level tetrahedral C++ backend defaults to one internal thread per point to avoid nested oversubscription. Thus the two principal phases use different parallel layouts:

- physical residual/context sweeps: many independent point workers × one low-level C++ thread;
- analytic DAG/Jacobian/GN batches: one native batch × many OpenMP threads.

At startup the runtime runs a native OpenMP probe and prints both the requested and observed team sizes. This is more reliable than counting processes in Windows Task Manager: Python thread-pool workers and OpenMP workers are threads inside the training process, not separate processes.

## Numerical invariants

The native layer does not alter residual tolerance, collocation points, node/degree/realization budgets, response topology, EM/thermal equations, long-time semantics, validation, certification or inference. Its outputs are regression-tested against the existing stable analytic evaluator and sparse weight-Jacobian implementation.
