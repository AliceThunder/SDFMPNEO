# C++ training backend

SDF-MPNEO keeps the training algorithm, analytic DAG, certification logic and model format in Python. Only the repeated tetrahedral and reduced-EM numerical kernels are optionally executed by a compiled C++ backend.

## Automatic build

The first geometry-working-set preparation calls `sdfmpneo.cpp_training_backend.backend_info(auto_build=True)`. If no compatible cached library exists, the runtime hashes `cpp_training_backend.cpp`, discovers an available compiler, builds a versioned shared library, validates the ABI in a fresh Python process, and writes a manifest under `sdfmpneo/_cpp_training_build/`.

Compiler discovery follows the same deployment pattern used by BFZI:

- Windows: `SDFMPNEO_CXX` / `CXX`, an initialized MSVC `cl`, `SDFMPNEO_VCVARS`, Visual Studio `vswhere` + `vcvars64.bat`, then `g++` / `clang++`;
- Linux: `SDFMPNEO_CXX` / `CXX`, then `g++`, `c++`, `clang++`;
- macOS: `SDFMPNEO_CXX` / `CXX`, then `clang++`, `c++`, `g++`; Homebrew `libomp` is detected when present.

OpenMP is preferred but not required because the main training parallelism remains the outer independent-point thread pool. Calls made through `ctypes.CDLL` release the Python GIL, allowing different point workers to execute C++ kernels concurrently.

If compilation/loading fails, the exact historical Python implementation remains available. The automatic build is attempted at most once per Python process after a failure, so hot-kernel fallback cannot repeatedly rediscover compilers or rebuild. If a compatible backend is built manually later in the same process, the manifest can still be loaded without another automatic build. Set `SDFMPNEO_CPP_DISABLE=1` to force Python fallback for debugging.

## Accelerated paths

The runtime currently accelerates or eliminates repeated work in four places without changing the mathematical model:

1. topology-preserving geometry charts reuse the reference edge/face/incidence topology instead of rebuilding it for every geometry;
2. P1 thermal tetrahedral mass/stiffness local forms are assembled in C++;
3. first-order Nedelec curl-curl/conductivity local forms are assembled in C++;
4. state-dependent reduced conductivity/loss polynomial integration used by `heat_source_and_jacobian_for_rhs()` is evaluated in C++.

Topology-only tree/cotree and conductive-gradient results are cached across geometry contexts, and terminal surface nodal weights use a vectorized exact P1 area accumulation.

## Threading

`SDFMPNEO_POINT_WORKERS` controls outer collocation/geometry parallelism. The compiled backend defaults to one internal thread per point (`SDFMPNEO_CPP_THREADS=1`) to avoid nested oversubscription. A caller may raise `SDFMPNEO_CPP_THREADS` when intentionally running with a small outer worker count.

The runtime prints the detected backend, point worker count, geometry-working-set time, slow residual-batch timings and joint Gauss-Newton timings so that a long silent startup is no longer ambiguous.

## Numerical scope

The backend does **not** change residual tolerance, collocation domain, candidate dictionary, node budget, polynomial degree, realization limit, EM/thermal equations, certification criteria or inference behavior. C++ outputs are regression-tested against the existing formulas. The backend is an implementation acceleration only.
