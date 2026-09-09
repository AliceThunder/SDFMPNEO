// Separate combined translation unit for the analytic DAG / Gauss-Newton backend.
// Reusing the established training backend source keeps ABI/compiler validation
// identical while allowing an independent source-hash/cache for DAG kernels.
#include "cpp_training_backend.cpp"
#include "cpp_dag_values.inc"
#include "cpp_dag_backend.inc"
#include "cpp_thread_probe.inc"
