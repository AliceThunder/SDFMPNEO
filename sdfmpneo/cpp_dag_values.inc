// Value-only analytic DAG batch. The topology assumptions are identical to the
// sparse Jacobian kernel: at most one response-node parent per response node.
SDF_EXPORT int sdfmpneo_dag_batch_values_f64(
    const double* initial,
    const double* operating,
    int64_t n_points,
    int n_modes,
    int n_operating,
    int n_nodes,
    const double* weights,
    const int32_t* targets,
    const int32_t* response_parent,
    const int32_t* dimensions,
    const int64_t* state_offsets,
    const int64_t* initial_factor_offsets,
    const int32_t* initial_factor_indices,
    const int64_t* operating_factor_offsets,
    const int32_t* operating_factor_indices,
    const double* value_rows,
    const double* slope_rows,
    const double* initial_values,
    const double* initial_slopes,
    double* a_out,
    double* da_out,
    int requested_threads) {
    if (n_points < 0 || n_modes <= 0 || n_operating < 0 || n_nodes < 0) return 3;
    const int threads = std::max(1, requested_threads);
    const int64_t total_state = state_offsets[n_nodes];
    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1) reduction(|:failed)
#endif
    for (int64_t p = 0; p < n_points; ++p) {
        std::vector<double> state((size_t)total_state, 0.0);
        double* a = a_out + p * n_modes;
        double* da = da_out + p * n_modes;
        const double* a0 = initial + p * n_modes;
        const double* u = operating + p * n_operating;
        const double* vr = value_rows + p * total_state;
        const double* sr = slope_rows + p * total_state;
        for (int m = 0; m < n_modes; ++m) {
            a[m] = initial_values[p * n_modes + m];
            da[m] = initial_slopes[p * n_modes + m];
        }
        for (int node = 0; node < n_nodes; ++node) {
            const int dim = dimensions[node];
            const int parent = response_parent[node];
            if (dim <= 1 || parent >= node || parent < -1) { failed |= 1; continue; }
            double scale = 1.0;
            for (int64_t k = initial_factor_offsets[node]; k < initial_factor_offsets[node + 1]; ++k) {
                const int index = initial_factor_indices[k];
                if (index < 0 || index >= n_modes) { failed |= 1; continue; }
                scale *= a0[index];
            }
            for (int64_t k = operating_factor_offsets[node]; k < operating_factor_offsets[node + 1]; ++k) {
                const int index = operating_factor_indices[k];
                if (index < 0 || index >= n_operating) { failed |= 1; continue; }
                scale *= u[index];
            }
            const double alpha = weights[node] * scale;
            double* b = state.data() + state_offsets[node];
            if (parent < 0) {
                if (dim != 2) { failed |= 1; continue; }
                b[0] = alpha; b[1] = 0.0;
            } else {
                const int pd = dimensions[parent];
                if (dim != pd + 1) { failed |= 1; continue; }
                const double* pb = state.data() + state_offsets[parent];
                for (int r = 0; r < pd; ++r) b[r] = alpha * pb[r];
                b[pd] = 0.0;
            }
            const int target = targets[node];
            if (target < 0 || target >= n_modes) { failed |= 1; continue; }
            const int64_t ro = state_offsets[node];
            double value = 0.0, slope = 0.0;
            for (int r = 0; r < dim; ++r) {
                value += vr[ro + r] * b[r];
                slope += sr[ro + r] * b[r];
            }
            a[target] += value;
            da[target] += slope;
        }
    }
    return failed ? 4 : 0;
}
