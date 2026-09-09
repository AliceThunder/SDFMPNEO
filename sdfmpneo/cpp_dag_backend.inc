// Native analytic-DAG / Gauss-Newton hot kernels.
// Included by cpp_training_backend.cpp so the existing BFZI-style loader,
// ABI validation and compiler discovery remain the single build path.

SDF_EXPORT int sdfmpneo_dag_batch_sparse_jacobian_f64(
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
    const int64_t* derivative_offsets,
    const int64_t* ancestor_offsets,
    const int32_t* ancestors,
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
    double* ja_out,
    double* jda_out,
    int requested_threads) {
    if (n_points < 0 || n_modes <= 0 || n_operating < 0 || n_nodes < 0) return 3;
    const int threads = std::max(1, requested_threads);
    const int64_t total_state = state_offsets[n_nodes];
    const int64_t total_deriv = derivative_offsets[n_nodes];
    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1) reduction(|:failed)
#endif
    for (int64_t p = 0; p < n_points; ++p) {
        std::vector<double> state((size_t)total_state, 0.0);
        std::vector<double> deriv((size_t)total_deriv, 0.0);
        double* a = a_out + p * n_modes;
        double* da = da_out + p * n_modes;
        double* ja = ja_out + p * (int64_t)n_modes * n_nodes;
        double* jda = jda_out + p * (int64_t)n_modes * n_nodes;
        const double* a0 = initial + p * n_modes;
        const double* u = operating + p * n_operating;
        const double* vr = value_rows + p * total_state;
        const double* sr = slope_rows + p * total_state;

        for (int m = 0; m < n_modes; ++m) {
            a[m] = initial_values[p * n_modes + m];
            da[m] = initial_slopes[p * n_modes + m];
        }
        std::fill(ja, ja + (int64_t)n_modes * n_nodes, 0.0);
        std::fill(jda, jda + (int64_t)n_modes * n_nodes, 0.0);

        for (int node = 0; node < n_nodes; ++node) {
            const int dim = dimensions[node];
            const int parent = response_parent[node];
            if (dim <= 1 || parent >= node || parent < -1) { failed |= 1; continue; }

            double static_scale = 1.0;
            for (int64_t k = initial_factor_offsets[node]; k < initial_factor_offsets[node + 1]; ++k) {
                const int index = initial_factor_indices[k];
                if (index < 0 || index >= n_modes) { failed |= 1; continue; }
                static_scale *= a0[index];
            }
            for (int64_t k = operating_factor_offsets[node]; k < operating_factor_offsets[node + 1]; ++k) {
                const int index = operating_factor_indices[k];
                if (index < 0 || index >= n_operating) { failed |= 1; continue; }
                static_scale *= u[index];
            }
            const double alpha = weights[node] * static_scale;
            double* b = state.data() + state_offsets[node];
            double* db = deriv.data() + derivative_offsets[node];
            const int64_t anc_begin = ancestor_offsets[node];
            const int anc_count = (int)(ancestor_offsets[node + 1] - anc_begin);
            if (anc_count <= 0 || ancestors[ancestor_offsets[node + 1] - 1] != node) {
                failed |= 1; continue;
            }

            if (parent < 0) {
                if (dim != 2 || anc_count != 1) { failed |= 1; continue; }
                b[0] = alpha;
                b[1] = 0.0;
                db[0] = static_scale;
                db[1] = 0.0;
            } else {
                const int parent_dim = dimensions[parent];
                const int parent_anc = (int)(ancestor_offsets[parent + 1] - ancestor_offsets[parent]);
                if (dim != parent_dim + 1 || anc_count != parent_anc + 1) {
                    failed |= 1; continue;
                }
                const double* pb = state.data() + state_offsets[parent];
                const double* pdb = deriv.data() + derivative_offsets[parent];
                for (int r = 0; r < parent_dim; ++r) {
                    b[r] = alpha * pb[r];
                    for (int c = 0; c < parent_anc; ++c)
                        db[(int64_t)r * anc_count + c] = alpha * pdb[(int64_t)r * parent_anc + c];
                    db[(int64_t)r * anc_count + parent_anc] = static_scale * pb[r];
                }
                b[parent_dim] = 0.0;
                for (int c = 0; c < anc_count; ++c)
                    db[(int64_t)parent_dim * anc_count + c] = 0.0;
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
            for (int c = 0; c < anc_count; ++c) {
                double dv = 0.0, ds = 0.0;
                for (int r = 0; r < dim; ++r) {
                    const double dbr = db[(int64_t)r * anc_count + c];
                    dv += vr[ro + r] * dbr;
                    ds += sr[ro + r] * dbr;
                }
                const int weight_index = ancestors[anc_begin + c];
                ja[(int64_t)target * n_nodes + weight_index] += dv;
                jda[(int64_t)target * n_nodes + weight_index] += ds;
            }
        }
    }
    return failed ? 4 : 0;
}

SDF_EXPORT int sdfmpneo_gn_linearize_f64(
    const double* da,
    const double* physical_f,
    const double* ja,
    const double* jda,
    const double* physical_jacobian,
    int64_t n_points,
    int n_modes,
    int n_weights,
    double* residual_out,
    double* jacobian_out,
    int requested_threads) {
    if (n_points < 0 || n_modes <= 0 || n_weights < 0) return 3;
    const int threads = std::max(1, requested_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int64_t p = 0; p < n_points; ++p) {
        const double* local_da = da + p * n_modes;
        const double* local_f = physical_f + p * n_modes;
        const double* local_ja = ja + p * (int64_t)n_modes * n_weights;
        const double* local_jda = jda + p * (int64_t)n_modes * n_weights;
        const double* jf = physical_jacobian + p * (int64_t)n_modes * n_modes;
        double* r = residual_out + p * n_modes;
        double* J = jacobian_out + p * (int64_t)n_modes * n_weights;
        for (int i = 0; i < n_modes; ++i) {
            r[i] = local_da[i] - local_f[i];
            for (int w = 0; w < n_weights; ++w) {
                double value = local_jda[(int64_t)i * n_weights + w];
                for (int k = 0; k < n_modes; ++k)
                    value -= jf[(int64_t)i * n_modes + k] * local_ja[(int64_t)k * n_weights + w];
                J[(int64_t)i * n_weights + w] = value;
            }
        }
    }
    return 0;
}
