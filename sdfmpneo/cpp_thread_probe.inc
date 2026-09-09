SDF_EXPORT int sdfmpneo_native_thread_probe(int requested_threads) {
    const int threads = std::max(1, requested_threads);
    int observed = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        observed = omp_get_num_threads();
    }
#endif
    return observed;
}
