# Post 0.8.1 implementation closure

The codebase has moved beyond the original MVP obligations. The remaining work is tracked by implementation closure rather than new model concepts.

## Closed in current architecture

- solution-data-free analytic evolution graph
- nonlinear electrothermal residual training
- continuous residual certificates over thermal/operating domains
- fixed-topology continuous geometry certification
- multiport work-conjugate outputs
- finite-time residual to state propagation
- spatial output certificates

## Closure tasks

1. Make factorization-free Riesz actions the default production path. Sparse LU remains a verification backend only.
2. Connect certified partial thermal eigenspectrum selection to all top-level constructors.
3. Introduce typed error propagation from electromagnetic state to heat source, thermal state and final outputs.
4. Replace CAD entity heuristics with persistent topology identifiers.
5. Add multi-chart geometry atlas support for remeshed shape families.
6. Add independent validation datasets and provenance manifests.

## Design rule

No empirical sampling, fitted constants or hidden tolerances may become certification inputs. Any unavailable proof keeps the branch uncertified.
