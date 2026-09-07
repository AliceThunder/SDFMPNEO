# Post 0.8.1 implementation closure

This checklist is now historical. The implementation tasks identified after 0.8.1 have been closed in the 0.9.0 code path; current production status is documented in `docs/PRODUCTION_STATUS_0_9.md`.

## Closed in 0.9.0

1. **Factorization-free Riesz is the nonlinear production default.** `TetrahedralElectroThermalCore.build_reduced_electromagnetics()` now selects the Morse face-circulation magnetic auxiliary plus state-aware conductive scalar-tree action when no compatibility backend is explicitly requested. Sparse LU remains a verification/correctness backend.
2. **Certified partial thermal spectrum is connected to top-level constructors.** With a thermal truncation request, the core first computes only low modes and uses an independent Li-Yau omitted-eigenvalue lower bound. The full discrete spectrum is a deterministic correctness fallback if the conservative partial certificate cannot prove the requested tolerance.
3. **Typed error propagation is explicit.** Error nodes carry physical layer and norm; incompatible quantities cannot be added. Thermal residual contributions use the exact finite-time residual-to-state comparison gain before direct thermal-state terms and final output sensitivities are applied.
4. **CAD boundary heuristics were removed.** The seawater artificial boundary is identified from the OCC spherical surface type rather than a center-of-mass/radius threshold, terminal CAD entities are resolved before Boolean operations, and their persistence is checked afterwards.
5. **Multi-chart geometry support exists.** Remeshed/topologically distinct shape families are represented as a finite union of independently certified charts instead of being forced into one fictitious affine chart. Different full FE dimensions are allowed; downstream retained thermal rank and operating interfaces remain explicit compatibility contracts.
6. **Independent-validation provenance is enforced.** Maxwell/COMSOL/experiment cases require a traceable evidence manifest and cannot be marked as training data. The repository intentionally does not fabricate external evidence; actual independent datasets remain an experimental/validation deliverable.
7. **Numerical warning hygiene is fail-closed.** Exact integer topology matrices are validated before dtype conversion, exact singular local LU pivots are translated to deterministic failure, and CI treats `ComplexWarning` and `LinAlgWarning` as errors.

## Remaining research/evidence obligations

- Acquire and version real independent Maxwell/COMSOL and, where available, experimental UWPT validation evidence. These data are validation-only and must remain excluded from training.
- Provide problem-specific regularity/reliability theorem constants needed to turn the generic mesh approximation interface into a finite certified mesh term for each declared multi-material geometry chart.
- Provide theorem-derived geometry derivatives for each chart so the cross-geometry physical proof composer can automatically close the full `[a0,G,U,t]` certificate. The composer exists; unsupported chart derivatives remain fail-closed.
- Complete certified compression/minimalization of large analytic state-space realizations and establish a global convergence result for residual-grown analytic-network construction.
- Demonstrate production scaling on realistically sized UWPT CAD meshes, including memory/iteration evidence for the factorization-free auxiliary actions and partial thermal spectrum path.

## Design rule

No empirical sampling, fitted constants, warning suppression, or hidden tolerances may become certification inputs. Any unavailable proof or external evidence keeps the corresponding claim explicitly uncertified.
