# SDF-MPNEO Traceability Matrix

> 目的：保证 PPT、论文和代码始终讲的是同一套方法。任何一个核心概念如果只出现在其中一处、另外两处没有对应项，就应视为设计漂移。

| Core concept | PPT | Paper | Code / Test |
|---|---|---|---|
| solution-data-free | Slide 12 | Introduction + Sec. IV | `training.py`; no label inputs |
| multiphysics strong coupling | Slides 1, 4, 11 | Sec. II-E, III-F | `model.py`, backend `assemble_reduced` |
| multirate EM / thermo-fluid | Slides 5, 11 | Sec. III-A | EM solve inside residual callback |
| physics seed + neural enrichment | Slide 6 | Sec. III-B | `PhysicsSeedBundle`, `reduction.py`, `seeds.py` |
| excitation-independent EM basis | Slides 3, 8 | Sec. III-D/E | `global_features` vs `slow_features`; CI leakage test |
| de Rham current space | Slide 7 | Sec. III-C | `topology.py`, `assemble_solenoidal_basis` |
| de Rham velocity space | Slides 7, 10 | Sec. III-C/F | same topology path for velocity |
| harmonic space for multiply connected domain | Slide 7 | Sec. III-C | `harmonic_current`, `harmonic_velocity` |
| gauge reduction | Slide 7 | Sec. III-C | topology rank checks |
| metric orthogonalization | Slide 6/7 | Sec. III-B/C | `metric_orthonormalize` |
| basis-collapse guard | optional backup | Methods implementation details | `BasisRankError` tests |
| matrix-free EM | Slide 8 | Sec. III-E | `physics/em_matrix_free.py` |
| source direct-to-reduced projection | Slide 8 | Sec. III-E | `source_projection(B)` |
| reusable geometry Green kernel under T feedback | Slide 8 | Sec. III-E/F | backend material-weight update |
| unified mixed-dimensional thermal state | Slide 9 | Sec. II-C, III-F | one `B_T`; thermal backend |
| conservative 1D-3D mortar | Slide 9 | Sec. II-C | `conservative_mortar_matrix`; PSD/null tests |
| RANS-SST flow | Slide 10 | Sec. II-D, III-F | flow backend |
| SST positivity | Slide 7/10 | Sec. III-C/F | `positive_sst_state` |
| magnetic Reynolds criterion | backup slide | Sec. II-D / Discussion | `motional_emf_required` |
| pseudo-transient Newton | Slide 11 | Sec. III-G | `solvers/equilibrium.py` |
| Riesz-whitened physics residual | Slide 12 | Sec. IV-B | backend `training_residual` |
| implicit differentiation | Slide 12 | Sec. IV-C | `implicit_parameter_gradients` |
| BOC | Slide 13 | Sec. IV-D | `hyperreduction.py` |
| positive BOC weights | Slide 13 | Sec. IV-D | NNLS + positivity test |
| independent certifier | Slide 13 | Sec. IV-E | backend `certify`; disjoint-set test |
| adaptive rank | Slide 14 | Sec. IV-F | `RankSchedule`, `model.forward` |
| physics fallback | Slide 14 | Sec. IV-F / Results | `requires_fallback` |
| lazy basis evaluation | backup / efficiency slide | Methods implementation + Results runtime | future `basis_generator.query` path |
| BFZI as external deterministic reference | validation slide | Results validation | external benchmark only |
| no BFZI code copying | not needed on main deck | Methods reproducibility note if needed | `docs/bfzi_reference_boundary.md` |
| reciprocity | physics validation slide | Results A | EM invariant tests |
| passivity / dissipation | physics validation slide | Results A | `minimum_dissipation_eigenvalue`, Joule consistency |
| heat conservation | physics validation slide | Results A | mortar conservation tests |
| zero solution labels | Slide 12/15 | Abstract, Intro, Results B | training API contains no targets |

---

# Figure-to-code mapping

| Recommended figure | Primary code evidence |
|---|---|
| Fig. 1 multiphysics UWPT schematic | geometry + coupling definitions |
| Fig. 2 overall architecture | `model.py`, `backends/base.py` |
| Fig. 3 seed + enrichment + de Rham | `reduction.py`, `topology.py`, `seeds.py` |
| Fig. 4 matrix-free EM + heat + RANS | `em_matrix_free.py`, thermal/flow backends |
| Fig. 5 training/certification | `training.py`, `hyperreduction.py` |
| Fig. 6 physics invariants | `tests/` |
| Fig. 7 independent validation | validation scripts/results |
| Fig. 8 adaptive rank/runtime | model diagnostics/timing output |
| Fig. 9 representative multiphysics fields | optional full-field reconstruction |

---

# Contribution-to-evidence mapping

## Contribution 1

**Solution-data-free physics-seeded neural trial-space learning**

Evidence required:

- zero label count;
- residual-only training code;
- physics seed construction without solution snapshots;
- neural-only vs physics-seeded ablation;
- convergence improvement.

## Contribution 2

**Physics-by-construction multiphysics spaces**

Evidence required:

- divergence defects;
- topology validation;
- SST positivity;
- thermal mortar conservation;
- reciprocity/passivity tests.

## Contribution 3

**Multirate matrix-free strongly coupled equilibrium**

Evidence required:

- EM action receives `[N,r]` blocks only;
- no dense production EM matrix;
- temperature feedback inside every nonlinear residual evaluation;
- strong vs one-way coupling comparison;
- runtime decomposition.

## Contribution 4

**Solution-free hyper-reduction with independent certification and adaptive capacity**

Evidence required:

- BOC built without solution snapshots;
- positive weights;
- solve/certifier disjoint sets;
- indicator vs actual error on validation cases;
- adaptive-rank distribution and speed/accuracy relation.

---

# Change-control rule

任何架构修改都必须同步检查：

1. `docs/master_design.md`
2. `docs/ppt_guide.md`
3. `docs/paper_guide.md`
4. `docs/implementation_plan.md`
5. 本 traceability matrix
6. 对应代码 contracts/tests

如果修改只发生在其中一层，则不得视为完成。
