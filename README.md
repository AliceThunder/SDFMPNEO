> **BFZI 研究分支｜当前研发进度（2026-11-03）：[📊 总体路线图 · 封装专项 · 验收看板](docs/research/bfzi_uwpt_electrothermal/PROJECT_PROGRESS.md)**  
> 本工作仅针对**圆形／圆角方形连续螺旋线圈 + 两套可变尺寸、任意合法静态位姿的绝缘封装 + 海水电磁热代理**；不把旧 vNext 超椭圆线圈算进本项目。最新本机数值日志（`7982bfa`，2026-11-04 UTC）：**60 passed**；固定 229 模态时精细 R/K 方位角 **3→4 的 q 差异仅 0.0423%（PASS）**、径向 0.06→0.04m PASS；之前 2→3 的 q 比较仍 FAIL，表明二阶偏粗。本轮已将 BFZI 细 R/K 主计算升到三阶，并修正外域试验在新半径退回小电流基的问题：下一步使用同一套 **229 物理模态**检查外径 0.52/0.70/0.85m 的 Z/q/共同海水域 J，**尚待本机实测**。229 磁 Green L 算子仍 FAIL，G1 仍阻塞，不能生成正式教师标签。最新证据和全部历史见 [进度看板](docs/research/bfzi_uwpt_electrothermal/PROJECT_PROGRESS.md)、[自动上传日志](docs/research/bfzi_uwpt_electrothermal/runs/latest_summary.md)。默认终端精简输出，执行后直接使用本地 Git 推送日志；**不使用 GitHub Actions**。参见 [代码入口](docs/research/bfzi_uwpt_electrothermal/CODE_ENTRY.md) 和 [理论](theory/main.tex)。

# SDF-MPNEO vNext

SDF-MPNEO vNext is the mesh-free-first electrothermal surrogate runtime. The
core model uses geometry-local conductor/surface/volume quadrature, dense or
matrix-free correctness backends, structure-preserving neural FAST artifacts,
and continuous thermal Green/interface evolution. It does **not** require a
fixed world voxel/FEM mesh or a finite thermal/world truncation box.

The current vNext branch supports arbitrary-pose finite-cross-section
superelliptic spiral coils, strictly nested or mutually disjoint superquadric
material regions, homogeneous isotropic or tensor-electric backgrounds,
piecewise-homogeneous isotropic or tensor-electric package materials,
continuous conductor/package/background loss fields, and time-dependent
thermal queries. REFERENCE, FAST, and CERTIFIED remain separate execution
paths so neural inference never silently replaces the correctness backend.

## BFZI-MQS prototype (current research branch)

**最新状态与下一关口以 [BFZI 进度看板](docs/research/bfzi_uwpt_electrothermal/PROJECT_PROGRESS.md) 为准**；下方保留历史开发命令及实验说明，部分当时的“尚未运行”叙述是研究记录快照，不应覆盖进度看板中的后续实测。

Install with `pip install -e ".[dev]"`. Run the small example with:

```bash
python run.py --mode bfzi             # twin-port seawater + wire + thermal demo
python run.py --mode bfzi-convergence         # lighter wall/basis/domain/kernel probes
python run.py --mode bfzi-convergence --full   # additional high-cost refinements
python run.py --mode bfzi-convergence --basis  # nested modes + independent R/K/Joule cubature
python run.py --mode bfzi-convergence --basis-green   # fixed R/K, reassemble Green L / vary softening
python run.py --mode bfzi-convergence --basis-green-refined # streamed fine R/K + independent Green L
python run.py --mode bfzi-convergence --basis-green-cross  # two independent magnetic quadratures, diagnostic
python run.py --mode bfzi-convergence --basis-green-gaussian # PSD Gaussian Green: A/B plus width-factor sensitivity
python run.py --mode bfzi-convergence --basis-green-ball # PSD exact uniform-ball Coulomb Green A/B and shape test
python run.py --mode bfzi-convergence --basis-ball # full uniform-ball L nested modal and held-out R/K
python run.py --mode bfzi-convergence --basis-domain-ball # independent outer radius with uniform-ball L
python run.py --mode bfzi --green-ball # experimental full MQS port/Joule and coil/sea thermal
python run.py --mode bfzi-snapshot --out bfzi_response.npz # save/load/query four-real-snapshot ROM
python run.py --mode bfzi-snapshot --out bfzi_response.npz --dataset-out bfzi_provisional_labels.npz # two distinct poses
python run.py --mode bfzi-generate --samples 12 --out bfzi_geometry_labels.npz # many provisional geometry labels
python run.py --mode bfzi-train --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt --epochs 400 # torch optional; J/Joule-aware 27-mode objective
python run.py --mode bfzi-train --dataset bfzi_geometry_labels.npz --model bfzi_neural_physics.pt --epochs 400 --field-loss-weight 1.0 --joule-loss-weight 0.15 # keep original checkpoint for A/B
python run.py --mode bfzi-infer --model bfzi_neural.pt # default RAW learned Z; no reference solve
python run.py --mode bfzi-infer --model bfzi_neural.pt --power-closure # diagnostic: rebuild Re(Z) from J
python run.py --mode bfzi-eval --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt # original held-out geometries
python run.py --mode bfzi-convergence --basis-ball-check-l # optional expensive 159-mode magnetic A/B
python run.py --mode bfzi-convergence --basis-ball-face # experimental 159→167 source-face-curl enrichment
python run.py --mode bfzi-convergence --basis-ball-face --basis-ball-check-l # independent 167-mode magnetic A/B
python run.py --mode bfzi-convergence --basis-ball-edge # 167→199 independent face-edge curl directions (expensive)
python run.py --mode bfzi-convergence --basis-ball-edge --basis-ball-check-l # 199-mode A/B magnetic cubature (more expensive)
python run.py --mode bfzi-convergence --basis-ball-aniso # 199→207 rotated anisotropic face-normal current curls
python run.py --mode bfzi-convergence --basis-ball-aniso --basis-ball-check-l # independent 207-mode magnetic A/B (measured: operator FAIL, response PASS)
python run.py --mode bfzi-convergence --basis-ball-targeted # completed 207→210→213→229: global J changed 89.21% then 20.10%
python run.py --mode bfzi-convergence --basis-domain-ball --basis-ball-targeted --with-tests # next: same 229 physical modes at 0.52/0.70/0.85m, common-water J, auto-pushed
python run.py --mode bfzi-convergence --basis-ball-targeted --no-publish-log # run and log locally without Git push
python run.py --mode bfzi-convergence --basis-ball-targeted --verbose-log # optional full console output
python run.py --mode bfzi-convergence --basis-ball-aniso --basis-ball-check-rk # face-normal R/K A/B, fixed Green L
python run.py --mode bfzi-convergence --basis-ball-aniso --basis-ball-ablate # full-parent energy bound and leave-one-mode physical influence
python run.py --mode bfzi-convergence --basis-domain  # 63-mode outer sphere radius sequence
python run.py --mode bfzi-convergence --basis-local   # radial/angle/wall/combined R/K ladder (fixed L)
python run.py --mode bfzi-convergence --basis-angular # polar/azimuth composite Gauss R/K tests
python run.py --mode bfzi-convergence --basis-geometry # enclosure-projected polar Gauss refinement
python run.py --mode bfzi-convergence --basis-radial   # physical radial panels, both K weak forms
python run.py --mode bfzi-convergence --basis-kparts   # sphere vs box-wall partition of K weak-form error
python run.py --mode bfzi-convergence --basis-wallpanels # physical face/edge Gauss panel refinement
python run.py --mode bfzi-convergence --basis-verify  # optional costly refined R/L/K solve
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
```

**BFZI held-out model evaluation (2026-11-02):**
The original 8-geometry dataset has only 6 training geometries and 2
held-out scenes, both outside at least one coordinate's training min/max.
This high-dimensional bounding-box flag does **not** by itself quantify
physical extrapolation distance. The saved 27-mode labels remain
provisional. Measured mean raw errors were Z=9.04%, J=79.60% and
local probe Joule=92.71%. Reconstructing Re(Z) from the poorly learned
current field increased Z error to 18.63%, despite a tiny algebraic
power-identity mismatch. **Raw independent port Z is now the inference
default**; pass `--power-closure` only to inspect the current-derived
passive reciprocal loss and its spatial energy identity. The old
`--raw-port-loss` flag is accepted as a redundant compatibility alias.
`bfzi-eval` reports both raw and closed accuracy, plus
`raw_mode_relative_error` and `raw_power_mismatch` on the same positive
cubature. These diagnostics are not a continuum accuracy certificate.

The 159-mode uniform-ball reference has *not* converged in its physical
trial space (95→159: dZ=23.9%, probe dq=50.5%). The
`--basis-ball-check-l` command now independently tests magnetic rule B
against A on the **same** 159-mode, same refined-R/K operator and fixed
probes: it prints dL/dZ/dJ/dq/dP and the R-inverse-weighted B-equation
residual of the A solution. The gate remains experimental and must be
run; a PASS would only validate that particular A/B comparison, not
the complete reference. The optional `--basis-ball-face` study adds
eight real solenoidal modes `curl(g C_port)` with complete physical
port potential and independent broad-face envelopes (159→167). It
reuses the nested parent solve to measure whether these **physics-guided**
directions matter for field/port accuracy; it neither replaces the
existing 159-mode benchmark by default nor claims convergence. Combine
both flags to run A/B on the 167-mode parent.

**Follow-up numerical evidence (54 passed, 2026-11-02):**
The independent 159-mode magnetic A/B comparison found
`dL=0.092214`, but driven `dZ=0.000286086`,
`dJ=0.000570731`, `dq=0.000191601`,
`dP=0.0000224192`, maximum individual probe `dq=0.003429`,
and B-system residual of the A solution `0.000895751` in the
R-inverse norm. Therefore the old combined magnetic gate **fails on
whole-operator dL**, while the physical *driven response* meets the
chosen A/B thresholds. The command now prints
`operator_gate`, `response_gate`, and `combined_gate` separately;
it does not silently lower the original operator threshold.
A/B response agreement does not establish convergence against a
third rule, finite-volume cell shape, outer-domain extension, or
further trial-space refinement. In fact, 95→159 still produced
`dZ=0.239214` and `dq=0.50494` (local max `dq=2.46521`).
Residual-driven enrichment needed **all 159 modes** to satisfy the
specified local-probe target. The rank-4 real snapshot reconstruction
is exact only relative to that unconverged parent.

**Physical-field-aware neural training experiment:**
The previous 27-mode MLP training minimized mean standardized output
MSE, not actual spatial field error. A sample with 21.37% mean raw
coefficient error produced 79.60% mean J and 92.71% mean local Joule
errors, so the measured field must affect training explicitly.
`bfzi-train` now uses a mixed objective with separately normalized
port Z and modal supervision, plus losses on the two complex
port-current fields and a fixed complex-drive local Joule field at
geometry-following near-package and bulk probes. The weights are
configurable via `--field-loss-weight` (default 1) and
`--joule-loss-weight` (default 0.15); pass both as zero for the
port/modal-only comparison. `bfzi-train` returns
`heldout_probe_j_relative_error`,
`heldout_probe_q_relative_error`, and `validation_objective`
alongside the earlier metrics. The new loss is currently enabled
only for the **explicitly reconstructible 27-mode v1 topology**;
other topologies still use the coefficient/port objective until
labels persist the complete physical-basis construction.
**Measured retraining outcome (2026-11-02, 54 passing tests):**
At 400 requested epochs, validation selected epoch 25 on the original
6-train/2-holdout provisional dataset. Using
`--field-loss-weight 1.0 --joule-loss-weight 0.15`, held-out
mean raw field J error fell **0.7960→0.7033** and local Joule error
**0.9271→0.8120**, while raw port Z error rose
**0.0904→0.0936**. With only two holdouts, both outside at least
one training coordinate's observed range, this is evidence that the
new spatial objective helps on these particular labels, **not**
evidence of generalization to arbitrary poses, package sizes or media.
The source teacher remains the same unverified 27-mode one. Further
training-data expansion should follow, not precede, physical teacher
convergence. The training summary also includes
`heldout_drive_j_relative_error` and
`heldout_drive_q_relative_error`, defined as the **mean of per-geometry
relative norms under the same complex port drive and probes as
`bfzi-eval`**. These are directly comparable to `raw_j_relative_error`
and `raw_probe_q_relative_error`. Older
`heldout_probe_j_relative_error` and
`heldout_probe_q_relative_error` are RMS loss summaries across both
unit-port fields and must not be numerically equated to driven holdout
means.

**Measured 159→167 face-guided enrichment (2026-11-02):**
On the same high-resolution streamed R/K and original magnetic A
uniform-ball rule, the eight source-face guided curl modes changed
`dZ=0.000968301`, aggregate probe `dq=0.0165902`,
and `dP=0.00103697`. Maximum single-probe
relative `dq=0.797511` occurs at a cooler location
(reference q ≈3.69e-4 W/m³, change ≈2.95e-4 W/m³).
The worst **absolute** heat difference is about 2.82% of the parent
peak q; at probes at least 10% as hot as that peak the largest
pointwise relative difference is about 3.23%.
These are different diagnostics: a small global heat norm does not
guarantee every local thermal query is accurate. The 167-mode parent
still lacks an independently run B magnetic integral and a stable
further physical-basis hierarchy. **Do not promote it to trusted
cross-geometry label generation.**

The residual-driven greedy comparison now continues to the true end
of the parent (including the 167th direction), rather than hitting
the former fixed 13-iteration 159-mode limit. It additionally reports
`hotspot_max_dq` (probes whose reference heat reaches at least 10%
of its maximum), `peak_scaled_max_dq` (worst absolute difference
normalized by the peak), and the original uncompromised
`max_probe_dq` strict acceptance criterion. A
`hotspot_match=True` is an informative weaker diagnostic, not
a strict success or a continuum certificate.

Next run, if needed after the inexpensive tests:

```bash
python run.py --mode bfzi-convergence --basis-ball-face --basis-ball-check-l
```

This keeps the 167-mode physical trial space and refined R/K fixed
while comparing independent magnetic A and B cubatures. The run
still performs expensive Green assembly; no A/B result for the
167-mode parent is currently available.

**Independent 167-mode A/B result (2026-11-02; 54 tests passed):**
The user's completed `--basis-ball-face --basis-ball-check-l` run
confirmed `dL=0.103331` (whole-operator FAIL) but
`dZ=0.000286255`, `dJ=0.000576561`,
`dq=0.000189255`, `dP=0.000022392`, maximal
individual probe `dq=0.00314065`, and B-on-A R-inverse
residual `0.000897201` (driven-response PASS). These independent
magnetic rules agree strongly on the driven response, but neither
rules out shared bias nor certifies physical completeness. The fixed
167-mode response compression has rank four in real coordinates only
at the same geometry/frequency.

The repaired residual-driven 167-mode selection completed at
`adaptive_13` and needed **all 167 modes** for the strict
all-probe ≤5% criterion. The 151-mode subset had
`hotspot_max_dq=0.00214038` and
`peak_scaled_max_dq=0.001425` (hotspot match), but still
`max_probe_dq=0.264188` because cooler locations respond
differently. Keep strict and hotspot criteria distinct.

**New optional 167→199 nested edge-lip trial:**
The next unresolved physical axis is the independence of near-perimeter
seawater current patterns. `--basis-ball-edge` keeps the complete
167-mode face-guided hierarchy and adds four narrower localized
Gaussian curl patches along the outer midpoints of each broad face,
for two tangential current directions and two boxes: 32 new real modes.
The generated modes remain curl potentials with the same insulating
wall and outer-boundary constraints. The diagnostic reports Z, **complex physical current-field dJ**,
all near-wall heat probes, hotspot/max-absolute heat sensitivity,
power and R-inverse equation residual against the unchanged 167
subspace. Optional `--basis-ball-check-l` also compares independent
magnetic quadrature rules within the full 199-mode space, reporting
`dReZ` and `dImZ` separately: an impedance dominated by resistance
can have a tiny combined `dZ` while the relative reactive impedance
error is much larger. These componentwise values are descriptive,
not a new operator-convergence certification gate.

This 199-mode experiment is **not yet run**, and should not be treated
as an accuracy improvement until the user executes it. The reduced
`dZ` from 159→167 alone does not prove spatial convergence;
the large 95→159 response change is the reason to test additional
independent near-edge directions, not to train more provisional labels.

**199-mode edge-lip outcome (2026-11-02; 55 passing tests):**
The user's `--basis-ball-edge` run completed 199 modes and
confirmed the extra 32 edge-lip directions are physically significant.
Relative to 167 modes, the 199-mode response had
`dZ=0.00292632`, complex driven `dJ=0.0512951`,
aggregate local `dq=0.0982027`, `dP=0.00299008`,
hotspot maximum pointwise `dq=0.110172`, and
peak-normalized maximum absolute heat change `0.0969358`.
Thus 167 modes **did not converge in current or near-wall Joule
heating**, despite much smaller integrated port changes. 199-mode
selective enrichment required all 199 modes for the strict fixed
probe criterion. The 175-mode `hotspot_match=True` did not pass
the strict all-probe test. Neither 199-mode magnetic A/B nor any
continuum physical convergence is established by these numbers.

**Rotated anisotropic source-face current trial (199→207):**
`--basis-ball-aniso` preserves all 199 older modes and adds eight
`curl(g C_port)` modes, two broad faces and two face-normal envelope
widths per package/port. In contrast to the older isotropic Gaussians,
their physically sized **tangential** footprint and independently
narrow **normal** footprint are represented by a positive definite
Gaussian precision matrix `G=R diag(1/w_x²,1/w_y²,1/w_z²) Rᵀ`,
which rotates with the complete package SE(3) pose. The derivative
of the same Gaussian appears in both its potential primitive and
its exact analytical curl; the old isotropic path is unchanged.
This separates previously confounded normal boundary-layer error
from lateral edge-pattern deficiency.

The nested diagnostic now also compares the individual complex
**unit-port current fields** (`dJ_port0`, `dJ_port1`,
`dJ_ports`), rather than relying only on one combined phasor
drive which may accidentally cancel errors. The magnetic A/B
response gate includes both-port current-field consistency.
**Measured 199→207 response (2026-11-03, 56 tests passing):**
The added normal/tangential anisotropic curls yield
`dZ=0.00149159`, `dJ=0.0303386`,
`dJ_ports=0.0308608`, `dq=0.030114`,
and `dP=0.00157138`. Hotspot maximum pointwise heat
changes by 6.60%; the largest per-point relative change is
4.07843 at fixed probe `[0,0,0.22]`, where q changes from
`8.11357e-6` to `4.12042e-5 W/m^3`.
This is a significant **sensitivity to missing trial directions**,
not a demonstrated improvement in accuracy. Residual-driven
selection still requires 207/207 modes for the strict all-probe
criterion, and the new full parent is not continuum-certified.
The independent 207-mode magnetic or local R/K A/B comparisons have
**not yet been measured**.

**Integrated electromagnetic current and normal-profile convergence:**
The nested modal, adaptive and magnetic A/B reports now include
`dJ_volume` for the combined port drive and
`dJ_ports_volume` for both unit-port fields. These are
resistance-weighted physical current norm ratios:
`sqrt(delta_u^H R delta_u / u_ref^H R u_ref)`, using the
same *refined* positive seawater volume R integration (not an
additional solver mesh or a small collection of spot samples).
They measure an approximate full-seawater L²(J/sqrt(sigma)) error
**relative to the chosen reference solution**, not a continuum
error certificate. The magnetic driven-response gate also requires
both volume-norm errors to be ≤1%.

A separate 40-point `*_normal_sweep` follows the outward normals
of both package broad faces at several distances and two tangential
positions. Its heat comparison is reported independently from the
original 16 fixed probes (whose earlier values are not redefined).
This targets the large near-receiver normal-direction sensitivity
without altering the old convergence series.

Use `--basis-ball-aniso --basis-ball-check-rk` for one **independent
normal integration** refinement (A order 5 vs B order 6): it keeps
all 207 basis directions, Green L, radial and tangential integration
settings fixed, streams the new R/K integrals, and prints
`dR`, both forms of `dK`, Z, pointwise/Joule, full-volume J,
and power differences. It does not check all spatial integration
directions. The existing `--basis-ball-check-l` instead changes
only the nonlocal magnetic Green rule; it is independent from the
new local-R/K option. These expensive studies may be run separately.

**Measured normal R/K verification (2026-11-03; 57 tests passed):**
The user's 207-mode fixed-basis face-normal A/B (Gauss 5→6) found
`dR=9.821e-5`, `dK=1.23374e-5`,
`dZ=0.000133542`, full-volume driven
`dJ_volume=0.000201619`, both-port
`dJ_ports_volume=0.000226541`, local
`dq=0.000116462`, and peak single-probe
relative difference `0.000716925`: the local-response
gate **passed**. This supports stability of the chosen face-normal
R/K cubature at 207 modes. It does *not* settle the independent
magnetic L operator, other quadrature axes or continuum basis error.
In contrast, **199→207** previously changed full-volume J by
`0.0396445` and both-port J by `0.0387992`, and normal-profile
hotspot heating by up to `0.0623939` relative to its peak. These
are material changes in the trial space, not normal-Gauss noise.

**Finite-parent error estimate and leave-one-mode diagnostic:**
The nested and residual-driven greedy outputs now include
`parent_dJ_energy_upper`, a coercivity-based upper bound on the
**relative current R norm error against the fully solved, fixed
finite-dimensional parent**. With `A=R+iωL`, `A u=-f` and
residual `r=A u_trial+f`, the absolute R-energy error is bounded
above by `sqrt(r^H R^{-1} r)`; the relative upper bound uses
`||u_trial||_R - ||r||_{R^{-1}}` as a conservative lower bound
for the unknown parent solution's R norm. `inf` means the bound
is inconclusive. It is deliberately not advertised as a continuum
a posteriori error certificate or a bound for numerical quadrature
bias.

Add `--basis-ball-ablate` (for example, together with
`--basis-ball-aniso`) to rank existing physical modes by the
result of **deleting one mode and re-solving all other Galerkin
equations**. This uses inverse-column rank-one corrections, not
207 separate Green matrix assemblies. Reports full-volume
`dJ_volume`, both-port `dJ_ports_volume`, peak-scaled
fixed-probe and 40-point normal-profile heat differences; top six
modes are listed for current energy and normal heating separately.
This is a diagnostic of which **included** modes matter.
Collective/correlated mode contributions cannot be inferred from
single-mode ablation; neither can the influence of as-yet missing
physical directions. The adaptive `SPARSE_MATCH` acceptance gate
now also requires both the combined-drive and both-unit-port
`dJ_volume` metrics to be at most 1% relative to the fixed parent,
in addition to the pre-existing port, power, aggregate heat and
all-probe criteria. The R-norm bound remains an independently reported
conservative diagnostic, not an assumed continuum certificate.

**207-mode independent magnetic A/B measurement (2026-11-03):**
For the same complete 207 physical modes and fine R/K, the new
uniform-ball magnetic B rule (15,154 nodes) gave `dL=0.126388`,
`dZ=0.000295995`, `dImZ=0.0128088`,
`dJ_volume=0.000875828`,
`dJ_ports_volume=0.000953918`, local `dq=4.45415e-5`,
and B-on-A R-inverse residual `0.000954538`.
The **operator gate FAILED**, while the currently driven **response
gate PASSED**. In particular the 1.28% reactive impedance sensitivity
is larger than the full-complex Z difference. This suggests that the
big 199→207 physical-field change is not simply an artifact of this
single magnetic cubature perturbation; it does **not** certify the
Green operator or the continuum response.

**Optional targeted 207→213→229 physical-space holdout (not yet run):**
`--basis-ball-targeted` retains the current 207 directions, adds six
independent large-scale curl modes at two physical scene length scales
(213), then sixteen anisotropic, pose-following complete-port curls
localized at the broad-face package corners (229). This controls for
both the prominent global direction and near-corner heat sensitivity
seen in the single-mode ablation experiment without replacing the
previous reference defaults or claiming that 229 directions are
more accurate. The same streamed R/K and magnetic A rule are used
for every nested stage; acceptance must compare *full-volume current*
and the package-normal heat profile, not merely Z. The flag may be
computationally expensive and requires local tests before interpretation.

**Do not generate cross-geometry high-fidelity labels from the
207-mode parent yet.**

The new API imports `BFZICoil`, `RigidPose`, `InsulatingBox`, `BFZIAssembly`,
`BfziScene`, `build_bfzi_source`, `solve_seawater_reference`,
`compress_seawater_response`, `BFZICurrentControlledEnvelope`, and
`BFZIVoltageControlledEnvelope` from `sdfmpneo_bfzi`.
Both ports require explicit return conductor routes wholly inside the sealed
boxes, and the finite wire diameter must be smaller than the nominal turn pitch.
The BFZI branch does not rename or feed its path into the legacy superellipse.

Physical scope: two sealed nonconductive nonmagnetic boxes, static SE(3)
poses, finite spherical conducting seawater domain, fixed frequency and
a divergence-free Galerkin seawater current basis. The original positive
meshless spherical-ray cubature remains available with
`SeaMQSConfig(quadrature_scheme="ray")`, including exact box-wall ray cuts
and optional `wall_layers`. **The new default is
`quadrature_scheme="box_shell"`**: disjoint rotated-box Gauss shells cover
every insulating face. A smooth, nonnegative partition of unity blends
the localized shell integral with the **full** spherical integral: the
sphere receives zero weight inside insulation and smoothly regains unit
weight away from the box; the local shells carry the complementary weight.
The spherical radial rule is further split around package distances.
This avoids hard angular masking of the expanded boxes and resolves
thin-package wall modes without depending on a coarse ray intersecting
a particular face. `shell_normal_order`, `shell_tangential_order`, and
`shell_width_factor` control near-wall refinement independently of
`radial_order`, `polar_order`, `azimuthal_order` for the spherical part.
Tangential Gauss rules are now **composite**: the physical face center and
both box edges are explicit interval boundaries; positive u²-mapped
exterior nodes cluster towards each edge without introducing a world mesh.
This addresses the earlier ~85% sensitivity to changing the tangential
Gauss order from four to five, but must be checked numerically rather than
assumed fixed. Global seawater mode widths also depend on physical package
reach instead of the artificial sphere radius, making domain refinements
less entangled with the basis shape. This uses positive
geometry-local quadrature, not a world spatial solver mesh. The original
run output showed 52% mode sensitivity and 71% cubature sensitivity, so
angular, basis, outer-domain and singular-kernel convergence remain explicitly
**unproven** until the new runs and successive refinements agree. `L_air` uses a softened finite-radius
Neumann approximation. Its self entries now subdivide the *analytic*
continuous BFZI-plus-return path into short panels and integrate each
straight-panel diagonal analytically, avoiding the strong coincident-node
bias of low-order line Gauss integration. The panel size can be refined via
`build_bfzi_source(..., air_panel_length_factor=2.0)`; the default is 4.0
wire radii per panel. This is still not a fully resolved GMD or
cross-section magnetic solution.
The seawater heat source integrates to the seawater port channel **under the
same cubature**; Green-kernel assembly remains memory-bounded in blocks.

The **ordinary** convergence probes now use the same 27-mode/two-tangent
trial space as the BFZI demo; the one-tangent 15-mode space is reported only
as a deliberately lower-order comparison. The dedicated `--basis` probe
uses one enriched reference assembly and exact Galerkin restriction to
compare 15, 27, 59 and 63 nested curl modes with *identical* volume
cubature, softening, outer boundary and source quadrature. The 27-to-59
enrichment adds 32 localized modes on the broad faces of thin insulating
pads. The 59-to-63 enrichment adds four physically informed, real,
source-guided solenoidal modes, two per complete port, using
`curl(g A_port / mu0)` with a box- and sphere-vanishing smooth envelope.
The actual closed BFZI coil-plus-return vector potential supplies the
direction field, not an arbitrary constant vector. These modes retain
the zero normal trace at insulating surfaces and global solenoidality.
They are optional in `build_sea_basis(scene, sources=source,
include_source_guided=True)` and do not silently change the standard
27-mode demo. The full all-face 123-mode hierarchy remains optional.
The same `--basis` run now uses **fine primary cubature**
`(radial,polar,azimuthal,shell_normal,shell_tangent)=(6,10,16,4,5)`
to assemble `R`, `K` and positive spatial heat nodes, while its expensive
nonlocal `L` double integral uses the *same physical modes and whitening*
but the earlier `(5,8,12,3,4)` points. This avoids increasing quadratic
Green-matrix cost merely to resolve locally sharp resistive/source terms.
The standard single-quadrature reference solver remains unchanged unless
`SeaMQSConfig(inductance_quadrature=(5,8,12,3,4))` is supplied.
It is a controlled approximation: independent verification of the
nonlocal `L` is still required.

The `--basis` run then checks the 63-mode result on **two
independent finer cubatures**: one refining the near-wall normal/tangential
rules, another refining the outer-sphere radial/angular rules. In addition
to changes in `R`, `K`, and the **frozen-current** Joule integral,
`SeawaterResponse.reintegrate_local_response(points, weights)` now
re-solves the two-port system using held-out `R/K` and **the original
Green `L`**, without any new `O(Q²)` magnetic assembly. This reports
`dZ`, fixed-probe `dq`, and `dP` for the changed local operators,
which are more directly meaningful than a full matrix-norm difference.
These are same-L numerical sensitivity results, not a new truth solution.
The independent `--basis-green` sweep uses the exact original
`R/K` and heat cubature, but recomputes and re-solves the symmetric
Green `L` on alternative magnetic quadratures. At one fixed magnetic
cubature it also compares 25, 18 and 12 mm kernel softenings. The
underlying Green distances are shared between the alternate softenings.
This isolates `dL`, induced `dZ/dq/dP` and the original-current
equation residual. **Softening must not be extrapolated to zero at fixed
cubature:** point-collocation self terms scale like `w_i²/a` and may
produce an artificial divergence, even though the continuum Coulomb
double integral is finite. Refine the magnetic rule jointly with `a`.
**Outer-boundary basis fix (2026-11-02).** Earlier domain scans were
confounded by the factor `1-r²/R²` multiplying every current trial
function: moving the artificial sphere changed the basis throughout
the charging region. `build_sea_basis` now defaults to a compact
`boundary_layer` spherical cutoff with a fixed physical transition
width determined by enclosure reach (or explicit `outer_taper_width`).
Within the common physical interior both the cutoff and its gradient
are exactly 1 and 0 respectively, so raw near-pad trial modes are
invariant under outer radius changes. The legacy `quadratic` option
remains selectable for comparisons. The `box_shell` radial rule now
splits exactly at the cutoff transition onset; the R/K and independent
Green L cubatures use the same split. `--basis-domain` prints
`raw_mode_dW` at fixed near-pad probes to expose unwanted trial
space changes. This updated domain sequence **must be re-run**;
results cannot be numerically compared to prior outputs as if only
the outer radius had changed, because the basis construction changed.
The compact cutoff also imposes a stronger zero-current trace on the
artificial boundary, and still needs convergence against larger
domains and richer long-range modes.

The separate `--basis-domain` mode solves the same 63-mode *construction*
on outer radii 0.52, 0.70, 0.85 and 1.05 m. Those are distinct
boundary-conditioned trial spaces, so domain changes are not a
fixed-matrix L-only test. Both switches can be costly; neither certifies
the infinite-domain limit.

**Local R/K convergence extension (2026-11-02).** A local test on
the previous reference returned resolved port changes of **1.6657%**
for near-wall refinement and **2.8853%** for outer sphere quadrature
refinement, even though the same-L energy algebra closes. The new
`--basis-local` option therefore independently varies outer *radial*
order, outer *angular* orders, near-wall orders, then computes two
combined progressively finer R/K solutions with the **same 63 physical
trial modes, same Green L, and same source**. It reports port `dZ`,
probe `dq`, and seawater power `dP` against the original and between
successive combined stages. Only the combined stages form a
monotonically refined quadrature hierarchy; the single-axis rows are
diagnostic, not a sequential convergence curve. These refined responses
are R/K-only references, not certified continuum solutions.

**Composite angular sphere integration (2026-11-02).** On the
35-test-passing local `af9f56d` reference, changing just the outer
angles caused `base_dZ=0.0878657` and `base_dP=0.0998883`.
The combined R/K fine-grid change was `step_dZ=0.0109989`,
`step_dq=0.187754` and `step_dP=0.0129961`, so the original,
nonmonotonic single-axis comparisons do **not** establish convergence.

The sphere component of `box_shell_quadrature` now supports two
optional controls: `polar_panel_count` partitions the cosine
polar coordinate into equal Gauss panels (must divide the total
`polar_order` to preserve north-south symmetry), and
`azimuthal_panel_order` uses Gauss-Legendre points inside each fixed
periodic azimuth sector. Default `(1,1)` preserves the historical
global Gauss-mu / single midpoint-phi rule. Slab integration near the
box walls is unchanged. These parameters are available in
`SeaMQSConfig`. They apply to primary R/K and to Green L only when
Green L uses the same primary cubature. With explicitly independent
`inductance_quadrature`, Green L uses its own angular rule.

The latest local `aa958f5` run passed **36 tests** and reported
`polar_split2` `base_dZ=0.154019` and `base_dP=0.177437`.
The mixed refinement `polar2_phi3` to `polar2_phi3_fine` changed
`dZ=0.138306`, `dq=0.0818322`, `dP=0.16552`.
Thus just repartitioning ten total polar points is **not** equivalent
to refining a quadrature, and a polar/azimuth simultaneous change
does not isolate which direction still needs more accuracy.

The updated `--basis-angular` first checks midpoint-versus-Gauss
azimuth, then runs the *same* two-polar-panel, two-Gauss-points-per-
azimuth-sector rule at total polar orders **10, 14, 18 and 22**,
with radial order, sector count, shell orders, mode basis, source and
Green L unchanged. Finally it separately compares azimuth Gauss order
two versus three at the same 22 polar nodes. Each row reports
`dR/dK`, re-solved `dZ/dq/dP`, and four absolute
`probe_q` values; each following row additionally prints
`step_from_previous` and the maximum per-probe relative change.
The run additionally prints `K_only_re_solved` using the *same*
new coupling integral but the original R, Green L and spatial Joule
nodes. This isolates how much of each response change is attributable
to the source-coupling matrix rather than the resistive operator.
`SeawaterResponse.resolve_with_coupling(K)` exposes this same passive
re-solve for externally reintegrated two-port coupling matrices;
matching R and heat nodes still ensure exact discrete energy closure.
It does **not** make an arbitrarily supplied K physically accurate.

Only the polar-order 10/14/18/22 rows form a meaningful
single-axis order refinement. The Gauss nodes are not mathematically
nested, so small consecutive differences alone are not a rigorous
error bound. The old quadrature stays as the default.

Composite Gauss angular rules cost extra spherical field evaluations.
In particular, small total port-power errors can hide much larger
local heat-source changes; choose reference orders from verified
pointwise and port stability, not node count alone.

**Geometry-aligned polar R/K refinement (2026-11-02).** The
37-test-passing `10a615d` run showed that varying two fixed polar
panels' Gaussian degree from 14→18→22 changed the re-solved port
`dZ=0.0483211` and `0.0243709` respectively, while changing
azimuth Gauss order from 2→3 at 22 polar nodes gave only
`dZ=0.000767303`. Almost all of the polar-induced port change was
reproduced by K-only re-solves; the pressing problem is **geometry-
localized source coupling K angular integration**, not a
generalized lack of outer-sphere azimuthal resolution.

The new optional `polar_geometry_order` for
`BfziScene.box_shell_quadrature` and `SeaMQSConfig` partitions the
cosine-polar interval at the minimum/maximum projected polar angles
of the actual oriented insulation-box corners **and** the same
corners on their partition-of-unity blending shells. The panel
boundaries depend on the physical scene and box-shell blending width,
not on the chosen Gauss integration order. It uses exactly
`polar_geometry_order` positive Gauss nodes **per geometric
interval**; this is distinct from the global `polar_order`
parameter, which is ignored for this one optional polar rule.
Geometry panels cannot be combined with uniform `polar_panel_count>1`.
R/K honor this option. Green L inherits it only if no separate `inductance_quadrature` is specified; an independent Green L quadrature keeps its own lower-cost angular rule.
Default behavior remains unchanged.

`python run.py --mode bfzi-convergence --basis-geometry`
holds the original 63-mode basis, source, R whitening, Green L,
radial and box-wall cubature, and azimuth sectors fixed. It tests
per-panel polar Gauss orders 3, 4, 5, 6, 8 at azimuthal Gauss
order 2, then varies only azimuth Gauss 2→3 at polar order 8.
It reports `dR/dK`, baseline and consecutive-step
`dZ/dq/dP`, maximum individual probe `dq`, and separate
**K-only and R-only** re-solves. This is a hypothesis about
the dominant numerical error mechanism, **not** evidence
that the geometry-panel approach is converged until local results
verify it. More panels can increase field-evaluation cost.

**After 38 passing tests: K polar convergence and next error source.**
The user-run geometry-aligned integration sequence on `4d02f50`
showed `geometry_mu6→geometry_mu8` changes
`dZ=0.000242124`, `dq=0.000444726`,
`dP=0.000296029`; `geometry_mu8` with azimuth
Gauss order two versus three gave `dZ=0.000413271`,
`dq=0.000109597`, `dP=0.000500205`.
This strongly supports convergence of the **spherical angular
portion** of primary R/K at the fixed 63-mode basis, Green L,
radial and wall cubatures. It does **not** certify R/K, Joule
or full electromagnetic reference convergence. Relative to the
old global-angular reference, the converged-in-angle
`geometry_mu8` response still differs by ~6.38% in port Z:
the original reference had a substantial angular bias.

`--basis-geometry` now also re-integrates the same
`geometry_mu8` solution with radial order 6→7,
box-wall orders 4/5→5/6, and both together. These held-out
comparisons **all use geometry_mu8 as the baseline**, not
the old under-resolved global-angular reference.
They reveal which one-volume R/K errors remain after
the angular bottleneck is addressed.

**Important Green cost separation.** When
`SeaMQSConfig.inductance_quadrature=(radial,polar,azimuthal,
shell_normal,shell_tangential)` is explicitly given, the
independent Green L cubature now uses exactly its own
global Gauss-mu/midpoint-azimuth angular rule,
**not** the R/K `polar_geometry_order`,
`polar_panel_count` or `azimuthal_panel_order`.
Otherwise, enabling high-resolution R/K geometry panels
would accidentally increase L double-integration work from
thousands to tens of thousands of nodes, causing a
quadratic-cost explosion. The outer radial panel cap is
still shared, and the physical trial modes and R-derived
whitening remain identical. Without an independent L
configuration, L still uses the primary R/K nodes; specify
`inductance_quadrature` for a practical high-resolution
geometry-R/K reference. Green L convergence and softening
must be verified separately.

**Integration-by-parts / curl-dual source coupling K (2026-11-02).**
In the user-run `4363d9a` reference, **39 tests passed**.
With geometry-angle quadrature held at `geometry_mu8`, radial order
6→7 changed `dZ=0.0230224`, `dq=0.0519701`
and maximum individual probe q by **10.19%**.
Increasing box-wall Gauss orders 4/5→5/6 instead changed
`dZ=0.0170344` and `dP=0.0193525`.
The combined variation was `dZ=0.040216`,
`dq=0.050943`. Therefore, **angular convergence does not mean
all R/K components have converged**.

There is an exact (continuous) integration-by-parts identity for
every BFZI curl trial current `J_m=curl(F_m)`, because its vector
primitive `F_m` vanishes on insulating-box and artificial
outer-domain boundaries:

```text
K_mp = mu0 * integral_sea J_m dot C_port dV
     =       integral_sea F_m dot B_port dV,
B_port = mu0 * curl(C_port)
```

The second expression has no direct spatial derivatives of the
rapidly changing box-wall mode envelope, and may offer much smoother
K cubature. The new `SeaCurrentBasis.potential_primitive(points)`
returns the actual primitive for every constant-vector and source-guided
mode, without changing the physical trial space.
`SeaMQSConfig(coupling_form="curl_dual")` opts into the second
form for a complete reference solve;
`coupling_form="current_potential"` is still the default.
`SeawaterResponse.reintegrate_source_coupling(points,weights,
coupling_form="curl_dual")` independently integrates K on chosen
nodes; `resolve_with_coupling(K)` can then re-solve using unchanged R,
L and heat nodes. The `--basis-geometry` run now compares
**both coupling formulas** on `geometry_mu8` and on the
independent radial/near-wall held-out cubatures, printing
`curl_dual_K` step changes in port Z, pointwise heat and power.

This is **not** a new electromagnetic model: both K forms have the
same continuous limit when the boundary traces vanish, and the
discrete port/volume energy closure still holds. Comparing them
provides an independent convergence indicator, not a guarantee
that one form is already converged. Avoid choosing the final
coupling form until the measured independent radial/near-wall
refinements are stable.

**Radial geometry cuts and two-form K convergence (2026-11-02).**
The user-run `8f46998` experiment passed **41 tests** and
showed that the new curl-dual K formulation does not automatically
cure all one-volume integration error. With the same `geometry_mu8`
polar rule, updating only radial Gauss degree from 6 to 7 changed
the historical K solution by `dZ=0.0230224` and maximum
probe `dq=0.101934`, versus `dZ=0.0369192` and maximum
probe `dq=0.219208` under the dual weak form. At the **same**
original `geometry_mu8` nodes, the two continuous-equivalent
formulations differed by `dK=0.0242518`,
`dZ=0.0356102`, `dq=0.038667` and
`dP=0.036494`. It is **not** justified to switch the default
source-coupling integration until radial/wall discretization is
controlled and the two forms approach one another.

`box_shell_quadrature(..., radial_geometry_cuts=True)` now optionally
inserts **actual box and enlarged blending-shell radial extents**
into the existing spherical integration intervals. For each oriented
enclosure it finds the exact minimum sphere-center-to-box distance,
plus the maximum sphere-center-to-corner radius. This remains
geometry-driven for arbitrary poses and does not introduce a
world volume mesh. Combined with the existing
`max_radial_panel_width` in meters, sphere Gauss panels keep
controlled physical radial sizes. These options do not alter
the separate geometry-local box-wall slabs. The new
`SeaMQSConfig.radial_geometry_cuts` is applied to the primary
R/K/Joule quadrature; an explicitly independent Green L
quadrature retains its own less costly radial cuts. Legacy
behavior remains the default.

`python run.py --mode bfzi-convergence --basis-radial`
builds the original 63-mode source/Green L just once and
then independently compares the `geometry_mu8` radial reference,
new actual box radius cuts, and caps of **0.12, 0.06, 0.03 m**.
For every integration it re-solves R/K using the historical
curl-current K and the boundary-dual primitive K, holding
Green L and physical modes fixed. It reports differences
between the two forms on **identical R and nodes** and
consecutive fixed-geometry radial-stage differences in
`R/K/Z/q/P`, including the largest individual probe
heat change. The last row compares a radial Gauss-degree
6→7 increase on the **same** 0.06 m panelization,
rather than confusing that change with a 0.03 m cap.
A small form gap and small same-family successive changes
are both necessary practical indicators, but do not prove
whole-model convergence. This branch has not yet had the
new radial command validated by the user's local pytest.

**After 42 passing tests: localized K integration error (2026-11-02).**
The user-run `307b86e` experiment confirmed physical radial
panel convergence. With exact geometry cuts and geometry mu8 Gauss
panels, shrinking radial panels from 0.06 m to 0.03 m changed
the curl-current K solution by `dZ=0.0000659457`,
`dq=0.0000627926`, `dP=0.0000764537`,
and the dual K solution by `dZ=0.0000391847`,
`dq=0.00003342`, `dP=0.0000440785`.
But the two mathematically equivalent K forms did **not** approach
each other: their discrepancy stabilized at
`dK≈0.04567`, `dZ≈0.03771`, `dP≈0.03020`.
This rules out simply increasing radial integration orders as
a productive next step. The source Biot-Savart field and vector
potential use the same point-line discretization; their analytic
curl relationship is implemented by cross products consistently,
but their **integrated** weak-form equality still depends on
sufficiently accurate near-box and full-sphere integrations.

`BfziScene.box_shell_quadrature(...,return_sphere_count=True)`
now optionally includes a third result with the number of
spherical nodes at the beginning of its output. The remaining
nodes are the disjoint six-face box slabs. This is an optional
diagnostic output; ordinary callers still receive only
`(points,weights)` and their integration results are unchanged.

The new dedicated `--basis-kparts` command holds the same
63-mode physical basis, independently coarse Green L, geometry
radial cuts with a 0.06 m cap, polar geometry mu8 and
2-point azimuth-sector Gauss fixed. It increases **only** the
near-wall Gauss normal/tangential orders through 4/5, 5/6,
6/7 and 8/9, then changes only the geometry polar order
8→10 as a separate angular heldout. Each row reports
`K_gap_by_region`: the normalized contributions to the
curl/dual K difference from spherical and near-wall slab
integrations, their signed alignment/cancellation, the
full weak-form impedance/heat/power discrepancy and each
weak form's successive response change. This identifies the
**numerical location** of the persistent weak-form mismatch;
it does not prejudge which method is correct. It is much more
targeted than broad brute-force refinement, though the
seawater point evaluations are still substantial.

**Near-wall R/Joule convergence and composite face panels (2026-11-02).**
The user-run `dd16b7c` experiment completed **43 tests** and
localized the two K weak forms' discrepancy at the
sphere/box-slab partition. The normalized weak-form differences
in **each region** are around 0.82 of the total K norm but
almost perfectly anti-aligned (`alignment≈-0.999995`).
These are large complementary *subdomain integration-by-parts
boundary terms*, **not** independent 82% physical errors.
The total weak-form K discrepancy falls from 0.0456624 at
wall 4/5 to 0.00367166 at wall 8/9, and the port discrepancy
falls from 0.0377317 to 0.000971325. A separate mu8→mu10
polar refinement at wall 8/9 changes port Z by only
`1.73114e-05` (historical K).

However the near-wall **R** operator changes by 0.205192
between wall 5/6 and wall 6/7, and by 0.085981 between
wall 6/7 and wall 8/9, with significant spatial heat
changes. Thus a small K weak-form gap alone is NOT enough
to certify electromagnetic Joule heating, and increasing
global shell Gauss order is still an inefficient strategy.

An optional **physical tangential half-face and edge-strip
composite Gauss rule** is now implemented in
`BfziScene.box_shell_quadrature(
shell_tangential_max_panel_width=<meters>)` and
`SeaMQSConfig.shell_tangential_max_panel_width`.
The physical box face center and true edges remain
integration boundaries. Each half-face is subdivided into
intervals no longer than the given width; an independent
positive Gauss rule is used on each. Exterior edge/corner
strips retain their positive `u²` clustering near the
insulating edge, now with optional smaller composite
`u`-intervals controlled by the same physical width.
All slab nodes follow each oriented box's local axes,
support arbitrary box poses, and use no global world mesh.
The sphere, primary R/K/Joule physics and independent
Green L remain unchanged. The default `None` reproduces
the previous near-wall rule.

`python run.py --mode bfzi-convergence --basis-wallpanels`
reuses the sphere/slab source K comparison while holding
63 modes, Green L, geometry-mu8, physical radial cap 0.06m
and shell normal order 5 fixed. It compares the old
unsegmented tangential order 6, then 0.05m and 0.025m
physical panel caps at the same order, and finally increases
tangential order 6→7 on the same 0.025m panels. Each
stage prints region-separated weak-form K discrepancies
and step changes in R, K, port impedance, heat probes
and power. At this revision, this improvement has **not
yet** been validated by the user's next local experiment.

**Wall-panel CLI forwarding correction (2026-11-02).**
The user-run `fef1b8b` completed **44 tests**, but the
`--basis-wallpanels` rows `wall_base_5_6`, `wall_cap050_5_6`
and `wall_cap025_5_6` all had exactly 19,024 slab nodes
and exactly zero numerical change. Inspection found a specific
CLI bug: the experiment correctly populated
`SeaMQSConfig.shell_tangential_max_panel_width` but omitted
passing this value into its direct
`scene.box_shell_quadrature(...)` call. Thus those apparent
zero-error comparisons were invalid; the underlying composite
quadrature implementation was not tested in that run.
The experiment now forwards the width explicitly and stops with
a descriptive error if the 0.05m or 0.025m refinement does
not actually increase the number of near-wall slab nodes.
A lightweight solver-path regression also checks that the
main R/K/Joule integration forwards the face-panel width,
while independent Green L remains unaffected. Keep the
old 44-passing report associated only with the previous
revision; the corrected study needs a new local run.

**Wall-panel convergence after 44 passing tests (2026-11-02).**
The corrected local `19686b9` run confirmed that physical
face and edge-strip segmentation is active:
`slab_nodes=19024→38800→123072` for the unpanelled,
0.05m and 0.025m stages, with unchanged 204224-node
sphere. Refining width 0.05→0.025m changed
`dR=0.00740772`, `dZ=0.00187433`,
`dq=0.0376711` and `dP=0.00218495` (original
curl-current K form). Therefore the physical wall heat
distribution was **not yet converged in panel width**.
On the SAME 0.025m panels, increasing tangential
order 6→7 changed `dR=0.000147977`,
`dZ=5.28789e-06`, `dq=0.00126934`
and `dP=5.89506e-06`. The change 6→7 affects
the **interior-face** Gauss order 3→4, while the
exterior-edge Gauss order remains 3, indicating the
dominant old Gauss parity error came from sampling
the broad face patches. Two equivalent K weak forms
also became closer, with `weak_dK=0.0011528`
on the 0.025m, tangential order 7 panels.
Tests: **44 passed in 14.48 s** for that revision.

The follow-up introduces
`SeawaterResponse.reintegrate_local_operators(points,weights,
chunk_size=2048)`, which assembles the *same physical*
R and both K forms incrementally in bounded-size
point batches, without keeping the full three-dimensional
volume-mode tensor in every held-out response. This is
an operator-only interface for convergence studies;
`reintegrate_local_response` still stores full positive
Joule nodes for ordinary field/thermal predictions.
The `--basis-wallpanels` experiment now assembles
the fixed sphere operator once and adds only streamed
oriented-box slab contributions. The sequence compares
width 0.05, 0.025, 0.020 and 0.0167m at fixed
normal/tangential Gauss orders 5/6, then independently
tests tangent 6→7 and normal 5→6 at the same 0.0167m
panel width. It reports R/K/Z, four-probe heat and
power changes for **both** coupling weak forms while
holding the 63-mode space and independent Green L fixed.
The operator-only path is covered by a targeted
full-versus-chunked and sphere-plus-slab regression test.
The new run is **not yet numerically validated**.
Do not claim the thermal source, Green L, or complete
REFERENCE is converged merely because one local
Gaussian degree is stable.

**Independent Green L after physically refined R/K (2026-11-02).**
The user's `f99e73f` run passed **45 BFZI tests**.
At 0.0167m near-wall tangential panels,
`normal5,tangential6` compared with `tangential7`
produced `dR=2.27502e-05`, `dZ=2.98506e-06`,
`dq=9.38626e-05`; separately raising
normal 5→6 gave `dR=0.000151424`,
`dZ=0.000185796` and `dq=8.39262e-05`.
The old and dual K weak forms differed in port Z by
`weak_dZ≈0.00084`. Refining physical face width from
0.020 to 0.0167m still changed port
`dZ=0.00139931` and probe `dq=0.00174193`,
so these are stable local *comparison* values, not
an a priori full-reference error bound.

`SeawaterResponse.reintegrate_green_matrices(
points,weights,softenings=...,kernel_chunk_size=...)`
now returns just the independent symmetric Green L
matrices and the discrete coincident-point diagonal
numerator `D=(mu0/(4*pi))*sum(w_i²*|J_i|²)`.
The self-sampling share on softening `a` is
`D/(a*trace(L_a))`; this is a **collocation-alias
warning**, not a physical energy fraction in the
unsoftened continuous kernel. Multiple softening
lengths reuse pairwise point distances. Existing
`reintegrate_green_responses` reuses this same
assembly interface, retaining its existing API.
One small regression compares Green matrices from
the new interface with the previously supported
full-response path.

The new `python run.py --mode bfzi-convergence
--basis-green-refined` command first streams the
high-accuracy one-volume R/K on geometry-mu8,
0.06m maximum radial panels and 0.0167m
near-wall tangential panels (normal5,tangential6),
without storing a half-million volume J modes.
It then independently varies only Green L:
the existing baseline `(5,8,12,3,4)` with fixed
nodes and softening 0.025/0.018/0.012m, a
wall-integration refinement `(5,8,12,3,5)`
at 0.025m, and an outer-angular refinement
`(5,10,16,3,4)` at 0.025m.
All experiments use the exact same 63-mode physical
basis, R, source, trial-mode whitening and either
original or curl-dual K. They report the Green node
count, normalized L changes, coincident-node
fraction, and **re-solved** port Z, four-probe
heat and sea power changes. It is an intentionally
costly `O(N_G²)` *magnetic* calculation and does
not yet certify any softened/unsoftened continuum
limit. Do not infer the electromagnetic field
reference is accurate merely from passive
dissipation or local R/K convergence.

**Green diagonal alias and cross-rule diagnostic (2026-11-02).**
The user's `e3c0a48` run passed **46 BFZI tests** and
measured independent Green errors on 461648 fixed refined
R/K/Joule nodes. The original 7504-node magnetic rule at
softening 25mm had a *discrete same-node self* share of
`diag_trace=0.189174`. At fixed magnetic nodes,
softening 18mm gave `dL=0.0923051`,
`dZ=0.00261412`, `dq=0.000388053`;
12mm gave `dL=0.224198`, `dZ=0.00632986`,
`dq=0.000927456`, while
`diag_trace=0.30113`. At 25mm, refining just the
Green shell from (3,4) to (3,5) gave
`dL=0.0160046`, `dZ=0.000117128`;
refining Green outer angular nodes from (8,12) to
(10,16) gave `dL=0.0896424`,
`dZ=0.00268619`.
Port Z's relative insensitivity to L is NOT evidence
that the nonlocal magnetic field is converged.

The optional diagnostic
`SeawaterResponse.reintegrate_cross_green_matrices(
left_points,left_weights,right_points,right_weights,
softenings=...)` computes the true
`(left × right)` Green double sum with independently
sampled wet-domain quadratures. It returns the
symmetrized matrix and the number of exactly overlapping
cross pairs; with identical left/right nodes it
reproduces the existing symmetric one-rule Green
operator. With disjoint rules the artificial
`i=j` term disappears. This **does not make the
result automatically more accurate**: changing one
quadrature rule introduces other integration errors.
Furthermore a two-rule symmetric cross matrix is
NOT guaranteed PSD, so this is explicitly **not**
a replacement for the passive default Green L.

Run `python run.py --mode bfzi-convergence
--basis-green-cross` to first stream the same refined
63-mode R/K, then compare the symmetric 7504-node
self-rule A with the symmetric cross-rule A×B.
Rule A is (5,8,12,3,4); independently sampled
rule B is (5,9,14,4,5), each interpreted as
(radial,polar,azimuthal,normal,tangential).
Three fixed softening lengths 25/18/12mm reuse
Green pair distances. The output reports cross
coincidence count, `dL/dZ/dq/dP`, old
`diag_trace`, the **smallest cross-matrix
eigenvalue**, and both original/curl-dual K
re-solves while keeping one fixed fine R/K.
If this discrepancy is large it motivates a
consistent near-pair integration/positive Green
kernel correction, not a blind extrapolation
of softening to zero. This command contains
`O(N_A N_B)` Green work and has **not yet been
run on the user's machine**.

**Green acceptance gate: failed; positive Gaussian-volume candidate (2026-11-02).**
The user's `26dea37` run passed **46 BFZI tests**
but failed the previously agreed magnetic convergence gate.
At 25 mm old Plummer softening, the independent
A (7504 nodes) vs A×B (B=15154 nodes) Green
rules showed `dL=0.186255`, `dZ=0.00565544`,
`dq=0.000459404`, `dP=0.000507797`.
The fixed gate requires `dL<=0.05`,
`dZ<=0.002`, `dq<=0.005`, `dP<=0.002`;
so magnetic L and port Z **fail**, despite zero
exactly coincident cross pairs and a positive
smallest cross L eigenvalue (`4.445835e-11 H`).
At softening 18/12 mm, `dL` grows to
`0.243143/0.316715`, tracking the previous
same-node diagonal fraction `0.233301/0.30113`.
Cross-rule differences also include changes in
one of the quadrature rules; they cannot be
assigned exclusively to the self-point term.

A new **experimental, structure-preserving**
Green quadrature candidate is available:
`SeawaterResponse.reintegrate_gaussian_green_matrices(
points, weights, extra_blurs=(0.,), width_factor=1.,
kernel_chunk_size=128)`.
Each positive-volume cubature node of weight `w_i`
is assigned the same one-axis variance as an
equal-volume uniform sphere:
`h_i = (3 w_i/(4 pi))^(1/3)/sqrt(5)`, scaled by
`width_factor`. The exact Coulomb interaction of
two *isotropic Gaussian current clouds* is
`G_ij = erf(d/sqrt(2*(h_i²+h_j²+a²)))/d`,
with continuous self limit
`sqrt(2/pi)/sqrt(h_i²+h_j²+a²)`.
Here `a` is an **additional Gaussian blur** length,
not the previous Plummer softening; `a=0`
is finite because the clouds have positive widths.
In Fourier space, each cloud multiplies a
positive Coulomb kernel by Gaussian smoothing
factors, preserving a **positive-semidefinite
single-rule L** without dropping diagonal
terms. This is only an approximate cubature-cell
volume moment matching, **not** an analytic
integration of the actual oriented package
corners, nor a certified continuum Coulomb
operator. Width-factor and geometric boundary
convergence remain mandatory.

**User-run Gaussian outcome at `e39c48e` (2026-11-02).**
The independent `pytest` run passed **47 tests in 31.60 s**.
With the same 461648-node streamed R/K operators and both K weak
forms, a=0 Gaussian Green A (7504 nodes) versus B (15154 nodes)
returned `dL=0.0250117`, `dZ=0.000379592`,
`dq=0.000142698`, and `dP=0.0000403977`.
These **pass** the predeclared magnetic/port/probe/power
A/B thresholds of `0.05/0.002/0.005/0.002`;
the curl-dual K differences are also below them.
Both L matrices are numerically PSD, with minimum eigenvalues
`3.4080103e-10 H` (A) and `2.9597503e-10 H` (B).
However, Gaussian self-pair shares of `0.149648` (A) and
`0.123415` (B) and the single prescribed width model leave
a real near-field systematic-error question. This is a
**discrete A/B gate pass**, **not** continuum Green validation or
permission to promote this candidate to production.

The updated `--basis-green-gaussian` experiment holds fixed the
63-mode Galerkin basis, 461648-node R/K, two K weak forms, outer
domain and zero *additional* Gaussian blur. It compares two
independent magnetic rules at width factor 1.0, and also tests
rule A at width factors `(0.75, 1.0, 1.25)`. The new tuple-valued
`width_factor` parameter to
`SeawaterResponse.reintegrate_gaussian_green_matrices`
reuses **one** expensive pair-distance calculation per block across
the width ladder, instead of repeating the whole O(N²) distance
work. Single-valued calls remain compatible. Matrices and self
contributions are returned width-major, then blur-major.

`gaussian_B_vs_A`, `width_0.75_vs_A1`, and
`width_1.25_vs_A1` each report `dL/dZ/dq/dP`,
the curl-dual K counterpart, and a fixed-threshold PASS/FAIL.
`gaussian_screen` summarizes A/B, width sensitivity, and PSD.
Passing width *perturbations* is only a robustness screen:
it does not establish the true width→0/N→∞ continuum limit.
The production `SeaMQSConfig`/Plummer Green operator is
**unchanged**. Hold promotion until additional modal,
outer-domain and electrothermal closure checks; don't switch
the neural-network teacher based only on these screens.

Run on the research branch:

```powershell
git pull
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
python run.py --mode bfzi-convergence --basis-green-gaussian
```

If the width gate fails, correct the near-field/cell averaging
itself and repeat the *same* gate; do not relax limits or
blindly multiply quadrature orders. No GitHub Actions are used.

**Failed width gate and analytic uniform-ball alternative (2026-11-02).**
On revision `35f05f9`, the user verified **47 tests passed
in 31.68 s**. At fixed independent rule A, moving the Gaussian
width factor from 1.0 to 0.75 gave
`dL=0.0733328`, `dZ=0.00198059`; moving to 1.25
gave `dL=0.0518377`, `dZ=0.0013855`. Both **fail**
the fixed `dL<=0.05` gate even though the A/B rule
comparison and the PSD check pass. Promotion remains **HOLD**.

The experimental `--basis-green-ball` run now assembles an
**exact analytic uniform-solid-ball pair Coulomb interaction**
for each equal-volume node cloud. In contrast to infinite-tail
Gaussian clouds, two disjoint balls interact as exactly `1/d`,
while fully contained/overlapping balls and self-interactions use
closed-form finite values (equal-radius self `6/(5r)`).
The model is unsoftened and has **no tunable blur/width factor**.
The Fourier/Coulomb energy of the resulting real vector-current
clouds is positive semidefinite. It reports the B-versus-A
rule difference and, at matched A nodes, the difference
between ball and original Gaussian cloud shapes, using the
same **5%/0.2%/0.5%/0.2%** thresholds for diagnostics.
The run holds fixed the 63 modes, fine R/K, dual K forms
and outer seawater domain.

This is a *core-kernel cross-check*, **not** a claim that a Gauss
weight is the volume of a true isotropic material cell. In fact,
`box_shell_quadrature` has positive partition-of-unity
weights that vary with integration rules. If the new A/B or
ball-versus-Gaussian shape gate fails, the next physics step
is an explicitly geometry-aware near-pair integration of
the actual smooth partitions (or a validated continuous
Green formulation), **not** arbitrary width fitting,
increased quadrature degree or PSD projection.
The existing `SeaMQSConfig` production Green is unchanged.
No GitHub Actions are used.

Next local commands:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
python run.py --mode bfzi-convergence --basis-green-ball
```

**Uniform-ball Green: experimental full MQS reference selection (2026-11-02).**
The user's `e5d26ff` run passed **48 tests in 34.08 s**.
Its 63-mode, 461648-node held-R/K magnetic screen yielded:
`ball_B_vs_A: dL=0.0236467, dZ=0.000321814,
dq=0.00014229, dP=0.0000351903`;
`ball_vs_gaussian_A: dL=0.0102897,
dZ=0.00028554, dq=0.0000257011,
dP=0.000023242`. Both pass fixed
`5%/0.2%/0.5%/0.2%` screens including dual K, and
both uniform-ball L matrices are PSD.
This closes the **candidate finite-volume magnetic kernel
engineering screen**, not the continuum/outer-domain
accuracy proof. The earlier Gaussian *width* sensitivity
failure remains documented: comparing two different cloud
shape prescriptions at factor 1.0 does not make arbitrary
Gaussian width choices well determined.

`SeaMQSConfig(inductance_kernel="uniform_ball")`
now selects the analytic ball Coulomb energy inside the full
two-port MQS solve (retaining positive magnetic quadrature,
common R-whitened Galerkin basis, original R/K/heat nodes,
and the complete modal solve). The default remains
`inductance_kernel="plummer"` so existing workflows are
unchanged. Under `uniform_ball`, `field_softening`
is accepted for configuration compatibility but **does not
enter L**; it is neither tuned nor silently interpreted as
a ball radius. The ball radius derives from positive magnetic
quadrature weight. This is an experimental volume-cloud
model of the meshless cubature, not exact integration over
the true curved/partition-weighted cell.

Three new end-to-end verification entry points retain the same
kernel selection through the physical calculations:
`--basis-ball` uses the **461648-node, operator-only fine R/K**
rule from the previously accepted near-wall refinement, combined
with the 7504-node uniform-ball magnetic A rule. The 95-mode
space adds 32 strictly nested face-edge curl patches to the
63 original physical modes, while level 4 adds another 64
corner/interior-current patches on the broad enclosure faces.
The full **159-mode parent** now supplies independent
holdouts of 63→95→159 (15/27/59/63/95/159 nested spaces).
All projected re-solves use the *same* fine R/K and PSD L,
reporting Z, probe heat, power and a parent-space
R-inverse Galerkin residual. Current-potential and curl-dual
K formulations are compared on the 159-mode parent.
The modal screen evaluates four original bulk probes **plus twelve
oriented near-wall probes** (both broad faces of each package,
center plus tangential offsets), printing global `dq` and
`max_probe_dq`.

An additional **residual-driven selective-enrichment** pass
starts from the physical 63-mode subset. It R-orthogonalizes
missing candidate directions with a Schur-complement diagonal
to avoid selecting near-duplicate Gaussian patches, ranks
their dual-port Galerkin residual, and adds up to eight at
a time. The sparse re-solved Z, near-wall/bulk heat and
power are checked against the assembled 159-mode parent:
`dZ<=0.002, dq<=0.005, dP<=0.002,
max_probe_dq<=0.05`. These are *offline candidate-space*
metrics, not an independent continuum accuracy certificate
or neural FAST training. The decisive next physics holdout
is **95→159 local Joule heat**, not the automatically zero
residual of a full 95-dimensional parent.

`--basis-domain-ball` builds fresh basis/port/magnetic operators
on five independent outer radii (0.52, 0.70, 0.85, 1.05 and 1.30 m).
It independently **streams fine R/K on every radius** using the
same 0.06 m radial and 0.0167 m near-wall tangential panel caps;
magnetic L remains uniform-ball, and both K weak forms are
checked at each radius. It compares successive port Z, probe
Joule heat and modal sea power without storing the large
volume Joule tensor. This is more expensive than the previous
coarse-R/K domain screen, intentionally avoiding a false
radius plateau from under-integrated near-wall currents.

`--mode bfzi --green-ball` runs complete two-port port Z,
spatial Joule/wire losses and illustrative current/voltage-driven
thermal channels with the selected ball L. The thermal ROM still
models conductor temperature feedback **only**, not the fully
temperature-dependent seawater properties. None of these
experimental modes auto-enable neural FAST. Modal and domain
convergence require inspection of their actual new numerical
logs, not a declared pass based only on Green A/B agreement.

Run sequentially, stopping for failures:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
python run.py --mode bfzi-convergence --basis-ball
python run.py --mode bfzi-convergence --basis-domain-ball
python run.py --mode bfzi --green-ball
```

The additional full-path regression checks unchanged R/K on
switching the kernel, independent ball L assembly, finite PSD
spectrum, equal port/modal/spatial Joule power, and both thermal
drive envelopes. **This new revision has not yet been executed
on the user's machine.** No GitHub Actions are used.
Do not start BFZI neural training on absolute reference labels
until the new modal/domain reports are checked; neural input
geometry coverage and held-out geometries will be a separate
validation gate.

**New user-run modal/domain/thermal results (2026-11-02).**
The user's `8f74a9b` run passed **49 tests in 35.72 s**.
The previous 63-mode parent versus 59-mode subset yielded
`dZ=0.0000670036`, `dq=0.0198242`,
`dP=0.0000691795`; the **1.98242% probe-heat
difference fails** the existing 0.5% spatial gate
even though the impedance and modal power appear converged.
The 59-mode parent residual was `0.00821778`
against the full 63-mode space, and does not bound
errors outside that space. The user's next run on
`2ac541b` passed **50 tests in 36.54 s** and established:
`guided_port_curls_to_edge_face_patches` (63→95):
`dZ=0.0106665`, `dq=0.327308`,
`max_probe_dq=0.577651`, `dP=0.0129773`;
the 63-mode R-inverse residual against the 95-mode
parent was `0.100378`. **All spatial and port
convergence gates still fail**. The 95-mode parent
also changes near-wall heat by 30–58% at some sampled
locations; this is physical-basis truncation and
cannot be cured by more Green kernel polishing alone.
The newly added 159-mode candidate/greedy holdout has
not yet been executed.

The 63-mode outer-radius sequence with independently fine
R/K showed the **1.05 → 1.30 m** step
`dZ=0.000362475`, `dq=0.0000510951`,
`dP=0.00000829131`. These meet the fixed
5%/0.2%/0.5%/0.2% operator/port/probe/power
screen's available port/probe/power limits, although
a further radius holdout would be needed for a strict
infinite-domain certificate. The prior **0.85 → 1.05 m**
step (`dZ=0.00802547`, `dP=0.00374837`)
was not yet stable.

The full demo `--mode bfzi --green-ball` preserved
port / volume / conductor power identities and ran
current- and voltage-controlled thermal envelopes.
Its **27 retained modes** and illustrative thermal
material parameters are *not* the 63/95-mode fine
reference, nor a calibrated seawater thermal transient.

The selected Green kernel name now propagates through
Galerkin restriction and modal compression. When
`inductance_kernel="uniform_ball"`, querying
`magnetic_field` uses the analytic field of
finite uniform solid current clouds (exterior
`1/d^3`; interior `1/r^3`) rather than silently
reverting to the Plummer field. This corrects the
model mismatch in the former field-decoding path,
but does not remove the cell-shape approximation.

The next run is a **single 95-mode modal holdout**, not
another Green integration/width sweep:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
python run.py --mode bfzi-convergence --basis-ball
```

After 63→95 spatial and residual convergence is measured,
the next implementation milestone is labeled BFZI
geometry/pose sampling, independent geometry holdouts,
train/save/load/infer and passive structured port/Joule
outputs; do not use the legacy 27-mode demo as an
accuracy teacher. No GitHub Actions are used.

**User's 159-mode / adaptive results and new response-snapshot path (2026-11-02).**
The user's `68d2777` run passed **50 tests in 36.65 s**.
With fine 461648-node R/K and independent uniform-ball A magnetic L,
the successive 63→95→159 trial spaces are **not** converged.
For the **95→159** step, `dZ=0.239214`, `dq=0.50494`,
`max_probe_dq=2.46521` and `dP=0.279814`.
The 63→159 R-inverse residual is `0.450642` and
95→159 residual `0.441551`. The existing adaptive residual-greedy
selector required the entire 159-mode candidate space:
`adaptive_screen: FULL_PARENT_ONLY`. This rules out using
63, 95 or 159 spatial Joule labels as certified data.
No arbitrary extra face patch level will be interpreted as
a justified continuum error bound.

**Crucial distinction: finite-space reference accuracy vs
response rank.** At a fixed geometry, excitation frequency and
two-port linear MQS state, two complex unit-port current snapshots
span at most **four real physical directions**.
The new `BFZISnapshotROM.from_operators` constructs an
R-orthonormal real Galerkin subspace from those snapshots,
projects the *same* R/L/K onto rank 1–4, re-solves the reduced
passive complex system, and evaluates J(x), Joule q(x),
symmetric port Z and sea dissipative power. Rank 4 must reconstruct
both unit-port responses of its **discrete parent** to numerical
accuracy. This is an algebraic response-manifold observation;
it says **nothing** about parent-space convergence or
generalization to unseen poses/geometry/frequency.

Unlike `SeawaterResponse`, the snapshot artifact requires
**no stored large heat integration tensor**. You can build one
directly from streamed fine R/K and the independent Green L:
`BFZISnapshotROM.from_operators(sources, basis,
parent.transform, R_fine, L_ball, K_fine, rank=4)`.
`.save(path)` writes a non-pickled numerical `.npz`
containing projection, R/L/K and dual-port states;
`BFZISnapshotROM.load(path, sources, basis)` validates
a deterministic physical-scene/source/basis signature
before allowing arbitrary seawater-location J and Joule
heat queries. No need to repeat the expensive matrix
assembly at load time. Changing geometry requires a new
label/ROM or a **future trained geometry-conditioned
predictor**: the file does not claim to generalize.

The lightweight `--mode bfzi-snapshot` demo executes
the full save→load→query path on a **provisional 27-mode
teacher** so the plumbing can be verified without
the huge 159-mode solve. The `--basis-ball` scientific
screen also reports `snapshot_rank_2` vs
`snapshot_rank_4` against the exact *same* 159-mode parent,
measuring port, heat, max-probe and power differences.
It does not automatically save a research label.

**Important separate risk:** the published Gaussian/ball
magnetic A/B checks were performed on **63 modes only**.
Their success cannot be assumed to hold for 159 narrower
current patches. For a targeted follow-up,
`--basis-ball-check-l` repeats the 159-mode fine R/K
test and independently assembles the 15154-node B rule
uniform-ball L, reports B/A differences of the complete
159-mode port/near-wall heat responses for both K forms,
without modifying the Green implementation.
This option is substantially more expensive. If it fails,
the 159-mode L *discretization* must be improved before
inferring that spatial-basis truncation alone explains
all changes. If it passes, pursue non-Gaussian,
source-aligned spatial current representations rather than
blindly multiplying Gaussian patches.

Suggested **lightweight first** commands:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
python run.py --mode bfzi-snapshot --out bfzi_response.npz
```

Only after those succeed, run the targeted heavier check
when practical:

```powershell
python run.py --mode bfzi-convergence --basis-ball-check-l
```

No GitHub Actions are used. A trained cross-geometry neural
FAST model has **not yet been built**; the compact response
artifact establishes a physically interpretable target and
save/load/query baseline, not a neural prediction.

**Provisional geometry-conditional response label contract (2026-11-02).**
The user's `e907ad2` run passed **51 tests in 37.07 s**,
then successfully saved/reloaded the 27→4 real-snapshot ROM,
with relative `dZ=1.2573013e-16`, four-probe Joule
`dq=1.4191841e-15` and `0 W` modal/port
power closure. This establishes a reliable
**fixed-scene compression and numeric artifact**; the
underlying 27-mode physical response is still provisional.

`sdfmpneo_bfzi/sea_dataset.py` now defines a stable
geometry-to-dual-port-response **sample contract**, not a
trained neural network. `geometry_features(sources)`
extracts both complete BFZI geometry parameter sets,
their arbitrary proper 3×3 rigid orientations,
enclosure extents, finite wire properties and local
return waypoints, seawater conductivity, frequency and
outer-domain data. Variable-length return paths are
padded with an explicit point count up to a caller-set
capacity (default eight); larger paths are rejected
rather than silently clipped.

`BFZIResponseLabel.from_snapshot(rom)` stores two
complex physical **raw current-mode response vectors**
`raw_unit_modes = raw_projection @ unit_modes`, plus
the complete reciprocal seawater port Z and scene/source
signature. These targets are invariant to the
indeterminate signs/rotations of the four-dimensional
SVD compression basis. For use in a common neural output
space, all geometries must still employ the **same
deterministic physical basis construction and ordering**.

`save_response_labels(path, labels)` and
`load_response_labels(path)` provide a non-pickled
fixed-width array file suitable for later supervised
training; mixed feature/mode counts, identical geometry
records and repeated signatures are rejected.
All dataset files are explicitly marked
`reference_status="provisional"`, and this stage
**does not train on them automatically**. The label
schema does not yet encode a validated
generalization domain, geometry-stratified test split,
or certification of electromagnetic/thermal fidelity.

Optional end-to-end dataset smoke example:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py -q
python run.py --mode bfzi-snapshot --out bfzi_response.npz --dataset-out bfzi_provisional_labels.npz
```

The example builds **two independently solved receiver
poses** at the same reference order and emits both
provisional records. This tests feature diversity and
portable labels; two examples are intentionally **not**
represented as sufficient training coverage. Before
valid quantitative training labels are generated,
run the independent 159-mode magnetic A/B refinement:

```powershell
python run.py --mode bfzi-convergence --basis-ball-check-l
```

No GitHub Actions are used; there is no BFZI neural
FAST model in this branch yet.

**First experimental BFZI neural training/inference shell (2026-11-02).**
The user successfully ran the prior multi-geometry data smoke on
`9e29ca9`: **51/51 tests in 35.96 s** and two independently
solved `provisional` geometry records were saved. The rank-4
snapshot still reproduces only its 27-mode parent
(`dZ=1.2573013e-16`, probe `dq=1.4191841e-15`,
zero numerical port/modal power mismatch).

The new explicit-opt-in `bfzi-generate` samples multiple
independent BFZI configurations (both 6-DOF poses,
both oriented enclosure sizes, coil shape parameters,
and seawater conductivity) with
an unchanged deterministic **27-mode** physical curl
topology and the uniform-ball reference. It saves each
sample as a separately solved provisional response with
raw physical mode coefficients and port Z. The default
`--samples 12` is a small functional dataset, **not**
coverage for arbitrary pose, size, material or marine
conditions. Each sample still requires an offline MQS
solve, and can be slow.

The new `bfzi-train` PyTorch MLP predicts the complex
physical raw-mode unit-port response and the passive
reciprocal 2×2 seawater reaction impedance from the
geometry feature vector. Training/held-out validation are
split at the **whole geometry level**, with feature and
target normalization fit ONLY on training geometries.
The model learns the *real Cholesky factor* of the
nonnegative port-loss matrix and the three independent
imaginary entries, making predicted Z reciprocal and
its real part PSD by construction. The checkpoint records
reference status, training/validation counts, normalization
and held-out signatures; loading uses numeric tensors
and `torch.load(..., weights_only=True)`.

`bfzi-infer` loads this model and queries the
deterministic scene's geometry-dependent analytic curl
basis and port source; **no sea R/L/K Green assembly**
is performed online. It produces passive reciprocal
seawater impedance, complete air+seawater+finite-conductor
port Z, the two conductor loss channels, total port power,
arbitrary-location complex J and positive Joule heat,
plus an extrapolation diagnostic based on observed
training-feature ranges. However, `J` and `Z`
are currently **distinct supervised outputs**:
integrated spatial Joule power is NOT guaranteed equal
to port dissipation, unlike the exact Galerkin ROM.
Do not call this experimental model a fully
structure-preserving electrothermal surrogate.
Its tiny provisional data are inadequate for physical
accuracy or out-of-distribution generalization.

To use the neural option, install optional Torch
(`pip install -e ".[neural]"`) and run:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py tests/test_bfzi_neural_v1.py -q
python run.py --mode bfzi-generate --samples 12 --seed 23 --out bfzi_geometry_labels.npz
python run.py --mode bfzi-train --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt --epochs 400
python run.py --mode bfzi-infer --model bfzi_neural.pt
```

If Torch is not installed, the optional neural test will
skip and the existing physics tests still work; training
requires installing that optional dependency. A 12-sample
experiment can demonstrate plumbing and geometry-held-out
provisional agreement, but its errors do **not** measure
real-world electromagnetic accuracy. **Do not publish or
deploy** these results as reliable thermal/safety
predictions. Next: independent 159-mode magnetic A/B
screen, non-Gaussian spatial reference basis, more
geometries/physics, and a response/port-loss consistency
constraint. None of these are solved by the present MLP.
No GitHub Actions are used.

**First trained-BFZI smoke and inference Joule consistency (2026-11-02).**
The user's `137733c` execution passed **53 tests in 42.22 s**,
generated eight independent `provisional` 27-mode geometries,
trained on six and held out two. Best epoch 14 of 400 gave
normalized validation MSE `1.345816`, held-out
raw current-mode relative error `0.213656` (21.4%),
and held-out seawater port Z error `0.090424` (9.0%).
This is successful SOFTWARE training, **not** learned
electromagnetic accuracy. More epochs are not a solution to
a 6-geometry / dozens-of-features problem. The report
`Outside observed varying-feature range: True`
means the inference scene is outside at least one
coordinate's **six-training-scene range**; with so few
independent geometries this is expected and a warning
against interpreting predictions quantitatively.

**Inference-only local-R closure** now defaults to on.
`BFZINeuralSurrogate.predict(sources,basis,
power_consistent=True)` evaluates a small, positive,
near-wall-aware `box_shell_quadrature` with default
orders `(4,6,8,2,3)` and streams the geometry-dependent
local resistance R in bounded chunks. It does **NOT**
assemble the nonlocal magnetic inductance matrix or
redo a full MQS solve. The learned complex raw-mode
unit responses U produce the Hermitian Gram
`U.conj().T @ R @ U`. Because general independently
learned U may violate the reciprocal two-port identity
`Im(Gram[0,1])=0`, the second port field receives the
minimal R-norm correction (first port fixed),
`u2 -= 1j * Im(Gram[0,1])/Gram[0,0] * u1`.
The resulting **real symmetric PSD Gram** replaces
the learned seawater impedance's real part, while its
learned reciprocal imaginary part is retained.
This ensures, for arbitrary *complex* port current
phasors, exact **discrete**
`port_seawater_power(currents) ==
closure_weights @ seawater_loss(closure_points,currents)`
up to roundoff, on the **specified positive closure
rule**. The field correction magnitude and imaginary
Gram defect are reported. This changes the learned
response, and neither quadrature resolution nor
magnetic reactance/true Joule distribution is certified.
It is not an exact solution of the Maxwell MQS
equations.

The uncorrected fastest path remains available:
`predict(...,power_consistent=False)` or
`bfzi-infer --raw-port-loss`, which retains independent
learned field and port power without any closure claim.
The new `bfzi-infer` output prints spatial cubature
power, matched port loss, correction size and
out-of-training-range flag. To exercise the **existing
trained checkpoint**, no new sample generation or
retraining is needed:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py tests/test_bfzi_neural_v1.py -q
python run.py --mode bfzi-infer --model bfzi_neural.pt
python run.py --mode bfzi-infer --model bfzi_neural.pt --raw-port-loss
```

The earlier 9.0% held-out Z metric pertains to the
original **uncorrected** MLP target; the loss-closure
postprocessing changes its real part. It must be
re-evaluated against *independently validated*
geometry holdouts. The essential independent
159-mode magnetic A/B check remains
`python run.py --mode bfzi-convergence --basis-ball-check-l`.
No GitHub Actions are used.

**Geometry-held-out neural spatial accuracy evaluator (2026-11-02).**
The user's `0423006` run passed **53 tests in 40.39 s**.
The local-R closure predicted spatial heat power
`0.00010943817170283617 W` versus port sea power
`0.00010943817153353738 W`: relative closure
`1.5469812e-9`, with a **0.1202%** relative
correction to the raw two-port current response.
The reciprocal Gram imaginary defect was
`1.02357e-9`, and `outside_training_range=True`.
The uncorrected MLP instead predicted seawater port
power `0.00011542299 W`, about **5.2% higher**.
This shows successful *discrete consistency*, NOT
an improvement in physical predictive accuracy.

A new `scene_from_geometry_features` reconstructs
the full two-coil scene (including both rigid poses,
enclosures, conductor material parameters, and
explicit return routes) and frequency from the
saved label vector. This permits independent
**geometry-held-out error assessment** without
rebuilding the expensive sea Galerkin R/L/K.
`--mode bfzi-eval` uses the checkpoint's recorded
`validation_signatures` and **only those actual
unseen sample geometries**. For each sample it
compares the saved provisional physical
27-mode current and port Z against both the
uncorrected model and the local-R-power-closed model.
Reported quantities include complex port Z error,
spatial complex J error, near-wall Joule q error,
port sea-power error, extrapolation warnings,
current correction, and achieved discrete closure.
Both per-geometry values and mean/max results
are emitted. Crucially, the script **does not
replace held-out geometric samples with the
fixed demonstration scene**, and it never declares
a lower closure error a lower physical-model error.

Use existing `bfzi_neural.pt` and
`bfzi_geometry_labels.npz`; neither retraining
nor additional reference solves are necessary:

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_geometry_v1.py tests/test_bfzi_mqs_v1.py tests/test_bfzi_neural_v1.py -q
python run.py --mode bfzi-eval --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt
```

The current feature-label v1 format does **not**
persist source line integration order or the
general physical modal topology. Evaluation
currently reconstructs the original **27-mode**
training basis with default `--source-order 4`
(the eight-sample dataset used order 4).
If data were generated using
`bfzi-generate --source-order N`, supply
`bfzi-eval --source-order N` to match the
physical source signature. Higher modal
topologies require a future explicit basis
descriptor. All reported errors remain relative
to the explicitly **provisional** labels, not a
certified ocean-continuum solution.

**Outer radial physical resolution.** The spherical portion of
`box_shell_quadrature` additionally supports
`max_radial_panel_width` (meters): all physical radial intervals
longer than the cap are split into positive Gauss panels, without
altering the box shells. The `--basis-domain` sequence enables a
0.18 m cap for both R/K and Green L at radii 0.52, 0.70, 0.85, 1.05
and 1.30 m. This prevents the outermost Gauss interval from becoming
arbitrarily broad at fixed order as the sphere grows. Enabling this
cap changes the discretization, and its convergence must be checked
independently: it does not by itself validate the infinite-seawater
limit. Larger-radius Green L integration becomes costlier.

The optional `--basis-verify` solves the same fine 63-mode
`R/K` system again, this time recomputing `L` on that full finer
cubature rather than reusing the coarse Green rule. It reports resulting
`dZ`, `dq` and `dP` and can be **substantially** slower. Small holdout discrepancies
alone do **not** certify `L`, basis completeness, or outer-domain convergence.
These are trial-space/reference diagnostics, not neural surrogate training;
the 63-mode response is not yet certified.

The two complete paths use `BFZIAssembly.wire_radius` and the existing
`sdfmpneo_vnext.scene.ConductorMaterial` temperature law to compute
the **finite circular-wire internal impedance** (DC + cylindrical AC skin
effect, Bessel-ratio solution); this includes the spiral and explicit return.
The structured response has the mutually exclusive PSD channels
`conductor:0`, `conductor:1` and `seawater`. The conductor helper methods
`conductor_cross_section_loss`, `conductor_heat_nodes`, and
`conductor_volume_heat` return the cylindrical skin-current radial loss,
conservative line-source heat nodes, and positive meshless volumetric Joule
nodes respectively, for **both the BFZI spiral and its return**.
The existing vNext lumped thermal `StableThermalModel` can be stepped
using `BFZICurrentControlledEnvelope` or `BFZIVoltageControlledEnvelope`
(with a passive external series/tuning impedance); conductivity, coil power
and, for voltage control, port currents update at every fixed-point iterate.
The seawater electromagnetic response stays at its reference conductivity.
For a thermal ROM with a third input column, the envelopes also inject the
seawater Joule power into a separate heat channel. A thermal model may then
have more than two states, provided its first two *decoded temperatures*
refer to the two copper conductors; additional states can represent a reduced
seawater/insulation temperature. `seawater_heat_nodes()` gives positive
spatial volume nodes whose total exactly matches the seawater port channel
under the shared Galerkin cubature. The default two-channel coil thermal
model is unchanged. A third **lumped** seawater node is illustrative, not
a calibrated infinite-ocean thermal solver.

Limitations: this is **not yet** a converged infinite-ocean reference.
It does not resolve current redistribution from proximity, arbitrary
noncircular conductor sections, heating within seawater or packages,
temperature-dependent seawater MQS, resonant feedback, or a trained BFZI
neural FAST model. The convergence script reports differences, not proof
of convergence or certified accuracy.

---

## Install

Python 3.10+ is required.

```bash
pip install -e ".[dev]"
```

For neural FAST artifacts:

```bash
pip install -e ".[dev,neural]"
```

For the one-click PyQt training console:

```bash
pip install -e ".[dev,neural,gui]"
```

The NumPy/SciPy REFERENCE and CERTIFIED paths do not require PyTorch.

## One-click cached / resumable training

The root `run.py` is the main vNext training configuration and launch entry.
Edit `CONFIG` in that file, then run:

```bash
python run.py
```

The default mode opens a nonblocking PyQt6/pyqtgraph console. Training runs in
an independent `QProcess` and metrics are tailed by a `QThread`, so correctness
teacher solves and CUDA training do not block the UI. The window provides
start/resume, pause, resume and safe stop controls together with port/spatial
loss curves and teacher-cache progress.

Headless training uses the same configuration:

```bash
python run.py --mode train
```

Teacher truth is stored in a content-addressed cache. Keeping the sampler,
teacher and truth parameter space unchanged reuses existing deterministic
sample shards; changing sample count only generates missing shards. Network,
optimizer, device and GUI changes do not invalidate teacher data. Port and
spatial model checkpoints store optimizer/model/RNG/batch position state for
safe continuation after a stop.

The legacy GUI path remains in the Git history; its configuration, cache identity and
resume semantics.

## Scene model

The main objects are:

- `SuperellipseSpiral` + `ConductorMaterial` + `CoilObject`;
- optional `SuperquadricPackageGeometry` + material + `PackageObject`;
- arbitrary independent `RigidPose` values for coils and packages;
- a homogeneous background material;
- optional strictly nested or mutually disjoint material regions.

Isotropic electric materials may use `HomogeneousMedium`, `IsotropicMaterial`,
`DebyeMaterial`, `MultiDebyeMaterial`, `TabulatedMaterial`, or another passive
frequency-response implementation. Tensor-electric regions use
`TensorElectricMaterial`, with an SPD relative-permittivity tensor and a PSD
conductivity tensor whose principal axes are expressed in the containing
region frame. Package tensors therefore rotate with package pose, while a
background tensor is expressed in world axes.

The tensor-electric backend is a piecewise-homogeneous electric transmission
model. It is not a general continuously varying tensor VIE solver.

## REFERENCE / FAST / CERTIFIED

`MeshfreeVNextSystem` is the common runtime:

```python
from sdfmpneo_vnext import MeshfreeVNextSystem

system = MeshfreeVNextSystem(
    port_artifact,
    spatial_artifact=spatial_artifact,
)

ports_fast = system.fast_ports(scene, frequency_hz)
loss_fast = system.fast_spatial(scene, frequency_hz)

ports_ref = system.reference_ports(scene, frequency_hz)
loss_ref = system.reference_spatial(scene, frequency_hz)

certified = system.certified_ports(scene, frequency_hz)
```

REFERENCE dispatches conductor-only scenes to the mixed finite-cross-section
backend and heterogeneous/tensor-electric scenes to the coupled electric /
magnetic interface backend. Homogeneous lossy-background power is represented
by an unbounded exterior-domain integral rather than a finite world box.

FAST artifacts fail closed when geometry or material values fall outside the
training domain. Tensor-electric scenes additionally require artifacts marked
`supports_tensor_electric`; scalar material features are never used as an
anisotropy proxy.

## Isotropic hybrid FAST training

The existing hybrid dataset/training path remains available for isotropic
package/background material domains:

```bash
python examples/vnext_generate_hybrid_dataset.py data/hybrid --count 128

python examples/vnext_train_hybrid_residual.py \
  data/hybrid \
  artifacts/hybrid-port.pt \
  --epochs 200

python examples/vnext_train_hybrid_spatial.py \
  data/hybrid \
  artifacts/hybrid-port.pt \
  artifacts/hybrid-spatial.pt \
  --epochs 120
```

The port decoder is reciprocal/passive by construction and returns PSD
conductor plus aggregate electric-environment dissipation channels whose sum
closes to the dissipative part of the predicted impedance. The spatial model
then resolves those channels into continuous conductor, package, and optional
unbounded-background loss fields.

## Tensor-electric FAST port and spatial training

Tensor-electric training uses tensor-invariant material features for both the
port model and the spatial latent state. The spatial decoder reuses the mature
PSD conductor/package/background shape networks but is normalized against the
tensor-aware port channels; it therefore does not collapse anisotropic
permittivity/conductivity to scalar surrogates.

The current tensor FAST family is deliberately the **hybrid package-aware**
family: its teacher/trainer requires at least one package region. A tensor
homogeneous background may be sampled and learned together with those package
regions. Background-only tensor scenes are supported by REFERENCE/CERTIFIED,
but are not silently accepted by a tensor hybrid FAST artifact trained on a
package domain.

A minimal in-memory research workflow is:

```python
import numpy as np

from sdfmpneo_vnext import HybridSceneSamplerConfig, MQSConfig
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from sdfmpneo_vnext.tensor_training_data import TensorHybridTeacherSample
from sdfmpneo_vnext.tensor_spatial_training_data import (
    TensorHybridSpatialTeacherSample,
)
from sdfmpneo_vnext.tensor_neural import (
    train_tensor_hybrid_residual_surrogate,
)
from sdfmpneo_vnext.tensor_spatial_neural import (
    train_tensor_hybrid_spatial_loss_surrogate,
)

sampler = TensorHybridSceneSamplerConfig(
    base=HybridSceneSamplerConfig(),
    tensor_package_probability=1.0,
    tensor_background_probability=0.5,
)

teacher_config = MQSConfig()
rng = np.random.default_rng(7)
spatial_samples = []
for _ in range(16):
    scene, frequency_hz = sample_tensor_hybrid_scene(rng, sampler)
    port_truth = TensorHybridTeacherSample.generate(
        scene,
        frequency_hz,
        teacher_config=teacher_config,
    )
    spatial_samples.append(
        TensorHybridSpatialTeacherSample.generate(
            port_truth,
            teacher_config=teacher_config,
        )
    )

port, port_report = train_tensor_hybrid_residual_surrogate(
    tuple(sample.port for sample in spatial_samples),
    epochs=200,
)
spatial, spatial_report = train_tensor_hybrid_spatial_loss_surrogate(
    port,
    spatial_samples,
    epochs=120,
)
```

`TensorHybridSpatialLossArtifact.prepare(...)` produces the same prepared
continuous-field interface used by the isotropic hybrid path, including:

- conductor-local `local_dissipation_matrix/matrices` queries;
- package-local/world `package_*dissipation*` queries;
- unbounded-background dissipation queries when the background is lossy;
- exact normalization to the tensor FAST port dissipation channels;
- compatibility with `MeshfreeVNextSystem.fast_spatial` and continuous thermal
  evolution.

## Tensor artifact persistence

Tensor port and spatial artifacts can be published as one self-contained
bundle:

```python
from sdfmpneo_vnext.tensor_bundle import (
    publish_tensor_bundle,
    load_tensor_bundle,
)

publish_tensor_bundle(
    "artifacts/tensor-fast",
    port,
    spatial_artifact=spatial,
    overwrite=True,
)

loaded = load_tensor_bundle("artifacts/tensor-fast")
system = loaded.system
ports = system.fast_ports(scene, frequency_hz)
field = system.fast_spatial(scene, frequency_hz)
```

The manifest contains checksums, runtime capability metadata, and a semantic
fingerprint binding the spatial artifact to the exact tensor port weights and
normalization state.

### Tensor bundle JSON inference

```bash
sdfmpneo-vnext-tensor artifacts/tensor-fast request.json --device auto
```

`--device auto` prefers CUDA, then MPS, then CPU. Tensor bundle loading preserves
the dtype stored in the port/spatial artifacts.

## Continuous thermal fields

A FAST spatial field can be coupled directly to continuous thermal evolution:

```python
thermal = system.fast_continuous_thermal_field(
    scene,
    frequency_hz,
    HomogeneousThermalMedium(...),
)
temperature = thermal.temperature_step(position, time_s, currents)
```

The thermal field remains continuous in space and time. Package thermal
interfaces and anisotropic background/package thermal tensors remain distinct
from the electromagnetic tensor-electric model.

## Scope boundaries

The vNext MVP intentionally fails closed outside its declared physics domain.
In particular:

- tensor electric support is piecewise homogeneous, not a continuously varying tensor VIE;
- magnetic permeability is isotropic in the tensor-electric stage;
- partially intersecting packages are rejected; packages must be nested or disjoint;
- conductor branching/junction networks are not part of the spiral-coil MVP;
- tensor FAST training is AC-only; exact tensor DC uses REFERENCE/CERTIFIED;
- object-local quadrature/interface discretization is still a numerical approximation even though no global world mesh is required;
- phase change and nonlinear radiation are not part of the current thermal MVP.

## Performance notes

Historical vNext GPU batching, CPU multiprocessing,
precision policy and cached batch inference details.
