> **BFZI research branch (2026-11-01).** The active implementation is `sdfmpneo_bfzi/`: exact BFZI continuous-coil geometry, enclosed twin-port scenes, explicit return paths, direct air Neumann integrals, finite-domain seawater MQS Galerkin fields, shared Joule dissipation, and deterministic reduced subspaces. This is an **unconverged finite-domain prototype**, not yet a validated ocean/FEM surrogate. The vNext commands further below remain **legacy superellipse workflows**, not BFZI electrothermal training. See [new code instructions](docs/research/bfzi_uwpt_electrothermal/CODE_ENTRY.md) and [theory](theory/main.tex).

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
python run.py --mode bfzi-train --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt --epochs 400 # torch optional
python run.py --mode bfzi-infer --model bfzi_neural.pt # no electromagnetic reference solve
python run.py --mode bfzi-convergence --basis-ball-check-l # optional expensive 159-mode magnetic A/B
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
independent BFZI configurations (receiver 6-DOF pose,
both coil shape parameters, seawater conductivity) with
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
is performed online. It produces seawater impedance,
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
