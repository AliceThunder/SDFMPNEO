# D2 TEAM-7 three-dimensional numerical field convergence (2026-11-10)

**Plan:** [BFZI-PDR V1.1](../BFZI_PDR_V1_1_NUMERICAL_CONVERGENCE.md). **Physical input:** ElmerCSC TEAM7 genuine 50/200 Hz body coil, asymmetric aluminium plate, unchanged source/materials/geometry. **Experimental data:** not used.

## Method and limitations

Run `python -m examples.bfzi_d2_team7_volume --root D:\work\BFZI-D2-verify\runs\team7_hr --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_3d_regions.json`.

Read the actual `res50/transient_t0064.vtu` and `res200/transient_t0064.vtu` tetrahedral outputs from the successful four-cycle BDF2 simulations. Use **identical physical Cartesian 11×11×11 points** in each fixed interior box. Sample using VTK's physical tetrahedral interpolation, with all 1331 points valid in each compared case. Compare full `B=(Bx,By,Bz)` on common points, and `J=(Jx,Jy,Jz)` inside aluminium. Reports are `||u_i-u_j||_2/||u_j||_2` and RMS amplitude changes `|r_i-r_j|/r_j`. No smoothing, fitting to experiment, or ref-grid interpolation based on source labels.

The boxes in **metres**, all strictly away from material interfaces, are:
- `gap_air`: x=0.15..0.26, y=0.05..0.17, z=0.027..0.043 (above the plate, below coil).
- `upper_air`: x=0.10..0.25, y=0.06..0.20, z=0.17..0.23.
- `plate_bulk`: x=0.16..0.26, y=0.045..0.175, z=0.004..0.014 (outside eccentric hole).

**Interpretation constraint:** Elmer VTU `magnetic flux density e` and `current density e` appear as *point data*, representing exported/reconstructed values from elemental H(curl) fields. These physically aligned 3D samples are informative about spatial stability of this output **but are not raw edge-element curl coefficients or exact volume-quadrature energy**. Real magnetic energy and raw-element field comparison still require separate edge-element / elemental extraction. The sample-grid RMS is a fixed-volume indicator, not a physical energy certificate. All boxes have equal sample weights; the observed RMS change is not a rigorous continuum error bound.

## Actual results

| Frequency | Region | h15 → h10 B L2 | h10 → h075 B L2 | R0.15 → R0.20 B L2 | R0.20 → R0.30 B L2 |
|---|---|---:|---:|---:|---:|
| 50 Hz | air gap | 7.9479% | 7.1604% | 6.1004% | 6.8127% |
| 50 Hz | upper air | 8.2709% | 5.9261% | 7.1842% | 6.3185% |
| 50 Hz | aluminium bulk | 11.226% | 9.824% | 1.795% | 2.159% |
| 200 Hz | air gap | 8.2590% | 7.5027% | 6.4048% | 7.1734% |
| 200 Hz | upper air | 8.2946% | 5.9563% | 7.1936% | 6.3320% |
| 200 Hz | aluminium bulk | 28.474% | 23.968% | 2.919% | 3.555% |

For **J inside aluminium**, L2 pair differences were:

| Frequency | h15 → h10 | h10 → h075 | R0.15 → R0.20 | R0.20 → R0.30 |
|---|---:|---:|---:|---:|
| 50 Hz | 8.2773% | 7.5588% | 2.8711% | 2.7531% |
| 200 Hz | 19.2427% | 17.3987% | 5.1990% | 4.8051% |

Despite air gap B L2 changing by **7.9479%** on h15→h10, its full-vector RMS magnitude changes only **0.0446%** at 50 Hz. Conversely, 200 Hz bulk-conductor B remains ~24% different on h10→h075. The magnitude-only indicator is insufficient; spatial vector distribution and conductor fields remain OPEN.

The `air` (40/30/25mm) and `probe` (60/120/240 segments) scans are also included in [full machine-readable evidence](d2_team7_3d_regions.json). No experimental Bz/Jy data was accessed.

## Source representation: same-physical-step controlled test

The official TEAM7 SIF drives its 3 components through `Current Density k = Variable "time", "CoilCurrent e k"`. Elmer's official TEAM-7 tutorial provides an alternative `Use Elemental CoilCurrent=True` plus time-dependent `Current Density Multiplier` (see Elmer Tutorials, *Asymmetrical Conductor with a Hole*, section 31). On the **same h10 mesh and one actual 50 Hz BDF2 step**, both source routes produced identical full-mesh `magnetic flux density e` and identical aluminium `current density e`; solver exit code 0 in both. All fixed B/J probe outputs were identical. The coil-body exported `current density e` changed (original nonzero, new zero), while the coil-current input arrays were identical; the two routes have different reporting semantics for that field. Thus the source switch is **not** adopted as an accuracy remedy or silently applied to the canonical source. Actual one-step SIFs, logs, VTU and `f.dat` are at `runs/team7_hr/h10/case_res_src_*.sif` and `res_src_*`.

## Gate and next physical action

**TEAM-7 numerical precision OPEN; D2 OPEN; old G1 FAIL; teacher frozen.** Three-dimensional h/R regional evidence now exists independently of experimental data, and demonstrates substantial unresolved conductor spatial field error. Next isolate the **raw element-wise Whitney curl B** and real elemental energy / Joule integrals from Elmer's edge-element solution, with fixed material domains and physical volume quadrature; then test field recovery and spatial-order improvements on the same 50/200 Hz meshes before reconsidering R and independent time budgets.
