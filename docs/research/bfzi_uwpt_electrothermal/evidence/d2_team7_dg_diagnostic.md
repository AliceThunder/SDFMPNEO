# TEAM-7 discontinuous elemental B/J export diagnostic

**Date: 2026-11-10. Status: D2 OPEN.**

## Actual Elmer solve

On the official physical TEAM-7 h15 mesh (53,510 tetrahedra), the 50 Hz first BDF2 time step was rerun with the unchanged original CSC closed-coil current and WhitneyAVSolver. ResultOutputSolver was set to:

```sif
Discontinuous Bodies = Logical False
Discontinuous Galerkin = Logical True
Save Elemental Fields = Logical True
Save Nodal Fields = Logical True
```

The Elmer 26.2 run exited **0**. The resulting `resdgall/transient_t0001.vtu` contains **214,040 distinct vertices**, exactly four for each tetrahedron; no adjacent tetrahedra share field nodes in this representation. Thus its exported B/J fields retain inter-element discontinuity, unlike the previously analyzed continuous VTU point-field reconstruction. A pilot with `Save Nodal Fields=False` wrote zero point-field variables, so that variant is not used.

## Numerical cross-check with native Elmer integration

A direct 4-node linear-tetrahedron approximation was applied to the discontinuous exported vertex fields:

```text
<|v|²>_tetra = [sum_i |v_i|² + |sum_i v_i|²] / 20
W_DG = sum_e Vol(e) <|B|²>_e / (2 mu0)
P_DG = sum_{e in Al} Vol(e) <|J|²>_e / sigma_Al
```

`GeometryIds` in this actual VTU file are 1=coil, 2=air, 3=aluminium. Volumes are computed from each tetrahedron's physical coordinates. The result is:

| Quantity | Exported DG vertex approximation | Native Elmer `SaveScalars` | Relative difference |
|---|---:|---:|---:|
| Magnetic field energy (J) | 0.3982899157 | 0.4390936345 | **9.292715%** |
| Aluminium eddy power (W) | 7.8732366083 | 9.3450600267 | **15.749748%** |

The exported B vector varies within tetrahedra: the combined deviations of the four vertex values from their tetra mean have relative L2 of **24.08%**. Accordingly, the exported values are **discontinuous elementwise postprocessed nodal representations**, not raw Whitney edge-curl Gauss-point data. Applying linear-vertex quadrature to them demonstrably does **not** reproduce Elmer's native energy and power integration, and this difference must not be mistaken for a physical discrepancy between distinct solvers. The native values are retained as the integral benchmark.

## Reproduce and decision

```powershell
python -m pytest tests/test_bfzi_d2_team7_dg.py -q
python -m examples.bfzi_d2_team7_dg --vtu D:\work\BFZI-D2-verify\runs\team7_hr\h15\resdgall\transient_t0001.vtu --energy 0.4390936344562 --power 9.345060026748 --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_dg_diagnostic.json
```

**Decision:** Use native `MagnetoDynamicsCalcFields` / `SaveScalars` for integrated magnetic energy and eddy power. DG field export can support discontinuous spatial comparisons, but it is **not accepted as a raw Whitney curl field or as a certified physical energy integral**. The original elemental Whitney degrees of freedom / integration-point curl extraction remains an independent OPEN task. D2 OPEN, old G1 FAIL, neural teacher frozen.