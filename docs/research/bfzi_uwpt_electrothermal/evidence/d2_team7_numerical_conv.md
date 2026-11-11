# D2 TEAM-7 numerical convergence (no experimental-data fitting)

All comparisons use 101 fixed physical points at the fourth-period end phase, original elemental B and Jy. Percentages use the second case as denominator. The signed Bz line-mean difference is normalized by the reference Bz line RMS to avoid cancellation or near-zero mean.

| Frequency | Independent sweep | Pair | A1 Bz L2 | A1 Bxyz L2 | A1 mean Δ / RMS | A1 RMS Δ | A2 Bz L2 | Top Jy L2 | Bottom Jy L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 50 Hz | h | h15 → h10 | 11.64293% | 7.98117% | 0.29955% | 0.18870% | 19.41675% | 2.49029% | 3.90561% |
| 50 Hz | h | h10 → h075 | 10.89298% | 7.45231% | 3.52169% | 3.52607% | 16.47299% | 1.61344% | 1.59238% |
| 50 Hz | R | r15 → h10 | 10.57370% | 7.14589% | 2.78090% | 2.25241% | 9.95738% | 2.80915% | 3.43224% |
| 50 Hz | R | h10 → r30 | 12.31519% | 7.69029% | 0.79549% | 1.78263% | 9.48462% | 2.09613% | 3.37253% |
| 50 Hz | air | a40 → h10 | 8.29843% | 6.44568% | 1.41510% | 3.03393% | 9.73021% | 0.61755% | 0.73554% |
| 50 Hz | air | h10 → a25 | 7.62061% | 5.13046% | 1.60257% | 1.22648% | 15.49170% | 1.17955% | 0.74564% |
| 50 Hz | probe | probe10 → probe120 | 9.56097% | 7.08382% | 1.43545% | 2.60321% | 11.51224% | 6.78158% | 0.73830% |
| 50 Hz | probe | probe120 → probe240 | 8.58829% | 7.40807% | 2.03961% | 1.80598% | 10.92639% | 4.56011% | 1.10482% |
| 200 Hz | h | h15 → h10 | 13.53495% | 8.29421% | 0.06715% | 0.69138% | 23.00215% | 4.20926% | 7.83425% |
| 200 Hz | h | h10 → h075 | 12.70586% | 7.68156% | 4.14476% | 3.96955% | 19.89567% | 3.31848% | 6.15397% |
| 200 Hz | R | r15 → h10 | 12.23323% | 7.37559% | 2.88316% | 2.17636% | 12.08940% | 2.50661% | 4.17763% |
| 200 Hz | R | h10 → r30 | 14.28411% | 8.00435% | 1.05100% | 1.88866% | 11.38989% | 1.65294% | 5.16336% |
| 200 Hz | probe | probe120 → probe240 | 10.12530% | 7.72974% | 2.30133% | 1.88812% | 12.95207% | 8.64426% | 2.15291% |

**Decision: OPEN.** These are numerical intermesh differences, not experimental errors or certified errors to a continuum solution. Keep full-field H(curl), outer-domain, source and time consistency gates active. Small line-integrated differences alone do not certify convergence of the spatial B/J field.
