# Corrected standard metrics: classical delay baselines

Source: 504 development episodes, 14 anatomies × N=1/2/3 × delays 1/2/3,
nominal inlet flow 0.05 mm/s, fixed response parameters. Controllers are
stateful within each episode. `T90_300` equals 300 s when 90% clearance was
not reached; conditional T90 remains available in the raw summaries.

| Method | Delay | Full clear | Strict | Removal | AUC | T90_300 s | Wall s | Lost | Spacing violation s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive Settle | 1 | 38/42 | 38/42 | .976 | .786 | 119.6 | 1.20 | 0 | .00 |
| Adaptive Settle | 2 | 3/42 | 3/42 | .580 | .487 | 288.3 | 1.13 | 2 | .00 |
| Adaptive Settle | 3 | 0/42 | 0/42 | .148 | .135 | 300.0 | 19.39 | 2 | .00 |
| Fixed Settle | 1 | 16/42 | 16/42 | .738 | .616 | 221.8 | 1.20 | 1 | .00 |
| Fixed Settle | 2 | 27/42 | 27/42 | .876 | .710 | 178.2 | 0.00 | 0 | .00 |
| Fixed Settle | 3 | 30/42 | 30/42 | .908 | .736 | 155.9 | 0.00 | 0 | .00 |
| No Settle | 1 | 36/42 | 36/42 | .968 | .754 | 141.8 | 1.22 | 0 | .00 |
| No Settle | 2 | 0/42 | 0/42 | .143 | .129 | 300.0 | 16.05 | 2 | .00 |
| No Settle | 3 | 0/42 | 0/42 | .113 | .103 | 300.0 | 22.99 | 2 | .00 |
| Corrected damping | 1 | 35/42 | 35/42 | .953 | .712 | 154.0 | 6.35 | 0 | .02 |
| Corrected damping | 2 | 1/42 | 1/42 | .218 | .179 | 298.4 | 15.37 | 2 | .00 |
| Corrected damping | 3 | 0/42 | 0/42 | .134 | .118 | 300.0 | 25.58 | 2 | .00 |

Adaptive Settle is the required strong classical comparator. Its stateful
response estimate gives the best low-delay result but is fragile at delay 2/3.
The corrected damping controller is not competitive and is not a candidate for
learning comparisons.
