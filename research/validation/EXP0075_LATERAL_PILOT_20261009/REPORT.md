# Lateral residual pilot (training anatomies only)

Same seed-7101 checkpoint; inference action-mapping intervention, not retraining or a new learned module. Original Ours uses privileged flow supervision during its earlier training.
No held-out anatomy is used for candidate selection. All original metrics are unchanged.

| Condition | Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | Wall s | Lost total | Spacing pair-s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_delay | Ours original | 27 | 66.7 | 66.7 | 85.9 | 0.661 | 176.2 | 0.047 | 1 | 0.000 |
| high_delay | Candidate | 27 | 77.8 | 74.1 | 96.3 | 0.713 | 170.8 | 4.254 | 1 | 0.000 |
| low_delay | Ours original | 27 | 81.5 | 74.1 | 93.6 | 0.752 | 141.7 | 3.590 | 1 | 0.000 |
| low_delay | Candidate | 27 | 81.5 | 74.1 | 92.7 | 0.752 | 137.8 | 2.068 | 1 | 0.000 |
| strong_flow | Ours original | 27 | 25.9 | 22.2 | 52.2 | 0.410 | 254.8 | 6.877 | 1 | 0.052 |
| strong_flow | Candidate | 27 | 25.9 | 22.2 | 52.3 | 0.413 | 254.3 | 6.911 | 1 | 0.130 |
| variable_response | Ours original | 27 | 48.1 | 29.6 | 62.6 | 0.452 | 229.0 | 7.463 | 2 | 0.047 |
| variable_response | Candidate | 27 | 48.1 | 29.6 | 63.5 | 0.456 | 231.2 | 7.475 | 2 | 0.066 |
