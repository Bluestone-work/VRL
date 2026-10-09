# Benchmark v3 (microscope-visible obstacles): dev scenes, 14 anatomies x 30 seeds

Learned variants pool all seeds (number of seeds in brackets). Safe Success: all clots cleared, wall < 1 robot-s, no obstacle collision, no lost cluster, no cluster contact, spacing compliant.

| N | method | episodes | Safe % | Raw % | removal % | obstacle-hit eps % | static hits/ep | moving hits/ep | wall>=1 s % | lost % | T100 s | AUC % | path mm |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 路线追踪 + APF [1] | 41 | 39.0 | 58.5 | 75.6 | 7.3 | 0.00 | 0.10 | 46.3 | 0.0 | 182 | 28.4 | 186 |
| 2 | 路线追踪 + APF [1] | 41 | 43.9 | 73.2 | 89.0 | 7.3 | 0.02 | 0.15 | 43.9 | 0.0 | 116 | 26.6 | 276 |
| 3 | 路线追踪 + APF [1] | 40 | 37.5 | 77.5 | 92.1 | 17.5 | 0.10 | 0.33 | 45.0 | 0.0 | 86 | 23.3 | 332 |

## Paired differences in Safe Success vs route pursuit + APF (pp, 95 % bootstrap CI over scenes; learned = seed mean)

| method | N=1 | N=2 | N=3 | all |
|---|---|---|---|---|
| 路线追踪（无避障） | — | — | — | — |
| Transformer PPO（无规则先验） | — | — | — | — |
| MLP 残差 PPO | — | — | — | — |
| GRU 残差 PPO | — | — | — | — |
| T-IRPPO 无域随机化 | — | — | — | — |
| T-IRPPO（本文） | — | — | — | — |
