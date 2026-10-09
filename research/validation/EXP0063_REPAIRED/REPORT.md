# EXP0063 repaired Transformer reference

This development screening uses the repaired training path (`prior_truncation_hold_fixed_local_history`),
fixed response parameters, nominal inlet flow 0.05 mm/s, and the development scene offset 20.
The stateful Adaptive Settle, Fixed Settle, and No Settle results are recorded in
`FLOW_DELAY_AUDIT/standard_metrics_v3`; no registered test scenes are used.

The repaired Transformer completed 120 updates with training seed 6304. The first 125 completed
paired episodes are retained in `episodes.jsonl`; one long horizon episode was still running when
this report was prepared. Screening aggregates over completed episodes are:

| delay | completed | Strict | task success | removal | AUC | T90_300 (s) | wall (s) | lost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 12/42 | 12/42 | .717 | .597 | 239.3 | 5.74 | 1 |
| 2 | 42 | 21/42 | 21/42 | .823 | .674 | 198.1 | .73 | 0 |
| 3 | 41 | 27/41 | 28/41 | .885 | .711 | 167.7 | .18 | 0 |

These are exploratory, single-seed results and do not establish stable superiority. The delay-1
wall-contact cost and the incomplete final episode are preserved as limitations. Training and
evaluation seeds are recorded separately in `config.json` and `manifest.json`.

Reproduce training:

```bash
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.train_lysis_nav \
  --out research/runs/EXP0063_TRANSFORMER_REPAIRED_s6304 --updates 120 --workers 4 \
  --device cuda:0 --abcd A --seed 6304 --speed-prior --latency-max 3
```

Reproduce paired screening:

```bash
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.evaluate_repaired_transformer \
  --out research/validation/EXP0063_REPAIRED \
  --checkpoint research/runs/EXP0063_TRANSFORMER_REPAIRED_s6304/policy.pt \
  --workers 12
```
