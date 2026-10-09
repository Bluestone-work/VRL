# EXP0062 — history based flow adaptation

This is the first short implementation gate for the v6 flow benchmark. The
policy input remains the deployable route token and the previously executed
action. No flow, response gain, simulator position, anatomy or environment id
is passed to the policy. The eight-step window is 0.8 s at the current 0.1 s
control interval.

The existing `PriorSpeed` settle rule was found to limit a near-target command:
the old mapping was `clip(prior + 0.5*a0, 0, 1)`, so a zero-speed settle prior
could request at most 0.5. `command_legacy` preserves that mapping for the
mechanism ablation. The default mapping uses full residual authority and still
clips to the legal [0,1] speed range. All principal comparisons must use the
new mapping and report the mapping delta separately.

The auxiliary head predicts the next observed velocity change (token columns
19:22), normalized by the configured scale. Its default weight is 0.05 and
the module reports weighted and unweighted losses. This target is observable at
deployment and does not use hidden position or flow truth.

Short gate commands:

```bash
cd /home/wj/桌面/vascular_marl_local.tar.
/home/wj/miniconda3/envs/v/bin/python -m pytest tests/test_vascular_adaptive_research.py -q
/home/wj/miniconda3/envs/v/bin/python scripts/train_lysis_nav.py --out research/runs/EXP0062_smoke --minutes 0.05 --workers 1 --device cpu --adaptive-history --speed-prior --seed 6201
```

The four groups, full 14-anatomy/N=1/2/3 evaluation and three-seed budget are
not claimed complete until their raw JSONL records are present. Existing v6
results remain untouched.
