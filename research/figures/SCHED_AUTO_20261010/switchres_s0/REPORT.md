# Scheduled-Settle paired evaluation

Comparison uses the same scenes and `switch_settle_015` rule.

## high_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 71.4 | 69.0 | 86.5 | 160.6 | 0.00 | — |
| FlowResidual | 42 | 69.0 | 66.7 | 85.9 | 173.0 | 0.03 | -2.4 |

## low_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 90.5 | 88.1 | 95.5 | 131.9 | 9.76 | — |
| FlowResidual | 42 | 92.9 | 88.1 | 97.1 | 131.5 | 2.52 | +0.0 |

## moderate_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 50.0 | 50.0 | 78.1 | 196.3 | 4.74 | — |
| FlowResidual | 42 | 50.0 | 50.0 | 77.5 | 197.2 | 4.81 | +0.0 |

## ood_flow_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 40.5 | 35.7 | 65.6 | 226.9 | 0.64 | — |
| FlowResidual | 42 | 38.1 | 33.3 | 63.2 | 233.2 | 0.83 | -2.4 |

## strong_flow

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 14.3 | 11.9 | 42.3 | 274.5 | 7.47 | — |
| FlowResidual | 42 | 14.3 | 11.9 | 42.2 | 274.4 | 16.10 | +0.0 |

## variable_response

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 42.9 | 31.0 | 65.4 | 225.3 | 4.54 | — |
| FlowResidual | 42 | 42.9 | 28.6 | 64.0 | 225.8 | 4.39 | -2.4 |

## Overall

{
  "n": 252,
  "learner_strict": 0.4642857142857143,
  "rule_strict": 0.47619047619047616,
  "delta_strict_pp": -1.1904761904761862
}
