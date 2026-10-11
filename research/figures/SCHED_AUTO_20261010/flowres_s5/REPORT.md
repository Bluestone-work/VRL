# Scheduled-Settle paired evaluation

Comparison uses the same scenes and `switch_settle_015` rule.

## high_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 71.4 | 69.0 | 86.5 | 160.6 | 0.00 | — |
| FlowResidual | 42 | 73.8 | 71.4 | 87.6 | 156.2 | 4.50 | +2.4 |

## low_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 90.5 | 88.1 | 95.5 | 131.9 | 9.76 | — |
| FlowResidual | 42 | 92.9 | 88.1 | 97.1 | 129.4 | 2.41 | +0.0 |

## moderate_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 50.0 | 50.0 | 78.1 | 196.3 | 4.74 | — |
| FlowResidual | 42 | 50.0 | 50.0 | 78.7 | 196.6 | 4.63 | +0.0 |

## ood_flow_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 40.5 | 35.7 | 65.6 | 226.9 | 0.64 | — |
| FlowResidual | 42 | 42.9 | 38.1 | 66.4 | 223.7 | 1.65 | +2.4 |

## strong_flow

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 14.3 | 11.9 | 42.3 | 274.5 | 7.47 | — |
| FlowResidual | 42 | 14.3 | 11.9 | 42.3 | 274.6 | 7.43 | +0.0 |

## variable_response

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 42.9 | 31.0 | 65.4 | 225.3 | 4.54 | — |
| FlowResidual | 42 | 40.5 | 28.6 | 65.4 | 226.1 | 4.46 | -2.4 |

## Overall

{
  "n": 252,
  "learner_strict": 0.4801587301587302,
  "rule_strict": 0.47619047619047616,
  "delta_strict_pp": 0.39682539682540097
}
