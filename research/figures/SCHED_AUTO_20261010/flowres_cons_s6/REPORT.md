# Scheduled-Settle paired evaluation

Comparison uses the same scenes and `switch_settle_015` rule.

## high_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 71.4 | 69.0 | 86.5 | 160.6 | 0.00 | — |
| FlowResidual | 42 | 73.8 | 71.4 | 87.6 | 159.6 | 4.42 | +2.4 |

## low_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 90.5 | 88.1 | 95.5 | 131.9 | 9.76 | — |
| FlowResidual | 42 | 92.9 | 88.1 | 97.1 | 129.4 | 2.52 | +0.0 |

## moderate_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 50.0 | 50.0 | 78.1 | 196.3 | 4.74 | — |
| FlowResidual | 42 | 50.0 | 50.0 | 78.8 | 196.6 | 4.64 | +0.0 |

## ood_flow_delay

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 40.5 | 35.7 | 65.6 | 226.9 | 0.64 | — |
| FlowResidual | 42 | 40.5 | 35.7 | 64.1 | 222.3 | 1.45 | +0.0 |

## strong_flow

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 14.3 | 11.9 | 42.3 | 274.5 | 7.47 | — |
| FlowResidual | 42 | 14.3 | 11.9 | 42.4 | 274.5 | 7.48 | +0.0 |

## variable_response

| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwitchSettle | 42 | 42.9 | 31.0 | 65.4 | 225.3 | 4.54 | — |
| FlowResidual | 42 | 42.9 | 31.0 | 64.8 | 224.6 | 4.59 | +0.0 |

## Overall

{
  "n": 252,
  "learner_strict": 0.4801587301587302,
  "rule_strict": 0.47619047619047616,
  "delta_strict_pp": 0.39682539682540097
}
