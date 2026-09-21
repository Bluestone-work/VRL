# Results summary

Frozen code: `8c159ab153e2f651a5cbdfa7c22706c86dc203d8`.

结果由 scripts/research_report.py 从不可替换的原始记录计算；三训练种子，标准差ddof=1。
主比较是相同历史validation协议，额外development validation不是封存test，也不代表未见拓扑。

| 数据集/运行 | Success mean ± SD | Clot removal mean ± SD | Episode length mean ± SD | Wall contact/robot-step mean ± SD |
|---|---:|---:|---:|---:|
| archived_validation | 72.8571 ± 1.8898% | 92.7773 ± 0.6408% | 155.7952 ± 5.7866 | 34.9418 ± 1.8215% |
| reproduced_validation | 72.8571 ± 1.8898% | 92.7773 ± 0.6408% | 155.7952 ± 5.7866 | 34.9418 ± 1.8215% |
| development_validation | 74.6429 ± 0.3571% | 94.9299 ± 0.3601% | 152.3857 ± 2.4573 | 34.7250 ± 0.8269% |


EXP_0001工程复现gate：PASS。详见[BASELINE_REPORT](BASELINE_REPORT.md)。

世界模型、planning、unseen-topology测试均未开展，H1–H5仍为HYPOTHESIS — NOT VERIFIED。

## EXP_0002

| Device | Success | Mass removal | Episode length | Wall / robot-step | Collision / pair-step |
|---|---:|---:|---:|---:|---:|
| cpu | 74.6429 ± 0.3571% | 94.9299 ± 0.3601% | 152.3857 ± 2.4573 | 34.7250 ± 0.8269% | 8.1387 ± 0.1522% |
| cuda:0 | 75.5952 ± 0.7435% | 94.8884 ± 0.2695% | 150.5321 ± 2.7320 | 34.3926 ± 0.7891% | 7.9576 ± 0.1663% |

设备配对success差+0.9524pp，翻转10/840。同设备重复与CPU父接口检查通过；没有算法变化。详见[EVALUATION_PROTOCOL_REPORT](EVALUATION_PROTOCOL_REPORT.md)。
