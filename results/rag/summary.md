| | aligned | base |
|---|---|---|
| responses (ok / attempted) | 200 / 200 | 200 / 200 |
| accuracy, answerable (strict, 95% CI) | 66.7% [50%, 82%] | 84.0% [71%, 97%] |
| correct / partial / wrong / abstained | 100 / 39 / 11 / 0 | 126 / 15 / 5 / 4 |
| declines unanswerable (95% CI) | 42.0% [20%, 66%] | 36.0% [14%, 60%] |
| same label all 5 repeats | 82.5% | 87.5% |
| mean pairwise agreement | 91.0% | 93.5% |
| served OK (after retries) | 100.0% | 100.0% |
| served OK first try | 100.0% | 100.0% |
| latency p50 / p95 (s) | 2.0 / 4.9 | 3.2 / 6.2 |
| AUROC conf_mean → correct | 0.93 [0.85, 0.99] | – |
| AUROC conf_min → correct | 0.76 [0.60, 0.90] | – |
| AUROC conf_last → correct | 0.86 [0.73, 0.96] | – |
| ECE (conf_mean) | 0.253 | – |
| mean conf: answerable vs unanswerable | 86.9 vs 75.3 | – |

Aligned minus base accuracy: -17.3% [-35%, -1%] (question-clustered bootstrap).

In-sample threshold: conf_mean ≥ 86.4 keeps 50.5% of responses, risk 16.8% vs 39.5% with no threshold. AUROC(good) 0.81 [0.70, 0.92]; informative=True.