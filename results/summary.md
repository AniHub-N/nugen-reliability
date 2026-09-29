| | aligned | base |
|---|---|---|
| responses (ok / attempted) | 200 / 200 | 200 / 200 |
| accuracy, answerable (strict, 95% CI) | 9.3% [0%, 20%] | 12.0% [3%, 23%] |
| correct / partial / wrong / abstained | 14 / 37 / 86 / 13 | 18 / 41 / 87 / 4 |
| declines unanswerable (95% CI) | 28.0% [10%, 50%] | 26.0% [0%, 52%] |
| same label all 5 repeats | 72.5% | 75.0% |
| mean pairwise agreement | 85.5% | 87.8% |
| served OK (after retries) | 100.0% | 100.0% |
| served OK first try | 100.0% | 100.0% |
| latency p50 / p95 (s) | 1.5 / 3.5 | 2.8 / 7.1 |
| AUROC conf_mean → correct | 0.54 [0.34, 0.73] | – |
| AUROC conf_min → correct | 0.78 [0.47, 0.98] | – |
| AUROC conf_last → correct | 0.57 [0.41, 0.72] | – |
| ECE (conf_mean) | 0.660 | – |
| mean conf: answerable vs unanswerable | 75.3 vs 71.6 | – |

Aligned minus base accuracy: -2.7% [-13%, 7%] (question-clustered bootstrap).

In-sample threshold: conf_min ≥ 63.0 keeps 55.0% of responses, risk 78.2% vs 86.0% with no threshold. AUROC(good) 0.73 [0.52, 0.89]; informative=True.