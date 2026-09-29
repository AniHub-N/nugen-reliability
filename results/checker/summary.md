| condition / model | accuracy (answerable) | no filter: shown, error | checker (supported): shown, error, correct kept | Nugen score at same coverage: error | unanswerable blocked by checker |
|---|---|---|---|---|---|
| closed_book/aligned | 9% | 173, 92% | 62, 77%, 14/14 | 92% (≥79) | 96% |
| closed_book/base | 12% | 183, 90% | 30, 53%, 14/18 | no score | 96% |
| retrieval/aligned | 67% | 179, 44% | 135, 33%, 91/100 | 29% (≥80) | 92% |
| retrieval/base | 84% | 178, 29% | 139, 10%, 125/126 | no score | 100% |

App policy, aligned model with retrieval, leave-one-question-out (conf_mean):

| policy | shown | error | correct kept | unanswerable shown | thresholds |
|---|---|---|---|---|---|
| confidence only | 85/200 | 8% | 78/100 | 5/50 | 87–88 |
| confidence and checker | 84/200 | 7% | 78/100 | 0/50 | 86–88 |
