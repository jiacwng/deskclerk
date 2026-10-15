# BIRD execution accuracy

Model claude-haiku-5-5. 500 BIRD Mini-Dev questions, scored as the BIRD repository scores them: the predicted result set equals the gold result set, order-insensitive. Both read as the analyst, who sees every row.

| Approach | Correct | Accuracy | SQL error | Wrong result | Cost (USD) |
|---|---|---|---|---|---|
| agent | 283 | 0.566 | 0 | 217 | 0.7572 |
| baseline | 298 | 0.596 | 4 | 198 | 0.2145 |

Agent tokens: 5,742,246 input and 365,925 output. Baseline tokens: 1,449,675 input and 139,082 output.
