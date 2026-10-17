# BIRD execution accuracy

Model anthropic:claude-haiku-5-5. 498 BIRD Mini-Dev questions, scored as the BIRD repository scores them: the predicted result set equals the gold result set, order-insensitive. Both read as the analyst, who sees every row.

| Approach | Correct | Accuracy | SQL error | Wrong result | Cost (USD) |
|---|---|---|---|---|---|
| agent | 282 | 0.566 | 0 | 216 | 0.7545 |
| baseline | 296 | 0.594 | 4 | 198 | 0.2139 |

Only the agent got 29 questions right and only the baseline 43. The exact McNemar test on these paired outcomes gives p = 0.125, so the difference is not significant at the 5% level. The agent costs 3.5 times as much as the baseline. The agent is there to answer under each employee's permissions; BIRD checks that its SQL stays as accurate as one prompt.

Agent tokens: 5,722,324 input and 364,527 output. Baseline tokens: 1,445,825 input and 138,713 output.
