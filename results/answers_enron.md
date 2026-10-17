# Answer table on Enron

Model anthropic:claude-haiku-5-5. 200 EnronQA questions, each asked under row-level security by the custodian whose mailbox holds the email it was written from. 200 of the 200 rows read matched a loaded email its custodian reads.

| Correct, judged by Jev (of 200) | Email reached the model (of 200) | Correct when it did (of 125) | Refused | Out of steps | Screen withheld a passage | Cost (USD) |
|---|---|---|---|---|---|---|
| 124 (55.1% to 68.4%) | 125 (55.6% to 68.9%) | 111 (82.1% to 93.2%) | 12 | 5 | 1 | 0.2566 |

Counts come with their 95% Wilson score interval.
Correct: TypeSafe's Jev decision model, shown the email, the question, the reference answer and the answer, was asked "Does the answer match the gold answer?" and gave yes a probability of at least 0.5. The paper uses Llama-3.1-70B-Instruct as this judge; how often Jev agrees with a person on this task is not measured. A refusal or an answer out of steps counts as wrong and is not judged.
Email reached the model: a search result or an opened document shown to the model carries the email's id. Screen withheld a passage: answers where the injection screen kept at least one search result, opened email or query row from the model.

Questions and reference answers are the first of each row of EnronQA (MichaelR207/enron_qa_0922, CC BY 4.0). We wrote only the judge question.
