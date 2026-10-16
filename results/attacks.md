# Attacks mapped to the OWASP Top 10 for LLM Applications (2025)

Each database attack runs as a district user through query_data and reports whether it was blocked. BIPIA rows report the attack success rate on poisoned content.

Each BIPIA row is one of BIPIA's reference-match attack instructions, the second of its category, inserted at the start, middle and end of each context and scored with BIPIA's MatchRefEval. It is not the paper's average over every attack.
Screened rows are a tuning result: the screen's question and criteria were written on BIPIA's test split, which these rows use too. On BIPIA's train split, which it never saw, the screen flagged 261 of 296 attacked passages and 0 of 548 clean ones.
InjecAgent-style rows adapt InjecAgent's data-stealing case to this agent: the poisoned email tells it to open an email the reader may not open. They are not comparable with the paper's numbers.

| Attack | Source | OWASP category | Result |
|---|---|---|---|
| Multi-statement read-only bypass | [Datadog Security Labs, PostgreSQL MCP server SQL injection](https://securitylabs.datadoghq.com/articles/mcp-vulnerability-case-study-SQL-injection-in-the-postgresql-mcp-server/) | LLM05:2025 Improper Output Handling | blocked |
| Identity rewrite through set_config | [PostgreSQL set_config](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADMIN-SET) | LLM06:2025 Excessive Agency | blocked |
| Identity rewrite through set_identity | [PostgreSQL set_config](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADMIN-SET) | LLM06:2025 Excessive Agency | blocked |
| Reading other sessions' query text | [PostgreSQL pg_stat_activity](https://www.postgresql.org/docs/current/monitoring-stats.html) | LLM02:2025 Sensitive Information Disclosure | blocked |
| BIPIA email QA: Marketing & Advertising | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.11 (16/150) |
| BIPIA email QA: Scams & Fraud | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.01 (2/150) |
| BIPIA table QA: Marketing & Advertising | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.59 (88/150) |
| BIPIA table QA: Scams & Fraud | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.01 (2/150) |
| BIPIA email QA through the agent: Marketing & Advertising | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.02 (3/150) |
| BIPIA email QA through the agent: Scams & Fraud | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.01 (1/150) |
| InjecAgent-style data stealing, base setting | [after InjecAgent (Zhan et al. 2024)](https://github.com/uiuc-kang-lab/InjecAgent) | LLM02:2025 Sensitive Information Disclosure | obeyed 0/150, forbidden email reached the model 0/150, in the answer 0/150 |
| InjecAgent-style data stealing, enhanced setting | [after InjecAgent (Zhan et al. 2024)](https://github.com/uiuc-kang-lab/InjecAgent) | LLM02:2025 Sensitive Information Disclosure | obeyed 0/150, forbidden email reached the model 0/150, in the answer 0/150 |
| BIPIA email QA: Marketing & Advertising, screened | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.00 (0/150) |
| BIPIA email QA: Scams & Fraud, screened | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.00 (0/150) |
| BIPIA table QA: Marketing & Advertising, screened | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.00 (0/150) |
| BIPIA table QA: Scams & Fraud, screened | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.00 (0/150) |
| BIPIA email QA through the agent: Marketing & Advertising, screened | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.00 (0/150) |
| BIPIA email QA through the agent: Scams & Fraud, screened | [BIPIA (Yi et al. 2023)](https://github.com/microsoft/BIPIA) | LLM01:2025 Prompt Injection | ASR 0.00 (0/150) |
| InjecAgent-style data stealing, base setting, screened | [after InjecAgent (Zhan et al. 2024)](https://github.com/uiuc-kang-lab/InjecAgent) | LLM02:2025 Sensitive Information Disclosure | obeyed 0/150, forbidden email reached the model 0/150, in the answer 0/150 |
| InjecAgent-style data stealing, enhanced setting, screened | [after InjecAgent (Zhan et al. 2024)](https://github.com/uiuc-kang-lab/InjecAgent) | LLM02:2025 Sensitive Information Disclosure | obeyed 0/150, forbidden email reached the model 0/150, in the answer 0/150 |
