# Attacks mapped to the OWASP Top 10 for LLM Applications (2025)

Each database attack runs as a district user through query_data and reports whether it was blocked. BIPIA rows report the attack success rate on poisoned content.

| Attack | Source | OWASP category | Result |
|---|---|---|---|
| Multi-statement read-only bypass | [Datadog Security Labs, PostgreSQL MCP server SQL injection](https://securitylabs.datadoghq.com/articles/mcp-vulnerability-case-study-SQL-injection-in-the-postgresql-mcp-server/) | LLM05:2025 Improper Output Handling | blocked |
| Identity rewrite through set_config | [Datadog Security Labs, PostgreSQL MCP server SQL injection](https://securitylabs.datadoghq.com/articles/mcp-vulnerability-case-study-SQL-injection-in-the-postgresql-mcp-server/) | LLM06:2025 Excessive Agency | blocked |
| Identity rewrite through set_identity | [Datadog Security Labs, PostgreSQL MCP server SQL injection](https://securitylabs.datadoghq.com/articles/mcp-vulnerability-case-study-SQL-injection-in-the-postgresql-mcp-server/) | LLM06:2025 Excessive Agency | blocked |
| Reading other sessions' query text | [PostgreSQL pg_stat_activity](https://www.postgresql.org/docs/current/monitoring-stats.html) | LLM02:2025 Sensitive Information Disclosure | blocked |
