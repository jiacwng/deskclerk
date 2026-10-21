# Changelog

## Unreleased

- Connects each employee's own accounts on a list of MCP servers through OAuth with PKCE, and runs one agent loop per request with that employee's tools.
- Checks each write call against the tool's schema, and each destructive call against the ids the agent has seen, before it runs.
- Limits each employee to a spending budget and a request quota, and writes one JSON audit line per request.
- Signs employees in through OpenID Connect. Tested with Keycloak, which `compose.yaml` runs for development.
- Runs from one Docker image.
- Measures the agent on WorkBench: 443 of 459 held-out tasks correct, against 447 for WorkBench's own loop.
- Measures prompt injection defenses on AgentDojo: attack success 2.0% with none, 1.9% with spotlighting, 0.0% with a DeBERTa screen and 0.4% with Prompt Guard 2. Under attack, 81.9% of tasks complete with none, 38.7% with DeBERTa and 67.7% with Prompt Guard 2.
- Scores TypeSafe's Jev on 1,742 write calls: ROC AUC 0.47 on test.
- Screens every tool result with Jev when a TypeSafe key is set, and withholds a flagged result or one Jev cannot judge. On AgentDojo, 0 of 949 attacks succeed, and 84.5% of tasks complete with no attack and 20.7% under attack.
- Measures three injection screens on 1,975 tool outputs from AgentDojo: Jev flags 1,174 of 1,174 outputs with the attack's marker and 0 of 801 without it, Prompt Guard 2 flags 372 and 0, DeBERTa flags 755 and 243.
- Counts Jev's tokens in each employee's budget.
- Traces each request to Langfuse when its keys are set, with its text masked unless the audit log keeps it.
- Removes Jev from the write check, where it scored a ROC AUC of 0.47 on test, and removes `DESKCLERK_JEV_BUDGET_USD`.
