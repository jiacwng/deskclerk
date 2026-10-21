# Changelog

## Unreleased

- Connects each employee's own accounts on a list of MCP servers through OAuth with PKCE, and runs one agent loop per request with that employee's tools.
- Checks each write call against the tool's schema and the ids the agent has seen before it runs.
- Limits each employee to a spending budget and a request quota, and writes one JSON audit line per request.
- Signs employees in through any OpenID Connect provider, with a development Keycloak in `compose.yaml`.
- Runs from one Docker image.
- Measures the agent on WorkBench: 443 of 459 held-out tasks correct, against 447 for WorkBench's own loop.
- Measures prompt injection defenses on AgentDojo: attack success 2.0% with none, 1.9% with spotlighting, 0.0% with a DeBERTa screen.
- Scores TypeSafe's Jev on 1,742 write calls: ROC AUC 0.47 on test.
- Screens every tool result with Jev when a TypeSafe key is set, and withholds a flagged result or one Jev cannot judge. On AgentDojo, 0 of 949 attacks succeed and 84.5% of tasks complete with no attack.
- Counts Jev's tokens in each employee's budget.
- Breaking: removes Jev from the write check, where it scored a ROC AUC of 0.47 on test. Removes `DESKCLERK_JEV_BUDGET_USD`.
