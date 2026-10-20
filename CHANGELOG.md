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
