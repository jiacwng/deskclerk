# Security

## Reporting a vulnerability

Use "Report a vulnerability" under the Security tab of this repository. Please do not open a
public issue.

## What deskclerk defends against

- A tool result that tries to steer the agent. Jev screens each result. After a flag, later calls
  wait for the employee, read-only ones too.
- A call that changes data without the employee's consent. It waits for approval unless the
  employee chose Always allow for that tool.
- A destructive call on an id the agent made up. Its id arguments must appear in the request or an
  earlier tool result.
- A server that lies about its tools. A server is untrusted until the admin marks it trusted, and
  every call to it asks first.
- A server that changes a tool's definition. The changed tool is withheld.
- One employee reaching another employee's accounts or conversations.
- Edits to stored audit lines. `verify-audit` names the first changed line.

## What it does not defend against

- An admin, or anyone who holds the keys or the database.
- A call the employee approved, or one they chose to always allow before any flag.
- An injection the screen misses. On LLMail-Inject, Jev flags 75.2% of attacks.
- A poisoned tool definition on first use. Pinning trusts the first definition it sees.
- Read-only calls on a trusted server. They run without asking.
- An id an attacker planted in a tool result. The id check accepts it.
- One request that goes past the monthly budget. The budget is checked before each request.
- A bearer token a revoked employee already holds. It works until it expires.
- TypeSafe, which receives every request and tool result when a TypeSafe key is set.
- Removal of the newest audit lines.
