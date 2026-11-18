# Security

## Reporting a vulnerability

Use "Report a vulnerability" under the Security tab of this repository. Please do not open a
public issue.

## What deskclerk defends against

- A tool result that tries to steer the agent. Jev screens each result, and after a flag, later
  calls ask the employee first.
- A server that changes a tool's definition. The changed tool is withheld.
- One employee reaching another employee's accounts or conversations.
- Edits to stored audit lines. `verify-audit` names the first changed line.

## What it does not defend against

- An admin, or anyone who holds the keys or the database.
- A call the employee approved.
- TypeSafe, which receives every request and tool result when a TypeSafe key is set.
- Removal of the newest audit lines.
- Reads of `/metrics`, which needs no sign-in.
