# deskclerk

deskclerk is an internal company agent. An admin lists the company's MCP servers. Employees
sign in with the company login, connect their own accounts on those servers and ask the agent
to work there. Before a call that changes something runs, deskclerk checks it against the
tool's schema, the ids the agent has seen and the employee's request. Each employee has a
budget and a request quota, and every request goes to an audit log.

## Setup

You need Docker with Compose and an Anthropic API key. The steps below run deskclerk and a
development Keycloak on this machine. The app uses the host's network, which Docker Engine on
Linux and WSL supports.

1. Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`, `DESKCLERK_TOKEN_KEY` and
   `DESKCLERK_SESSION_KEY`. The file shows how to make each key. Without `TYPESAFE_API_KEY`,
   writes get the schema and id checks only.
2. List the MCP servers in `servers.toml`:

   ```toml
   [[server]]
   name = "calendar"
   url = "https://mcp.example.com/calendar"
   ```

   A server without sign-in takes `auth = "none"`. A server that needs a registered OAuth
   client takes `client_id` and `client_secret_env`, the name of the variable in `.env` that
   holds the secret. deskclerk trusts the read-only marks of the servers you list.
3. Start both services:

   ```
   docker compose up -d --wait
   ```

4. Open http://localhost:8000 and sign in as `analyst` with the password `deskclerk`.

The employees' MCP tokens are stored in the `deskclerk-data` volume, encrypted with
`DESKCLERK_TOKEN_KEY`. deskclerk refuses to start with a different key, so keep it.

For your company login, set `DESKCLERK_ISSUER`, `DESKCLERK_CLIENT_ID` and
`DESKCLERK_SIGN_IN_REDIRECT_URL` in `.env` and start the app alone with
`docker compose up -d --no-deps deskclerk`. Every setting is listed in
`src/deskclerk/config.py`.

## License

Apache 2.0. See `LICENSE`.
