# An MCP server over Streamable HTTP with three tools: search_documents, describe_data and
# query_data. Every call runs as the employee named in the request's bearer token, so an MCP
# client reads only the documents and rows that employee may read, and what it reads goes
# through the injection screen first, as the agent's tools do. The server publishes its OAuth
# protected resource metadata (RFC 9728), so a client finds Keycloak, registers itself there and
# signs the employee in; it accepts only tokens whose audience is its own URL (RFC 8707).
# Run locally: PYTHONPATH=src .venv/bin/python -m needtoknow.mcp_server

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import anyio.to_thread
import psycopg
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import AnyHttpUrl, Field

from needtoknow import data, db, retrieval
from needtoknow.auth import AuthError, Identity, SigningKeyLookup, authenticate, jwks_signing_key
from needtoknow.config import Settings, load_settings
from needtoknow.jev import JevError
from needtoknow.quota import Quota
from needtoknow.screen import SCREEN_DOWN, Screen, screen_from_settings, shown, shown_rows

# The Keycloak client scope that puts this server's URL in the token audience.
SCOPE = "needtoknow-mcp"


class EmployeeToken(AccessToken):
    identity: Identity


class EmployeeTokenVerifier:
    def __init__(self, settings: Settings, signing_key: SigningKeyLookup) -> None:
        self.settings = settings
        self.signing_key = signing_key

    # The SDK answers 401 when this returns None. The signing key lookup can block on Keycloak,
    # so it runs in a worker thread.
    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            identity = await anyio.to_thread.run_sync(
                authenticate, token, self.settings, self.signing_key, self.settings.mcp_url
            )
        except AuthError:
            return None
        return EmployeeToken(
            token=token,
            client_id=identity.client_id,
            scopes=sorted(identity.scopes),
            subject=identity.user_id,
            claims={"iss": self.settings.issuer},
            identity=identity,
        )


Input = Annotated[str, Field(min_length=1, max_length=retrieval.MAX_INPUT_CHARS)]


@contextmanager
def _reachable() -> Iterator[None]:
    try:
        yield
    except ValueError as error:
        raise ToolError(str(error)) from error
    except psycopg.OperationalError as error:
        raise ToolError("the database cannot be reached, try again later") from error


def create_server(
    settings: Settings, signing_key: SigningKeyLookup, screen: Screen | None
) -> MCPServer:
    pool = db.app_pool(settings, settings.pool_size)
    quota = Quota(settings.requests_per_minute)

    def employee() -> EmployeeToken:
        token = get_access_token()
        if not isinstance(token, EmployeeToken):
            raise ToolError("no signed-in employee")
        if not quota.allows(token.identity.user_id):
            raise ToolError("too many requests, wait a minute")
        return token

    server = MCPServer(
        name="needtoknow",
        token_verifier=EmployeeTokenVerifier(settings, signing_key),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(settings.issuer),
            resource_server_url=AnyHttpUrl(settings.mcp_url),
            required_scopes=[SCOPE],
            validate_token_resource=False,
        ),
    )

    @server.tool(description="Search the company documents the signed-in employee may read.")
    def search_documents(query: Input, k: int = 5) -> list[retrieval.Hit]:
        token = employee()
        with _reachable(), pool.connection() as connection:
            hits = retrieval.search(connection, token.identity.principals, query, k)
        if screen is None:
            return hits
        try:
            flags = screen.flagged([hit.text for hit in hits])
        except JevError:
            raise ToolError(SCREEN_DOWN) from None
        return [hit for hit, flag in zip(hits, flags, strict=True) if not flag]

    @server.tool(
        description="List the schemas of the company data, or one schema's tables and columns."
    )
    def describe_data(schema: Input | None = None) -> str:
        employee()
        with _reachable(), pool.connection() as connection:
            text = data.describe_data(connection, schema)
        return shown(screen, text, "The data description")[0]

    @server.tool(
        description="Run one read-only SQL query as the signed-in employee and return the rows."
    )
    def query_data(sql: Input) -> str:
        token = employee()
        with _reachable(), pool.connection() as connection:
            result = data.query_data(connection, token.identity.principals, sql)
        if result.failed:
            raise ToolError(shown(screen, result.text, "The query error")[0])
        return shown_rows(screen, result)[0]

    return server


def main() -> None:
    settings = load_settings()
    server = create_server(settings, jwks_signing_key(settings), screen_from_settings(settings))
    server.run("streamable-http", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
