# A real MCP server with OAuth for the tests: the SDK's server side on a free localhost port,
# an authorization server that approves every request, a read tool, a write tool and a tool
# that fails, plus a delete tool on request. With oauth=False it has no sign-in at all.

import secrets
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import uvicorn
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    IdentityAssertionParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.mcpserver import MCPServer
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.types import ToolAnnotations

PRESET_CLIENT_ID = "admin-registered"
PRESET_CLIENT_SECRET = "admin-secret"  # noqa: S105


class ApprovingProvider:
    def __init__(self, redirect_url: str) -> None:
        self.preset = OAuthClientInformationFull(
            client_id=PRESET_CLIENT_ID,
            client_secret=PRESET_CLIENT_SECRET,
            redirect_uris=[redirect_url],
            token_endpoint_auth_method="client_secret_post",
        )
        self.clients: dict[str, OAuthClientInformationFull] = {PRESET_CLIENT_ID: self.preset}
        self.codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}
        self.registrations = 0
        self.refreshes = 0
        self.resources: list[str] = []

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.registrations += 1
        self.clients[client_info.client_id] = client_info

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        code = secrets.token_urlsafe(24)
        if params.resource is not None:
            self.resources.append(params.resource)
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.codes.get(authorization_code)

    def _issue(self, client_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        access = f"access-{secrets.token_urlsafe(16)}"
        refresh = f"refresh-{secrets.token_urlsafe(16)}"
        self.access_tokens[access] = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(time.time()) + 3600,
            resource=resource,
        )
        self.refresh_tokens[refresh] = RefreshToken(
            token=refresh, client_id=client_id, scopes=scopes, resource=resource
        )
        return OAuthToken(
            access_token=access, expires_in=3600, refresh_token=refresh, scope=" ".join(scopes)
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        del self.codes[authorization_code.code]
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return self.refresh_tokens.get(refresh_token)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        self.refreshes += 1
        del self.refresh_tokens[refresh_token.token]
        return self._issue(client.client_id, scopes or refresh_token.scopes, refresh_token.resource)

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.access_tokens.get(token)

    async def exchange_identity_assertion(
        self, client: OAuthClientInformationFull, params: IdentityAssertionParams
    ) -> OAuthToken:
        raise TokenError(error="unsupported_grant_type")

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.access_tokens.pop(token.token, None)
        self.refresh_tokens.pop(token.token, None)


@dataclass
class RunningServer:
    url: str
    redirect_url: str
    provider: ApprovingProvider
    notes: list[tuple[str, str]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)


@contextmanager
def running_mcp_server(
    redirect_url: str, oauth: bool = True, delete_tool: bool = False
) -> Iterator[RunningServer]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    provider = ApprovingProvider(redirect_url)
    running = RunningServer(f"{base}/mcp", redirect_url, provider)

    if oauth:
        mcp = MCPServer(
            "notes",
            auth_server_provider=provider,
            auth=AuthSettings(
                issuer_url=base,
                resource_server_url=f"{base}/mcp",
                validate_token_resource=True,
                client_registration_options=ClientRegistrationOptions(enabled=True),
            ),
        )
    else:
        mcp = MCPServer("notes")

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    def read_note() -> str:
        return "the note"

    @mcp.tool()
    def write_note(text: str) -> str:
        access = get_access_token()
        running.notes.append((access.token if access else "", text))
        return f"wrote {text}"

    @mcp.tool()
    def break_note() -> str:
        raise ValueError("the note is locked")

    if delete_tool:

        @mcp.tool()
        def delete_note(note_id: str) -> str:
            running.deleted.append(note_id)
            return f"deleted {note_id}"

    server = uvicorn.Server(uvicorn.Config(mcp.streamable_http_app(), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield running
    finally:
        server.should_exit = True
        thread.join()
        sock.close()
