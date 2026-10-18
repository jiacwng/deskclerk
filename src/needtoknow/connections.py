# Connects an employee's own account to an MCP server through OAuth, then lists and calls that
# server's tools as the employee. The flow is two requests: begin gives the authorization URL,
# complete takes the code the server sent the browser back with.

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx2
from mcp.client import Client
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata
from mcp.types import TextContent

from needtoknow.servers import McpServer
from needtoknow.tokens import ServerTokenStorage, TokenStore

PUBLIC_CLIENT = "none"
HTTP_TIMEOUT = httpx2.Timeout(30, read=300)


class NotConnected(Exception):
    pass


class AlreadyConnected(Exception):
    pass


class UnknownState(Exception):
    pass


class FlowExpired(Exception):
    pass


class WrongUser(Exception):
    pass


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool


@dataclass
class _PendingFlow:
    user_id: str
    code: asyncio.Future[AuthorizationCodeResult]
    task: asyncio.Task[None]
    deadline: float


class OAuthConnections:
    def __init__(
        self,
        tokens: TokenStore,
        redirect_url: str,
        client_name: str,
        flow_timeout_seconds: float,
    ) -> None:
        self._tokens = tokens
        self._redirect_url = redirect_url
        self._client_name = client_name
        self._flow_timeout_seconds = flow_timeout_seconds
        self._pending: dict[str, _PendingFlow] = {}

    def _provider(
        self, storage: ServerTokenStorage, server: McpServer, redirect_handler: Any, callback: Any
    ) -> OAuthClientProvider:
        metadata = OAuthClientMetadata(
            redirect_uris=[self._redirect_url],
            client_name=self._client_name,
            grant_types=["authorization_code", "refresh_token"],
            token_endpoint_auth_method=PUBLIC_CLIENT,
            application_type="web",
        )
        provider = OAuthClientProvider(server.url, metadata, storage, redirect_handler, callback)
        provider.context.token_expiry_time = storage.token_expiry()
        return provider

    @asynccontextmanager
    async def _client(
        self, provider: OAuthClientProvider, server: McpServer
    ) -> AsyncIterator[Client]:
        async with httpx2.AsyncClient(auth=provider, timeout=HTTP_TIMEOUT) as http:
            transport = streamable_http_client(server.url, http_client=http)
            async with Client(transport) as client:
                yield client

    async def begin(self, user_id: str, server: McpServer) -> str:
        loop = asyncio.get_running_loop()
        storage = self._tokens.storage_for(user_id, server, self._redirect_url)
        authorization_url: asyncio.Future[str] = loop.create_future()
        code: asyncio.Future[AuthorizationCodeResult] = loop.create_future()

        async def redirect_handler(url: str) -> None:
            state = parse_qs(urlparse(url).query)["state"][0]
            self._pending[state] = _PendingFlow(
                user_id, code, task, loop.time() + self._flow_timeout_seconds
            )
            authorization_url.set_result(url)

        async def callback_handler() -> AuthorizationCodeResult:
            try:
                return await asyncio.wait_for(code, self._flow_timeout_seconds)
            except TimeoutError:
                raise FlowExpired(
                    f"the connection to {server.name} was not finished in time"
                ) from None

        async def authenticate() -> None:
            provider = self._provider(storage, server, redirect_handler, callback_handler)
            async with self._client(provider, server) as client:
                await client.list_tools()

        self._drop_expired_flows(loop.time())
        task = asyncio.create_task(authenticate())
        task.add_done_callback(_consume_exception)
        waiting: set[asyncio.Future[Any]] = {authorization_url, task}
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if authorization_url.done():
            return authorization_url.result()
        task.result()
        raise AlreadyConnected(f"{user_id} is already connected to {server.name}")

    async def complete(self, user_id: str, state: str, code: str) -> None:
        flow = self._pending.get(state)
        if flow is None:
            raise UnknownState("no connection is waiting for this state")
        if flow.user_id != user_id:
            raise WrongUser("this connection was started by another employee")
        del self._pending[state]
        if asyncio.get_running_loop().time() > flow.deadline:
            flow.task.cancel()
            raise FlowExpired("the connection was not finished in time")
        flow.code.set_result(AuthorizationCodeResult(code=code, state=state))
        await flow.task

    def _drop_expired_flows(self, now: float) -> None:
        for state, flow in list(self._pending.items()):
            if now > flow.deadline:
                flow.task.cancel()
                del self._pending[state]

    @asynccontextmanager
    async def _connected_client(self, user_id: str, server: McpServer) -> AsyncIterator[Client]:
        storage = self._tokens.storage_for(user_id, server, self._redirect_url)
        if await storage.get_tokens() is None:
            raise NotConnected(f"{user_id} has not connected {server.name}")

        async def redirect_handler(url: str) -> None:
            raise NotConnected(f"{user_id} must connect {server.name} again")

        async def callback_handler() -> AuthorizationCodeResult:
            raise NotConnected(f"{user_id} must connect {server.name} again")

        provider = self._provider(storage, server, redirect_handler, callback_handler)
        try:
            async with self._client(provider, server) as client:
                yield client
        except BaseExceptionGroup as group:
            not_connected, _ = group.split(NotConnected)
            if not_connected is None:
                raise
            raise not_connected.exceptions[0] from None

    async def tools(self, user_id: str, server: McpServer) -> list[ToolInfo]:
        async with self._connected_client(user_id, server) as client:
            listing = await client.list_tools()
        return [
            ToolInfo(tool.name, tool.description or "", tool.input_schema) for tool in listing.tools
        ]

    async def call(
        self, user_id: str, server: McpServer, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        async with self._connected_client(user_id, server) as client:
            result = await client.call_tool(tool_name, arguments)
        texts = [block.text for block in result.content if isinstance(block, TextContent)]
        return ToolResult("\n".join(texts), result.is_error)


def _consume_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
