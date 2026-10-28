# Connects an employee's own account to an MCP server through OAuth, then lists and calls that
# server's tools as the employee. The flow is two requests: begin gives the authorization URL,
# complete takes the code the server sent the browser back with. A flow in progress holds a live
# task inside the MCP SDK, so the copy of the app that began it keeps it. Every copy records the
# flow in Postgres and listens for NOTIFY: the copy that receives the browser's callback writes
# the code to a table, the copy that holds the flow completes it and reports the outcome.

import asyncio
import json
import logging
import ssl
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx2
import psycopg
from mcp.client import Client
from mcp.client.auth import OAuthClientProvider, OAuthFlowError
from mcp.client.auth.utils import is_valid_client_metadata_url
from mcp.client.streamable_http import StreamableHTTPError, streamable_http_client
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata
from mcp.shared.exceptions import MCPError
from mcp.types import ContentBlock, EmbeddedResource, ResourceLink, TextContent, Tool
from psycopg import AsyncConnection

from deskclerk.servers import McpServer
from deskclerk.tokens import ServerTokenStorage, TokenStore

PUBLIC_CLIENT = "none"
HTTP_TIMEOUT = httpx2.Timeout(30, read=300)
CALLBACK_CHANNEL = "oauth_callback"
OUTCOME_CHANNEL = "oauth_outcome"
LISTEN_START_SECONDS = 10
RECONNECT_SECONDS = 1
MAX_FAILURE_CHARS = 500
STALE_CALLBACK_SECONDS = 3600

_log = logging.getLogger(__name__)


class NotConnected(Exception):
    pass


class FlowFailed(Exception):
    pass


UPSTREAM_ERRORS = (
    NotConnected,
    FlowFailed,
    MCPError,
    StreamableHTTPError,
    OAuthFlowError,
    httpx2.HTTPError,
)


def first_leaf(error: BaseException) -> BaseException:
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


class NoSignInNeeded(Exception):
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
    read_only: bool
    destructive: bool
    title: str = ""


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool


@dataclass(frozen=True)
class _PendingFlow:
    code: asyncio.Future[AuthorizationCodeResult]
    task: asyncio.Task[None]
    deadline: float


@dataclass(frozen=True)
class _RecordedFlow:
    user_id: str
    server_name: str
    deadline: float


class OAuthConnections:
    def __init__(
        self,
        tokens: TokenStore,
        redirect_url: str,
        client_name: str,
        flow_timeout_seconds: float,
        client_metadata_url: str | None = None,
    ) -> None:
        self.tokens = tokens
        self._database = tokens.database
        self._client_metadata_url = (
            client_metadata_url if is_valid_client_metadata_url(client_metadata_url) else None
        )
        self._redirect_url = redirect_url
        self._client_name = client_name
        self._flow_timeout_seconds = flow_timeout_seconds
        self._pending: dict[str, _PendingFlow] = {}
        self._outcomes: dict[str, asyncio.Future[str | None]] = {}
        self._handlers: set[asyncio.Task[None]] = set()
        self._listening = asyncio.Event()
        self._database.create_tables(
            "CREATE TABLE IF NOT EXISTS oauth_flows ("
            "state TEXT PRIMARY KEY, user_id TEXT NOT NULL, server_name TEXT NOT NULL, "
            "deadline DOUBLE PRECISION NOT NULL)",
            "CREATE TABLE IF NOT EXISTS oauth_callbacks ("
            "state TEXT PRIMARY KEY, code TEXT, iss TEXT, error TEXT, "
            "created_at DOUBLE PRECISION NOT NULL)",
        )

    def client_metadata(self) -> OAuthClientMetadata:
        return OAuthClientMetadata(
            redirect_uris=[self._redirect_url],
            client_name=self._client_name,
            grant_types=["authorization_code", "refresh_token"],
            token_endpoint_auth_method=PUBLIC_CLIENT,
            application_type="web",
        )

    def _provider(
        self, storage: ServerTokenStorage, server: McpServer, redirect_handler: Any, callback: Any
    ) -> OAuthClientProvider:
        provider = OAuthClientProvider(
            server.url,
            self.client_metadata(),
            storage,
            redirect_handler,
            callback,
            client_metadata_url=self._client_metadata_url,
        )
        provider.context.token_expiry_time = storage.token_expiry()
        return provider

    @asynccontextmanager
    async def _client(
        self, provider: OAuthClientProvider | None, server: McpServer
    ) -> AsyncIterator[Client]:
        async with _http_client(provider, server) as http:
            transport = streamable_http_client(server.url, http_client=http)
            async with Client(transport) as client:
                yield client

    @asynccontextmanager
    async def listening(self) -> AsyncIterator[None]:
        listener = asyncio.create_task(self._listen())
        try:
            await asyncio.wait_for(self._listening.wait(), LISTEN_START_SECONDS)
            yield
        finally:
            listener.cancel()
            for handler in list(self._handlers):
                handler.cancel()
            await asyncio.gather(listener, *self._handlers, return_exceptions=True)

    async def _listen(self) -> None:
        while True:
            try:
                async with await AsyncConnection.connect(
                    self._database.url, autocommit=True
                ) as listener:
                    await listener.execute("LISTEN oauth_callback")
                    await listener.execute("LISTEN oauth_outcome")
                    for state in list(self._pending):
                        self._handle(state)
                    self._listening.set()
                    async for notice in listener.notifies():
                        if notice.channel == CALLBACK_CHANNEL:
                            self._handle(notice.payload)
                        else:
                            self._report_outcome(notice.payload)
            except psycopg.Error:
                _log.exception("the OAuth listener lost its database connection")
                self._listening.clear()
                await asyncio.sleep(RECONNECT_SECONDS)

    def _handle(self, state: str) -> None:
        handler = asyncio.create_task(self._finish_flow(state))
        self._handlers.add(handler)
        handler.add_done_callback(self._handlers.discard)

    def _report_outcome(self, payload: str) -> None:
        outcome = json.loads(payload)
        waiting = self._outcomes.get(outcome["state"])
        if waiting is not None and not waiting.done():
            waiting.set_result(outcome["failure"])

    async def _finish_flow(self, state: str) -> None:
        flow = self._pending.get(state)
        if flow is None:
            return
        async with self._database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM oauth_callbacks WHERE state = %s RETURNING code, iss, error",
                (state,),
            )
            callback = await cursor.fetchone()
            if callback is None:
                return
            await connection.execute("DELETE FROM oauth_flows WHERE state = %s", (state,))
        del self._pending[state]
        if callback["error"] is not None:
            flow.task.cancel()
            return
        flow.code.set_result(
            AuthorizationCodeResult(code=callback["code"], state=state, iss=callback["iss"])
        )
        failure = None
        try:
            await _outcome(flow.task)
        except Exception as error:
            _log.warning("connect failed to complete: %s", type(error).__name__)
            failure = f"{type(error).__name__}: {error}"[:MAX_FAILURE_CHARS]
        async with self._database.async_pool.connection() as connection:
            await connection.execute(
                "SELECT pg_notify(%s, %s)",
                (OUTCOME_CHANNEL, json.dumps({"state": state, "failure": failure})),
            )

    async def begin(self, user_id: str, server: McpServer) -> str:
        if server.auth == "none":
            raise NoSignInNeeded(f"{server.name} needs no sign-in")
        loop = asyncio.get_running_loop()
        storage = self.tokens.storage_for(user_id, server, self._redirect_url)
        authorization_url: asyncio.Future[str] = loop.create_future()
        code: asyncio.Future[AuthorizationCodeResult] = loop.create_future()

        async def redirect_handler(url: str) -> None:
            state = parse_qs(urlparse(url).query)["state"][0]
            deadline = time.time() + self._flow_timeout_seconds
            self._pending[state] = _PendingFlow(code, task, deadline)
            async with self._database.async_pool.connection() as connection:
                await connection.execute(
                    "INSERT INTO oauth_flows (state, user_id, server_name, deadline) "
                    "VALUES (%s, %s, %s, %s)",
                    (state, user_id, server.name, deadline),
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

        await self._drop_expired_flows(time.time())
        task = asyncio.create_task(authenticate())
        task.add_done_callback(_consume_exception)
        waiting: set[asyncio.Future[Any]] = {authorization_url, task}
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if authorization_url.done():
            return authorization_url.result()
        await _outcome(task)
        raise AlreadyConnected(f"{user_id} is already connected to {server.name}")

    async def complete(self, user_id: str, state: str, code: str, iss: str | None = None) -> str:
        flow = await self._recorded_flow(user_id, state)
        if time.time() > flow.deadline:
            await self._cancel(state)
            raise FlowExpired("the connection was not finished in time")
        waiting: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._outcomes[state] = waiting
        try:
            await self._send_callback(state, code=code, iss=iss)
            try:
                failure = await asyncio.wait_for(waiting, max(flow.deadline - time.time(), 0))
            except TimeoutError:
                raise FlowExpired("the connection was not finished in time") from None
        finally:
            del self._outcomes[state]
        if failure is not None:
            raise FlowFailed(failure)
        return flow.server_name

    async def cancel(self, user_id: str, state: str) -> None:
        await self._recorded_flow(user_id, state)
        await self._cancel(state)

    async def cancel_for(self, user_id: str, server_name: str) -> None:
        async with self._database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM oauth_flows WHERE user_id = %s AND server_name = %s RETURNING state",
                (user_id, server_name),
            )
            states = [row["state"] for row in await cursor.fetchall()]
        for state in states:
            await self._stop(state)

    async def cancel_all_for(self, user_id: str) -> None:
        async with self._database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM oauth_flows WHERE user_id = %s RETURNING state", (user_id,)
            )
            states = [row["state"] for row in await cursor.fetchall()]
        for state in states:
            await self._stop(state)

    async def _recorded_flow(self, user_id: str, state: str) -> _RecordedFlow:
        async with self._database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT user_id, server_name, deadline FROM oauth_flows WHERE state = %s",
                (state,),
            )
            row = await cursor.fetchone()
        if row is None:
            raise UnknownState("no connection is waiting for this state")
        if row["user_id"] != user_id:
            raise WrongUser("this connection was started by another employee")
        return _RecordedFlow(row["user_id"], row["server_name"], row["deadline"])

    async def _cancel(self, state: str) -> None:
        async with self._database.async_pool.connection() as connection:
            await connection.execute("DELETE FROM oauth_flows WHERE state = %s", (state,))
        await self._stop(state)

    async def _stop(self, state: str) -> None:
        flow = self._pending.pop(state, None)
        if flow is not None:
            flow.task.cancel()
        else:
            with suppress(UnknownState):
                await self._send_callback(state, error="cancelled")

    async def _send_callback(
        self,
        state: str,
        code: str | None = None,
        iss: str | None = None,
        error: str | None = None,
    ) -> None:
        async with (
            self._database.async_pool.connection() as connection,
            connection.transaction(),
        ):
            cursor = await connection.execute(
                "INSERT INTO oauth_callbacks (state, code, iss, error, created_at) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING state",
                (state, code, iss, error, time.time()),
            )
            if await cursor.fetchone() is None:
                raise UnknownState("no connection is waiting for this state")
            await connection.execute("SELECT pg_notify(%s, %s)", (CALLBACK_CHANNEL, state))

    async def _drop_expired_flows(self, now: float) -> None:
        for state, flow in list(self._pending.items()):
            if now > flow.deadline:
                flow.task.cancel()
                del self._pending[state]
        async with self._database.async_pool.connection() as connection:
            await connection.execute("DELETE FROM oauth_flows WHERE deadline < %s", (now,))
            await connection.execute(
                "DELETE FROM oauth_callbacks WHERE created_at < %s",
                (now - STALE_CALLBACK_SECONDS,),
            )

    @asynccontextmanager
    async def _connected_client(self, user_id: str, server: McpServer) -> AsyncIterator[Client]:
        if server.auth == "none":
            async with self._client(None, server) as client:
                yield client
            return
        storage = self.tokens.storage_for(user_id, server, self._redirect_url)
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
            self.tokens.delete_tokens(user_id, server.name)
            raise not_connected.exceptions[0] from None

    async def tools(self, user_id: str, server: McpServer) -> list[ToolInfo]:
        async with self._connected_client(user_id, server) as client:
            listing = await client.list_tools()
        return [_tool_info(tool) for tool in listing.tools]

    async def call(
        self, user_id: str, server: McpServer, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        async with self._connected_client(user_id, server) as client:
            result = await client.call_tool(tool_name, arguments)
        return ToolResult(_result_text(result.content), result.is_error)


def _http_client(provider: OAuthClientProvider | None, server: McpServer) -> httpx2.AsyncClient:
    verify: ssl.SSLContext | bool = True
    if server.client_cert is not None and server.client_key is not None:
        verify = httpx2.create_ssl_context()
        verify.load_cert_chain(server.client_cert, server.client_key)
    return httpx2.AsyncClient(
        auth=provider, timeout=HTTP_TIMEOUT, headers=dict(server.headers), verify=verify
    )


def _result_text(content: list[ContentBlock]) -> str:
    lines: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            lines.append(block.text)
        elif isinstance(block, ResourceLink):
            lines.append(f"[resource {block.uri} omitted]")
        elif isinstance(block, EmbeddedResource):
            lines.append(f"[resource {block.resource.uri} omitted]")
        else:
            lines.append(f"[{block.mime_type} content omitted]")
    return "\n".join(lines)


def _tool_info(tool: Tool) -> ToolInfo:
    hints = tool.annotations
    read_only = hints is not None and hints.read_only_hint is True
    destructive = not read_only and (hints is None or hints.destructive_hint is not False)
    return ToolInfo(
        tool.name,
        tool.description or "",
        tool.input_schema,
        read_only,
        destructive,
        tool.title or "",
    )


async def _outcome(task: asyncio.Task[None]) -> None:
    try:
        await task
    except BaseExceptionGroup as group:
        upstream = group.subgroup(UPSTREAM_ERRORS)
        if upstream is None:
            raise
        raise first_leaf(upstream) from None


def _consume_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
