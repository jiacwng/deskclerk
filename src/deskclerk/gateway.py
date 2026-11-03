# The MCP gateway: deskclerk served at /mcp over streamable HTTP, so an employee's own MCP client
# lists and calls the tools of the servers the employee connected. The client signs in with the
# company login server's access token. Listing and calling go through the same offer, write check,
# approval rules, injection screen, quota, audit and metrics as the page. A call that needs the
# employee's approval asks through MCP elicitation, once per call.

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any
from weakref import WeakKeyDictionary

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.context import ServerRequestContext
from mcp.server.elicitation import AcceptedElicitation, elicit_with_validation
from mcp.server.lowlevel import Server
from mcp.server.session import ServerSession
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError, NoBackChannelError
from mcp_types import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    CallToolRequestParams,
    CallToolResult,
    ClientCapabilities,
    ElicitationCapability,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool

from deskclerk.agent import (
    ExecutedCall,
    Gate,
    McpServerFailed,
    Offer,
    OfferedTool,
    ProposedCall,
    cut_output,
    no_tool_span,
    offer_tools,
    server_failure,
)
from deskclerk.audit import Audit
from deskclerk.auth import AuthError, AuthUnavailable, Identity, SigningKeyLookup, authenticate
from deskclerk.config import Settings
from deskclerk.connections import UPSTREAM_ERRORS, NotConnected, OAuthConnections
from deskclerk.jev import Jev
from deskclerk.metrics import Metrics
from deskclerk.providers import BudgetExceeded
from deskclerk.quota import Quota
from deskclerk.screen import is_flagged, jev_screen
from deskclerk.serverstore import ServerCatalog
from deskclerk.spending import Spending, SpendingStore, month_of
from deskclerk.tokens import TokenStore

_log = logging.getLogger(__name__)


class IdentityToken(AccessToken):
    identity: Identity


class ApprovalForm(BaseModel):
    approve: bool = Field(default=False, title="Approve this call")


@dataclass(frozen=True)
class _Done:
    outcome: str
    text: str
    server: str = ""
    tool: str = ""
    is_error: bool = True
    ran: bool = False
    decision: str | None = None
    flagged: bool = False


class BearerVerifier:
    def __init__(
        self,
        settings: Settings,
        signing_key: SigningKeyLookup,
        is_deactivated: Callable[[str], Awaitable[bool]],
    ) -> None:
        self._settings = settings
        self._signing_key = signing_key
        self._is_deactivated = is_deactivated

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            identity = await run_in_threadpool(
                authenticate, token, self._settings, self._signing_key, self._settings.audience
            )
        except AuthError:
            return None
        except AuthUnavailable:
            _log.warning("the login server cannot be reached, so an MCP token was refused")
            return None
        if await self._is_deactivated(identity.user_id):
            return None
        return IdentityToken(
            token=token,
            client_id=identity.client_id,
            scopes=[],
            subject=identity.user_id,
            claims={"iss": self._settings.issuer},
            identity=identity,
        )


def _caller(ctx: ServerRequestContext[Any, Any]) -> Identity:
    user = None if ctx.request is None else ctx.request.scope.get("user")
    if isinstance(user, AuthenticatedUser) and isinstance(user.access_token, IdentityToken):
        return user.access_token.identity
    raise MCPError(INVALID_REQUEST, "send Authorization: Bearer <access token>")


def _listed(name: str, tool: OfferedTool) -> Tool:
    info = tool.info
    return Tool(
        name=name,
        title=info.title or None,
        description=info.description,
        input_schema=info.input_schema,
        annotations=ToolAnnotations(
            read_only_hint=info.read_only, destructive_hint=info.destructive
        ),
    )


class Gateway:
    def __init__(
        self,
        settings: Settings,
        catalog: ServerCatalog,
        connections: OAuthConnections,
        tokens: TokenStore,
        quota: Quota,
        audit: Audit,
        metrics: Metrics,
        gate: Gate,
        jev: Jev | None,
        spending_store: SpendingStore,
        signing_key: SigningKeyLookup,
        is_deactivated: Callable[[str], Awaitable[bool]],
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._connections = connections
        self._tokens = tokens
        self._quota = quota
        self._audit = audit
        self._metrics = metrics
        self._gate = gate
        self._jev = jev
        self._spending_store = spending_store
        self._timed = metrics.timed(no_tool_span)
        self._seen: WeakKeyDictionary[ServerSession, list[str]] = WeakKeyDictionary()
        self._server: Server[Any] = Server(
            "deskclerk", on_list_tools=self._list_tools, on_call_tool=self._call_tool
        )
        self.app: Starlette = self._server.streamable_http_app(
            auth=AuthSettings(
                issuer_url=settings.issuer,
                resource_server_url=f"{settings.public_url}/mcp",
                validate_token_resource=False,
            ),
            token_verifier=BearerVerifier(settings, signing_key, is_deactivated),
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )

    def running(self) -> AbstractAsyncContextManager[None]:
        return self._server.session_manager.run()

    async def _offer(self, identity: Identity) -> Offer:
        servers = [s for s in self._catalog.offered if s.open_to(identity.groups)]
        return await offer_tools(
            identity.user_id,
            servers,
            self._connections,
            self._tokens,
            self._settings.listing_seconds,
        )

    async def _list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        try:
            offer = await self._offer(_caller(ctx))
        except McpServerFailed as failure:
            raise MCPError(INTERNAL_ERROR, str(failure)) from None
        return ListToolsResult(tools=[_listed(name, tool) for name, tool in offer.tools.items()])

    async def _call_tool(
        self, ctx: ServerRequestContext[Any, Any], params: CallToolRequestParams
    ) -> CallToolResult:
        identity = _caller(ctx)
        started = time.monotonic()
        arguments = params.arguments or {}
        done = await self._run(ctx, identity, params.name, arguments)
        self._audit.gateway_call(
            identity,
            done.server,
            done.tool,
            arguments,
            done.outcome,
            started,
            done.text,
            done.decision,
            done.flagged,
        )
        executed = (
            ExecutedCall(done.server, done.tool, arguments, done.text, done.is_error)
            if done.ran
            else None
        )
        self._metrics.gateway_call_finished(done.outcome, executed, done.decision, done.flagged)
        return CallToolResult(content=[TextContent(text=done.text)], is_error=done.is_error)

    async def _run(
        self,
        ctx: ServerRequestContext[Any, Any],
        identity: Identity,
        name: str,
        arguments: dict[str, Any],
    ) -> _Done:
        if not await self._quota.allows(identity.user_id):
            return _Done("rate_limited", f"at most {self._settings.requests_per_minute} a minute")
        spending = Spending(
            self._settings.price_model,
            self._settings.budget_usd,
            self._spending_store,
            identity.user_id,
            month_of(datetime.now(UTC)),
        )
        try:
            spending.check()
        except BudgetExceeded as error:
            return _Done("over_budget", str(error))
        try:
            offer = await self._offer(identity)
        except McpServerFailed as failure:
            return _Done("upstream_error", str(failure))
        tool = offer.tools.get(name)
        if tool is None:
            return _Done("unknown_tool", self._unknown_tool_text(name, offer))
        return await self._call_offered(
            ctx, identity, name, tool, arguments, name in offer.always, spending
        )

    def _unknown_tool_text(self, name: str, offer: Offer) -> str:
        text = f"There is no tool named {name}."
        if offer.not_connected:
            text += (
                f" You have not connected {', '.join(offer.not_connected)} yet. "
                f"Connect them at {self._settings.public_url} to use their tools."
            )
        return text

    async def _call_offered(
        self,
        ctx: ServerRequestContext[Any, Any],
        identity: Identity,
        name: str,
        tool: OfferedTool,
        arguments: dict[str, Any],
        always_allowed: bool,
        spending: Spending,
    ) -> _Done:
        server_name, info = tool.server.name, tool.info
        finish = partial(_Done, server=server_name, tool=info.name)
        seen = self._seen.setdefault(ctx.session, [])
        decision = None
        if not info.read_only:
            verdict = await self._gate(ProposedCall("", server_name, info, arguments, tuple(seen)))
            if not verdict.allowed:
                return finish("blocked", f"This call was blocked and not run: {verdict.reason}")
            decision = "always" if always_allowed else await self._approval(ctx, tool, arguments)
            if decision == "no_elicitation":
                return finish("no_elicitation", self._cannot_ask_text())
            if decision == "timed_out":
                return finish("timed_out", "The employee did not answer in time. Nothing was run.")
            if decision == "reject":
                return finish("rejected", "The employee declined this call.", decision=decision)
        try:
            with self._timed(server_name, info.name, arguments):
                result = await self._connections.call(
                    identity.user_id, tool.server, info.name, arguments
                )
        except NotConnected:
            text = f"Connect {server_name} at {self._settings.public_url} and try again."
            return finish("not_connected", text, decision=decision)
        except (*UPSTREAM_ERRORS, BaseExceptionGroup) as error:
            failure = server_failure(server_name, error)
            if failure is None:
                raise
            return finish("upstream_error", str(failure), decision=decision)
        asked = f"call {name} with {json.dumps(arguments)}"
        screen = jev_screen(self._jev, spending.record_jev)
        text = await screen(asked, cut_output(result.text, self._settings.tool_output_chars))
        seen.append(text)
        return finish(
            "tool_error" if result.is_error else "ok",
            text,
            is_error=result.is_error,
            ran=True,
            decision=decision,
            flagged=is_flagged(text),
        )

    def _cannot_ask_text(self) -> str:
        return (
            "This call changes something and needs the employee's approval, but this MCP client "
            "cannot ask for it (it does not support elicitation). Nothing was run. Use a client "
            f"that supports elicitation, or make the request at {self._settings.public_url}."
        )

    async def _approval(
        self, ctx: ServerRequestContext[Any, Any], tool: OfferedTool, arguments: dict[str, Any]
    ) -> str:
        if not ctx.session.check_client_capability(
            ClientCapabilities(elicitation=ElicitationCapability())
        ):
            return "no_elicitation"
        message = (
            f"Allow {tool.server.name} to run {tool.info.name}?\n{json.dumps(arguments, indent=2)}"
        )
        try:
            async with asyncio.timeout(self._settings.ask_seconds):
                answer = await elicit_with_validation(
                    ctx.session, message, ApprovalForm, ctx.request_id
                )
        except TimeoutError:
            return "timed_out"
        except NoBackChannelError:
            return "no_elicitation"
        except ValueError:
            return "reject"
        approved = isinstance(answer, AcceptedElicitation) and answer.data.approve
        return "approve" if approved else "reject"
