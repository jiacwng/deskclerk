# The web front of the agent. An employee signs in with the company's OpenID Connect provider,
# connects their MCP servers and asks the agent, which waits for the employee's confirmation before
# a call that changes something. Browsers hold a session cookie that names a row in the session
# table; other clients send a bearer token. Every dependency is passed to create_app.

import asyncio
import base64
import hashlib
import hmac
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from typing import Any, Literal, NoReturn
from uuid import UUID, uuid4

import anthropic
import anyio.from_thread
import httpx
import psycopg
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.sse import KEEPALIVE_COMMENT, format_sse_event
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from deskclerk import cookies, oidc
from deskclerk.admin import AdminSummary, recent_rows, server_rows, usage_rows
from deskclerk.agent import (
    AlwaysAllowRefused,
    CallEnded,
    CallStarted,
    ConversationWaiting,
    Decision,
    Edit,
    ExecutedCall,
    Gate,
    McpServerFailed,
    NothingToConfirm,
    RunEvent,
    WrongDecisions,
    no_tool_span,
    run_agent,
    waiting_calls,
)
from deskclerk.audit import Audit
from deskclerk.auth import (
    AuthError,
    AuthUnavailable,
    Identity,
    SigningKeyLookup,
    authenticate,
    identify,
    logout_target,
)
from deskclerk.bodylimit import LimitRequestSize
from deskclerk.config import Settings
from deskclerk.connections import (
    UPSTREAM_ERRORS,
    AlreadyConnected,
    FlowExpired,
    NoSignInNeeded,
    OAuthConnections,
    UnknownState,
    WrongUser,
    first_leaf,
)
from deskclerk.conversations import (
    NoCheckpointBefore,
    delete_conversation,
    delete_dropped_turns,
    fork_before_request,
    last_request,
    list_conversations,
    open_checkpointer,
    past_turns,
    record_turn,
    saved_messages,
    thread_id,
)
from deskclerk.database import Database
from deskclerk.erasure import erase_user
from deskclerk.instructions import system_prompt
from deskclerk.jev import Jev
from deskclerk.metrics import Metrics
from deskclerk.pinning import MenuEntry, allowed_tool_names, read_menu, revoke_tool
from deskclerk.providers import BudgetExceeded, Usage, cost_of
from deskclerk.quota import Quota
from deskclerk.retention import delete_old_threads, delete_old_threads_daily
from deskclerk.scim import ScimUsers, mount_scim
from deskclerk.scope import jev_tool_selection
from deskclerk.screen import jev_screen
from deskclerk.servers import McpServer
from deskclerk.sessions import SessionStore
from deskclerk.spending import Spending, SpendingStore, month_of
from deskclerk.tokens import TokenStore
from deskclerk.tracing import Tracing

SESSION_COOKIE = "deskclerk_session"
LOGIN_COOKIE = "deskclerk_login"
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "deskclerk"
MAX_REQUEST_CHARS = 4000
LOGIN_LIFETIME_SECONDS = 600
SECONDS_PER_HOUR = 3600
BEARER = {"WWW-Authenticate": "Bearer"}
MAX_REASON_CHARS = 200
STREAM_PING_SECONDS = 15.0

_log = logging.getLogger(__name__)

CONNECT_ERRORS: dict[type[Exception], int] = {
    UnknownState: 400,
    WrongUser: 403,
    FlowExpired: 410,
    AlreadyConnected: 409,
    NoSignInNeeded: 400,
}
try:
    import openai

    MODEL_ERRORS: tuple[type[Exception], ...] = (anthropic.APIError, openai.APIError)
except ImportError:
    MODEL_ERRORS = (anthropic.APIError,)


def _clip(text: str | None) -> str:
    return repr((text or "")[:MAX_REASON_CHARS])


def _upstream_reason(error: BaseException) -> str:
    leaf = first_leaf(error)
    return f"{type(leaf).__name__}: {str(leaf)[:MAX_REASON_CHARS]}"


def _link_failure(code: str) -> RedirectResponse:
    return RedirectResponse(f"/?error={code}", status_code=303)


def _upstream_detail(error: BaseException) -> str:
    if isinstance(error, McpServerFailed):
        return f"the MCP server {error.server_name} did not answer ({error.cause})"
    if isinstance(error, MODEL_ERRORS):
        return f"the model did not answer ({type(error).__name__})"
    return f"an MCP server did not answer ({type(error).__name__})"


class MissingSessionKey(RuntimeError):
    pass


class FailedTurn(Exception):
    def __init__(self, status: int, detail: str, executed: list[ExecutedCall]) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.executed = executed


class AskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    conversation: UUID | None = None
    edit_of: int | None = Field(default=None, ge=0)


class EditIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edit: dict[str, Any]


class ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: UUID
    decisions: dict[str, Literal["approve", "approve_always", "reject"] | EditIn]


class ServerOut(BaseModel):
    name: str
    url: str
    title: str | None
    description: str | None
    examples: list[str]
    connected: bool
    tools: list[MenuEntry]
    always_allowed: list[str]


class ConnectOut(BaseModel):
    authorization_url: str


class ExecutedOut(BaseModel):
    server: str
    tool: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


class BlockedOut(BaseModel):
    server: str
    tool: str
    arguments: dict[str, Any]
    reason: str
    rejected: bool


class PendingOut(BaseModel):
    id: str
    server: str
    tool: str
    description: str
    arguments: dict[str, Any]
    destructive: bool
    always_allowable: bool


class TurnOut(BaseModel):
    request: str
    answer: str


class ConversationOut(BaseModel):
    conversation: UUID
    turns: list[TurnOut]
    pending: list[PendingOut]


class ConversationSummaryOut(BaseModel):
    id: UUID
    title: str
    updated_at: datetime


class AskOut(BaseModel):
    conversation: UUID
    answer: str
    executed: list[ExecutedOut]
    blocked: list[BlockedOut]
    pending: list[PendingOut]
    not_connected: list[str]
    unavailable: list[str]
    step_limit_hit: bool
    cut_off: bool
    flagged: int
    changed_tools: list[str]
    cost_usd: float


class CallOut(BaseModel):
    id: str
    server: str
    tool: str
    read_only: bool
    status: Literal["running", "ok", "error"]


class AnswerOut(BaseModel):
    text: str


class ErrorOut(BaseModel):
    status: int
    detail: str
    calls: list[ExecutedOut]


@dataclass(frozen=True)
class StreamEvent:
    name: Literal["call", "pending", "answer", "done", "error"]
    data: BaseModel


Emit = Callable[[StreamEvent], None]


def _stream_event(event: RunEvent) -> StreamEvent:
    if isinstance(event, CallStarted):
        return StreamEvent(
            "call",
            CallOut(
                id=event.id,
                server=event.server,
                tool=event.tool,
                read_only=event.read_only,
                status="running",
            ),
        )
    if isinstance(event, CallEnded):
        status = "error" if event.is_error else "ok"
        return StreamEvent(
            "call",
            CallOut(
                id=event.id,
                server=event.server,
                tool=event.tool,
                read_only=event.read_only,
                status=status,
            ),
        )
    return StreamEvent("answer", AnswerOut(text=event.text))


def _wire(event: StreamEvent) -> bytes:
    return format_sse_event(data_str=event.data.model_dump_json(), event=event.name)


async def _wire_events(
    queue: asyncio.Queue[StreamEvent | None], producer: asyncio.Task[None]
) -> AsyncIterator[bytes]:
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), STREAM_PING_SECONDS)
            except TimeoutError:
                yield KEEPALIVE_COMMENT
                continue
            if event is None:
                return
            yield _wire(event)
    finally:
        producer.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.wait([producer])


@dataclass(frozen=True)
class Caller:
    identity: Identity
    by_cookie: bool


VENDOR_SCRIPTS = ("marked-18.1.0.umd.js", "purify-3.4.16.min.js")


def _sha256_source(data: bytes) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()


# The page has one inline script, one style block and two vendored scripts, so the policy allows
# exactly those by hash and nothing else. Each vendored script carries an integrity attribute that
# must equal the hash of the file the app serves, which browsers need to match a hash to a file.
def _content_security_policy(page: str, vendor: dict[str, bytes]) -> str:
    blocks = {}
    for tag in ("script", "style"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", page, re.DOTALL)
        if match is None:
            raise ValueError(f"index.html has no <{tag}> block")
        blocks[tag] = [_sha256_source(match.group(1).encode())]
    for name, data in vendor.items():
        source = _sha256_source(data)
        if f'<script src="/vendor/{name}" integrity="{source}"></script>' not in page:
            raise ValueError(f"index.html does not load vendor/{name} with integrity {source}")
        blocks["script"].append(source)
    fixed = ["default-src 'none'", "connect-src 'self'", "base-uri 'none'"]
    fixed += ["form-action 'self'", "frame-ancestors 'none'"]
    allowed = [
        f"{tag}-src " + " ".join(f"'{source}'" for source in found) for tag, found in blocks.items()
    ]
    return "; ".join([*fixed, *allowed])


def create_app(
    settings: Settings,
    database: Database,
    servers: list[McpServer],
    connections: OAuthConnections,
    tokens: TokenStore,
    spending_store: SpendingStore,
    sessions: SessionStore,
    model: BaseChatModel,
    gate: Gate,
    jev: Jev | None,
    signing_key: SigningKeyLookup,
    registry: CollectorRegistry,
    tracing: Tracing | None = None,
    idp_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    if settings.session_key is None:
        raise MissingSessionKey("DESKCLERK_SESSION_KEY is not set; put it in .env")
    session_key = settings.session_key.get_secret_value().encode()
    secure_cookies = settings.sign_in_redirect_url.startswith("https://")
    session_seconds = settings.session_hours * SECONDS_PER_HOUR
    cost_of(settings.price_model, Usage(0, 0, 0, 0))
    quota = Quota(database, settings.requests_per_minute)
    audit = Audit(settings.audit_text, settings.audit_key)
    metrics = Metrics(registry)
    discovered: dict[str, oidc.Endpoints] = {}
    ask_failures: tuple[type[Exception], ...] = (
        McpServerFailed,
        *MODEL_ERRORS,
        *UPSTREAM_ERRORS,
    )
    server_by_name = {server.name: server for server in servers}
    page = resources.files("deskclerk").joinpath("static/index.html").read_text(encoding="utf-8")
    vendor = {
        name: resources.files("deskclerk").joinpath(f"static/vendor/{name}").read_bytes()
        for name in VENDOR_SCRIPTS
    }
    admin_page = resources.files("deskclerk").joinpath("static/admin.html").read_text("utf-8")
    page_headers = {
        "Content-Security-Policy": _content_security_policy(page, vendor),
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }
    admin_headers = {
        **page_headers,
        "Content-Security-Policy": _content_security_policy(admin_page, {}),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with database.serving(), connections.listening():
            checkpointer = await open_checkpointer(database, tokens.fernet)
            app.state.checkpointer = checkpointer
            sweep = None
            if settings.retention_days > 0:
                await delete_old_threads(database, checkpointer, settings.retention_days)
                sweep = asyncio.create_task(
                    delete_old_threads_daily(database, checkpointer, settings.retention_days)
                )
            yield
            if sweep is not None:
                sweep.cancel()

    app = FastAPI(
        title="deskclerk", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    app.add_middleware(LimitRequestSize)

    @app.exception_handler(FailedTurn)
    async def failed_turn(_: Request, failure: FailedTurn) -> JSONResponse:
        calls = [ExecutedOut(**vars(call)).model_dump() for call in failure.executed]
        return JSONResponse({"detail": failure.detail, "calls": calls}, status_code=failure.status)

    scim_users = ScimUsers(database) if settings.scim_token is not None else None

    async def is_deactivated(user_id: str) -> bool:
        return scim_users is not None and await run_in_threadpool(
            scim_users.is_deactivated, user_id
        )

    async def refuse_deactivated(user_id: str) -> None:
        if await is_deactivated(user_id):
            raise HTTPException(403, "this account has been deactivated")

    async def caller(request: Request) -> Caller:
        header = request.headers.get("authorization")
        if header is not None:
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise HTTPException(401, "send Authorization: Bearer <access token>", BEARER)
            try:
                identity = await run_in_threadpool(
                    authenticate, token, settings, signing_key, settings.audience
                )
            except AuthError as error:
                raise HTTPException(401, str(error), BEARER) from None
            except AuthUnavailable:
                raise HTTPException(503, "the login server cannot be reached") from None
            await refuse_deactivated(identity.user_id)
            return Caller(identity, by_cookie=False)
        session_identity = await run_in_threadpool(
            sessions.lookup, request.cookies.get(SESSION_COOKIE, "")
        )
        if session_identity is None:
            raise HTTPException(401, "sign in at /login", BEARER)
        await refuse_deactivated(session_identity.user_id)
        return Caller(session_identity, by_cookie=True)

    async def posting_caller(request: Request, who: Caller = Depends(caller)) -> Caller:
        if who.by_cookie and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
            raise HTTPException(403, f"send the header {CSRF_HEADER}: {CSRF_VALUE}")
        return who

    def is_admin(identity: Identity) -> bool:
        return bool(settings.admin_group) and settings.admin_group in identity.groups

    async def admin_caller(who: Caller = Depends(caller)) -> Caller:
        if not is_admin(who.identity):
            raise HTTPException(403, "this page is for the admins group")
        return who

    def known_server(name: str, identity: Identity) -> McpServer:
        server = server_by_name.get(name)
        if server is None or not server.open_to(identity.groups):
            raise HTTPException(404, f"no server named {name}")
        return server

    def idp_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=idp_transport, timeout=oidc.TIMEOUT_SECONDS)

    async def endpoints() -> oidc.Endpoints:
        if "endpoints" not in discovered:
            async with idp_client() as client:
                discovered["endpoints"] = await oidc.discover(settings, client)
        return discovered["endpoints"]

    async def erase_when_idle(user_id: str) -> None:
        async with database.user_lock(user_id):
            await connections.cancel_all_for(user_id)
            await run_in_threadpool(erase_user, user_id, tokens, sessions, spending_store, audit)

    if settings.scim_token is not None and scim_users is not None:
        mount_scim(
            app,
            settings.scim_token.get_secret_value(),
            settings.public_url,
            scim_users,
            lambda user_id: anyio.from_thread.run(erase_when_idle, user_id),
        )

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        failed = []
        try:
            await run_in_threadpool(tokens.check)
        except psycopg.Error:
            failed.append("token store")
        if not hasattr(app.state, "checkpointer"):
            failed.append("checkpointer")
        try:
            await endpoints()
        except AuthUnavailable:
            failed.append("login server discovery")
        if failed:
            return JSONResponse({"not_ready": failed}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(page, headers=page_headers)

    if settings.admin_group:

        @app.get("/admin", response_class=HTMLResponse)
        async def admin(request: Request) -> Response:
            try:
                await admin_caller(await caller(request))
            except HTTPException as refused:
                if refused.status_code == 401:
                    return RedirectResponse("/login", status_code=302)
                raise
            return HTMLResponse(admin_page, headers=admin_headers)

        @app.get("/admin/summary")
        async def admin_summary(_: Caller = Depends(admin_caller)) -> AdminSummary:
            month = month_of(datetime.now(UTC))
            return AdminSummary(
                month=month,
                servers=await run_in_threadpool(server_rows, servers, tokens),
                usage=await run_in_threadpool(
                    usage_rows, spending_store, month, settings.budget_usd
                ),
                recent=await recent_rows(database, settings.audit_text),
            )

    @app.get("/vendor/{name}")
    async def vendor_script(name: str) -> Response:
        script = vendor.get(name)
        if script is None:
            raise HTTPException(404, f"no script named {name}")
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "public, max-age=31536000, immutable",
        }
        return Response(script, media_type="text/javascript", headers=headers)

    def sign_in_failed(code: str, reason: str) -> RedirectResponse:
        _log.warning("sign-in failed: %s", reason)
        return _link_failure(code)

    @app.get("/login")
    async def login() -> Response:
        try:
            found = await endpoints()
        except AuthUnavailable as unreachable:
            return sign_in_failed("login_unreachable", str(unreachable))
        begun = oidc.start(settings, found)
        pending = {"state": begun.state, "verifier": begun.verifier, "nonce": begun.nonce}
        value = cookies.sign(session_key, "login", pending, LOGIN_LIFETIME_SECONDS)
        response = RedirectResponse(begun.url, status_code=302)
        response.set_cookie(
            LOGIN_COOKIE,
            value,
            LOGIN_LIFETIME_SECONDS,
            path="/login",
            httponly=True,
            samesite="lax",
            secure=secure_cookies,
        )
        return response

    @app.get("/login/callback")
    async def login_callback(
        request: Request,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> Response:
        if error is not None:
            reason = f"the login server returned error={_clip(error)} "
            return sign_in_failed(
                "sign_in_refused", reason + f"error_description={_clip(error_description)}"
            )
        pending = cookies.unsign(session_key, "login", request.cookies.get(LOGIN_COOKIE, ""))
        if pending is None:
            return sign_in_failed(
                "sign_in_refused", "the login cookie is missing, expired or forged"
            )
        if code is None or state is None:
            return sign_in_failed("sign_in_refused", "the callback carries no code or no state")
        if not hmac.compare_digest(str(pending.get("state", "")).encode(), state.encode()):
            return sign_in_failed(
                "sign_in_refused", "the callback state does not match the login cookie"
            )
        try:
            found = await endpoints()
        except AuthUnavailable as unreachable:
            return sign_in_failed("login_unreachable", str(unreachable))
        async with idp_client() as client:
            try:
                verifier = str(pending.get("verifier", ""))
                token = await oidc.exchange_code(settings, found, client, code, verifier)
            except oidc.SignInRefused as refused:
                return sign_in_failed("sign_in_refused", str(refused))
            except oidc.NoIdToken as missing:
                return sign_in_failed("sign_in_refused", str(missing))
            except AuthUnavailable as unreachable:
                return sign_in_failed("login_unreachable", str(unreachable))
        try:
            identity = await run_in_threadpool(
                identify, token, settings, signing_key, str(pending.get("nonce", ""))
            )
        except AuthError as refused:
            return sign_in_failed("sign_in_refused", f"the ID token was refused: {refused}")
        except AuthUnavailable as unreachable:
            return sign_in_failed("login_unreachable", str(unreachable))
        if await is_deactivated(identity.user_id):
            return sign_in_failed("sign_in_refused", "account deactivated")
        session = await run_in_threadpool(sessions.create, identity, session_seconds)
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(LOGIN_COOKIE, path="/login")
        response.set_cookie(
            SESSION_COOKIE,
            session,
            session_seconds,
            path="/",
            httponly=True,
            samesite="lax",
            secure=secure_cookies,
        )
        return response

    @app.post("/logout", status_code=204)
    async def logout(request: Request, _: Caller = Depends(posting_caller)) -> Response:
        await run_in_threadpool(sessions.end, request.cookies.get(SESSION_COOKIE, ""))
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.post("/logout/backchannel")
    async def backchannel_logout(logout_token: str = Form()) -> Response:
        try:
            target = await run_in_threadpool(logout_target, logout_token, settings, signing_key)
        except AuthError:
            raise HTTPException(400, "the logout token was refused") from None
        except AuthUnavailable:
            raise HTTPException(503, "the login server cannot be reached") from None
        if target.sid is not None:
            await run_in_threadpool(sessions.end_by_sid, target.sid)
        elif target.sub is not None:
            await run_in_threadpool(sessions.end_by_subject, target.sub)
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    @app.get("/me")
    async def me(who: Caller = Depends(caller)) -> dict[str, str | float | bool]:
        month = month_of(datetime.now(UTC))
        spent = await run_in_threadpool(spending_store.load, who.identity.user_id, month)
        return {
            "user": who.identity.user_id,
            "name": who.identity.display_name,
            "spent_usd": spent,
            "budget_usd": settings.budget_usd,
            "admin": is_admin(who.identity),
            "unscreened": jev is None,
        }

    @app.get("/servers")
    async def list_servers(who: Caller = Depends(caller)) -> list[ServerOut]:
        user = who.identity.user_id
        listed = []
        for server in servers:
            if not server.open_to(who.identity.groups):
                continue
            connected = (
                server.auth == "none"
                or await run_in_threadpool(tokens.get, user, server.name, "tokens") is not None
            )
            menu = (
                await run_in_threadpool(read_menu, tokens, user, server.name) if connected else []
            )
            allowed = [t for t in menu if server.tools is None or t.name in server.tools]
            menu_names = {t.name for t in allowed}
            always = (
                [
                    name
                    for name in await run_in_threadpool(
                        allowed_tool_names, tokens, user, server.name
                    )
                    if name in menu_names
                ]
                if connected
                else []
            )
            listed.append(
                ServerOut(
                    name=server.name,
                    url=server.url,
                    title=server.title,
                    description=server.description,
                    examples=list(server.examples),
                    connected=connected,
                    tools=allowed,
                    always_allowed=always,
                )
            )
        return listed

    @app.post("/servers/{name}/connect")
    async def connect(name: str, who: Caller = Depends(posting_caller)) -> ConnectOut:
        server = known_server(name, who.identity)
        try:
            url = await connections.begin(who.identity.user_id, server)
        except tuple(CONNECT_ERRORS) as error:
            raise HTTPException(CONNECT_ERRORS[type(error)], str(error)) from None
        except UPSTREAM_ERRORS as failure:
            _log.warning("connect to %s failed to start: %s", name, _upstream_reason(failure))
            raise HTTPException(502, f"cannot start the sign-in to {name}") from None
        return ConnectOut(authorization_url=url)

    @app.get("/oauth/client.json")
    async def oauth_client_document() -> JSONResponse:
        document = connections.client_metadata().model_dump(mode="json", exclude_none=True)
        return JSONResponse(
            document | {"client_id": settings.client_metadata_url},
            headers={"Cache-Control": "public, max-age=3600"},
        )

    def connect_failed(failure: Exception) -> RedirectResponse:
        _log.warning("connect failed: %s", failure)
        expired = isinstance(failure, FlowExpired | UnknownState)
        return _link_failure("connect_expired" if expired else "connect_failed")

    @app.get("/oauth/callback")
    async def oauth_callback(
        state: str,
        code: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
        iss: str | None = None,
        who: Caller = Depends(caller),
    ) -> Response:
        if error is not None or code is None:
            try:
                await connections.cancel(who.identity.user_id, state)
            except tuple(CONNECT_ERRORS) as refused:
                return connect_failed(refused)
            _log.warning(
                "connect refused by the server: error=%s error_description=%s",
                _clip(error),
                _clip(error_description),
            )
            return _link_failure("connect_failed")
        try:
            server_name = await connections.complete(who.identity.user_id, state, code, iss)
        except tuple(CONNECT_ERRORS) as failure:
            return connect_failed(failure)
        except UPSTREAM_ERRORS as failure:
            _log.warning("connect failed to complete: %s", _upstream_reason(failure))
            return _link_failure("connect_failed")
        audit.connection(who.identity, "connect", server_name)
        return RedirectResponse("/", status_code=303)

    @app.post("/servers/{name}/disconnect", status_code=204)
    async def disconnect(name: str, who: Caller = Depends(posting_caller)) -> Response:
        server = known_server(name, who.identity)
        await connections.cancel_for(who.identity.user_id, server.name)
        await run_in_threadpool(tokens.delete_connection, who.identity.user_id, server.name)
        audit.connection(who.identity, "disconnect", server.name)
        return Response(status_code=204)

    @app.delete("/servers/{name}/always/{tool}", status_code=204)
    async def revoke_always_allowed(
        name: str, tool: str, who: Caller = Depends(posting_caller)
    ) -> Response:
        server = known_server(name, who.identity)
        user = who.identity.user_id
        async with database.user_lock(user):
            revoked = await run_in_threadpool(revoke_tool, tokens, user, server.name, tool)
        if not revoked:
            raise HTTPException(404, f"{tool} is not always allowed")
        return Response(status_code=204)

    @app.get("/conversations")
    async def conversation_list(who: Caller = Depends(caller)) -> list[ConversationSummaryOut]:
        found = await list_conversations(database, who.identity.user_id)
        return [ConversationSummaryOut(**vars(summary)) for summary in found]

    @app.delete("/conversations/{conversation_id}", status_code=204)
    async def remove_conversation(
        conversation_id: UUID, who: Caller = Depends(posting_caller)
    ) -> Response:
        checkpointer: AsyncPostgresSaver = app.state.checkpointer
        user_id = who.identity.user_id
        async with database.user_lock(user_id):
            if not await saved_messages(checkpointer, thread_id(user_id, conversation_id)):
                raise HTTPException(404, "no such conversation")
            await delete_conversation(database, checkpointer, user_id, conversation_id)
        return Response(status_code=204)

    @app.get("/conversations/{conversation_id}")
    async def conversation_turns(
        conversation_id: UUID, who: Caller = Depends(caller)
    ) -> ConversationOut:
        checkpointer: AsyncPostgresSaver = app.state.checkpointer
        thread = thread_id(who.identity.user_id, conversation_id)
        history = await saved_messages(checkpointer, thread)
        if not history:
            raise HTTPException(404, "no such conversation")
        waiting = await waiting_calls(
            who.identity.user_id,
            [s for s in servers if s.open_to(who.identity.groups)],
            connections,
            tokens,
            checkpointer,
            thread,
            settings.listing_seconds,
        )
        return ConversationOut(
            conversation=conversation_id,
            turns=[
                TurnOut(request=request, answer=answer) for request, answer in past_turns(history)
            ],
            pending=[PendingOut(**vars(call)) for call in waiting],
        )

    @app.post("/ask", response_model=AskOut)
    async def ask(
        request: Request, body: AskBody, who: Caller = Depends(posting_caller)
    ) -> AskOut | StreamingResponse:
        return await answer_as_asked(
            request, who.identity, body.conversation, body.request, None, body.edit_of
        )

    @app.post("/ask/confirm", response_model=AskOut)
    async def confirm(
        request: Request, body: ConfirmBody, who: Caller = Depends(posting_caller)
    ) -> AskOut | StreamingResponse:
        decisions: dict[str, Decision] = {
            call_id: Edit(choice.edit) if isinstance(choice, EditIn) else choice
            for call_id, choice in body.decisions.items()
        }
        return await answer_as_asked(request, who.identity, body.conversation, "", decisions, None)

    async def answer_as_asked(
        request: Request,
        identity: Identity,
        conversation_id: UUID | None,
        text: str,
        decisions: dict[str, Decision] | None,
        edit_of: int | None,
    ) -> AskOut | StreamingResponse:
        conversation = conversation_id or uuid4()
        if "text/event-stream" not in request.headers.get("accept", ""):
            return await take_turn(
                identity,
                conversation_id,
                conversation,
                text,
                decisions,
                edit_of,
                lambda event: None,
                lambda: None,
            )
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        started = asyncio.Event()

        async def produce() -> None:
            try:
                done = await take_turn(
                    identity,
                    conversation_id,
                    conversation,
                    text,
                    decisions,
                    edit_of,
                    queue.put_nowait,
                    started.set,
                )
                queue.put_nowait(StreamEvent("done", done))
            except HTTPException:
                if not started.is_set():
                    raise
                failure = ErrorOut(status=500, detail="the ask failed", calls=[])
                queue.put_nowait(StreamEvent("error", failure))
            except FailedTurn as failed:
                calls = [ExecutedOut(**vars(call)) for call in failed.executed]
                failure = ErrorOut(status=failed.status, detail=failed.detail, calls=calls)
                queue.put_nowait(StreamEvent("error", failure))
            except Exception:
                _log.exception("an ask failed")
                queue.put_nowait(
                    StreamEvent("error", ErrorOut(status=500, detail="the ask failed", calls=[]))
                )
            finally:
                queue.put_nowait(None)

        producer = asyncio.create_task(produce())
        waiter = asyncio.create_task(started.wait())
        try:
            await asyncio.wait({producer, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            producer.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait([producer])
            raise
        finally:
            waiter.cancel()
        if producer.done() and not started.is_set() and (refusal := producer.exception()):
            raise refusal
        return StreamingResponse(
            _wire_events(queue, producer),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Conversation": str(conversation),
            },
        )

    async def take_turn(
        identity: Identity,
        conversation_id: UUID | None,
        conversation: UUID,
        text: str,
        decisions: dict[str, Decision] | None,
        edit_of: int | None,
        emit: Emit,
        running: Callable[[], None],
    ) -> AskOut:
        entry = "ask" if decisions is None else "confirm"
        started = time.monotonic()

        def refuse(
            status: int, detail: str, outcome: str, headers: dict[str, str] | None = None
        ) -> NoReturn:
            audit.asked(
                identity,
                conversation_id,
                text,
                started,
                outcome,
                None,
                0.0,
                entry=entry,
                edit_of=edit_of,
            )
            metrics.ask_finished(outcome, started, 0.0, (), None)
            raise HTTPException(status, detail, headers)

        if not await quota.allows(identity.user_id):
            refuse(
                429,
                f"at most {settings.requests_per_minute} asks a minute",
                "rate_limited",
                {"Retry-After": "60"},
            )
        checkpointer: AsyncPostgresSaver = app.state.checkpointer
        thread = thread_id(identity.user_id, conversation)
        async with database.user_lock(identity.user_id):
            await refuse_deactivated(identity.user_id)
            history = await saved_messages(checkpointer, thread)
            if (conversation_id is not None or edit_of is not None) and not history:
                refuse(404, "no such conversation", "unknown_conversation")
            fork_from = None
            if edit_of is not None:
                if edit_of >= len(past_turns(history)):
                    refuse(422, "no such request to edit", "wrong_edit")
                try:
                    fork = await fork_before_request(checkpointer, thread, edit_of)
                except NoCheckpointBefore:
                    refuse(422, "that request can no longer be edited", "wrong_edit")
                fork_from, history = fork.checkpoint_id, fork.messages
            if decisions is not None:
                text = last_request(history)
            month = month_of(datetime.now(UTC))
            spending = Spending(
                settings.price_model, settings.budget_usd, spending_store, identity.user_id, month
            )
            try:
                spending.check()
            except BudgetExceeded as error:
                refuse(402, str(error), "over_budget")
            spent_before = spending.cost_usd
            executed: list[ExecutedCall] = []
            try:
                with tracing.ask(identity.user_id) if tracing else nullcontext(None) as trace:
                    async with asyncio.timeout(settings.ask_seconds):
                        result = await run_agent(
                            identity.user_id,
                            text,
                            system_prompt(datetime.now(UTC).date()),
                            [s for s in servers if s.open_to(identity.groups)],
                            connections,
                            tokens,
                            checkpointer,
                            thread,
                            history,
                            model,
                            gate,
                            jev_screen(jev, spending.record_jev),
                            settings.step_limit,
                            spending.record,
                            trace.config if trace else None,
                            metrics.timed(trace.tool_span if trace else no_tool_span),
                            jev_tool_selection(jev, settings.tool_selection, spending.record_jev),
                            tool_output_chars=settings.tool_output_chars,
                            history_tokens=settings.history_tokens,
                            listing_seconds=settings.listing_seconds,
                            confirm_writes=True,
                            decisions=decisions,
                            executed=executed,
                            emit=lambda event: emit(_stream_event(event)),
                            running=running,
                            fork_from=fork_from,
                        )
                    if trace:
                        trace.finish(result.flagged)
            except ConversationWaiting:
                refuse(409, "this conversation waits for a confirmation", "conversation_waiting")
            except NothingToConfirm:
                refuse(409, "this conversation has nothing to confirm", "nothing_pending")
            except WrongDecisions:
                refuse(422, "decide every pending call exactly once", "wrong_decisions")
            except AlwaysAllowRefused:
                refuse(
                    422,
                    "only a call that does not delete or overwrite can be always allowed",
                    "wrong_decisions",
                )
            except asyncio.CancelledError:
                metrics.ask_finished(
                    "cancelled", started, spending.cost_usd - spent_before, executed, None
                )
                audit.asked(
                    identity,
                    conversation,
                    text,
                    started,
                    "cancelled",
                    None,
                    spending.cost_usd - spent_before,
                    "the client stopped the request",
                    entry,
                    executed,
                    edit_of,
                )
                await record_turn(database, identity.user_id, conversation, text, edit_of == 0)
                raise
            except TimeoutError:
                cost = spending.cost_usd - spent_before
                detail = (
                    f"the ask took longer than {settings.ask_seconds} seconds "
                    "(set DESKCLERK_ASK_SECONDS to change it)"
                )
                audit.asked(
                    identity,
                    conversation,
                    text,
                    started,
                    "timed_out",
                    None,
                    cost,
                    detail,
                    entry,
                    executed,
                    edit_of,
                )
                metrics.ask_finished("timed_out", started, cost, executed, None)
                raise FailedTurn(504, detail, executed) from None
            except ask_failures as error:
                cost = spending.cost_usd - spent_before
                raise upstream_failure(
                    identity, conversation, text, started, cost, error, entry, executed, edit_of
                ) from None
            except BaseExceptionGroup as group:
                failed = group.subgroup(ask_failures)
                if failed is None:
                    raise
                cost = spending.cost_usd - spent_before
                raise upstream_failure(
                    identity,
                    conversation,
                    text,
                    started,
                    cost,
                    first_leaf(failed),
                    entry,
                    executed,
                    edit_of,
                ) from None
            await delete_dropped_turns(database, thread)
            await record_turn(database, identity.user_id, conversation, text, edit_of == 0)
        cost = spending.cost_usd - spent_before
        outcome = "awaiting_confirmation" if result.pending else "answered"
        audit.asked(
            identity,
            conversation,
            text,
            started,
            outcome,
            result,
            cost,
            entry=entry,
            executed=result.executed,
            edit_of=edit_of,
        )
        metrics.ask_finished(outcome, started, cost, result.executed, result)
        for call in result.pending:
            emit(StreamEvent("pending", PendingOut(**vars(call))))
        return AskOut(
            conversation=conversation,
            answer=result.answer,
            executed=[ExecutedOut(**vars(call)) for call in result.executed],
            blocked=[BlockedOut(**vars(call)) for call in result.blocked],
            pending=[PendingOut(**vars(call)) for call in result.pending],
            not_connected=list(result.not_connected),
            unavailable=list(result.unavailable),
            step_limit_hit=result.step_limit_hit,
            cut_off=result.cut_off,
            flagged=result.flagged,
            changed_tools=list(result.changed_tools),
            cost_usd=cost,
        )

    def upstream_failure(
        identity: Identity,
        conversation: UUID | None,
        request: str,
        started: float,
        cost: float,
        error: BaseException,
        entry: str,
        executed: list[ExecutedCall],
        edit_of: int | None,
    ) -> FailedTurn:
        detail = _upstream_detail(error)
        audit.asked(
            identity,
            conversation,
            request,
            started,
            "upstream_error",
            None,
            cost,
            detail,
            entry,
            executed,
            edit_of,
        )
        metrics.ask_finished("upstream_error", started, cost, executed, None)
        return FailedTurn(502, detail, executed)

    return app
