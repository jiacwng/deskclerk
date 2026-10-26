# The web front of the agent. An employee signs in with the company's OpenID Connect provider,
# connects their MCP servers and asks the agent, which waits for the employee's confirmation before
# a call that changes something. Browsers hold a session cookie that names a row in the session
# table; other clients send a bearer token. Every dependency is passed to create_app.

import asyncio
import base64
import hashlib
import hmac
import re
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from typing import Any, NoReturn
from uuid import UUID, uuid4

import anthropic
import anyio.from_thread
import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from deskclerk import cookies, oidc
from deskclerk.agent import (
    AlwaysAllowRefused,
    ConversationWaiting,
    Decision,
    ExecutedCall,
    Gate,
    McpServerFailed,
    NothingToConfirm,
    WrongDecisions,
    no_tool_span,
    run_agent,
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
    last_request,
    open_checkpointer,
    saved_messages,
    thread_id,
)
from deskclerk.erasure import erase_user
from deskclerk.instructions import system_prompt
from deskclerk.jev import Jev
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
SIGN_IN_FAILED = "sign-in was refused or expired; start again at /login"

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
BROWSER_AUTH_REFUSED = "the login server's ID token was refused"


def _upstream_detail(error: BaseException) -> str:
    if isinstance(error, McpServerFailed):
        return f"the MCP server {error.server_name} did not answer ({error.cause})"
    if isinstance(error, MODEL_ERRORS):
        return f"the model did not answer ({type(error).__name__})"
    return f"an MCP server did not answer ({type(error).__name__})"


class MissingSessionKey(RuntimeError):
    pass


class AskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    conversation: UUID | None = None


class ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: UUID
    decisions: dict[str, Decision]


class ServerOut(BaseModel):
    name: str
    url: str
    connected: bool


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
    servers: list[McpServer],
    connections: OAuthConnections,
    tokens: TokenStore,
    spending_store: SpendingStore,
    sessions: SessionStore,
    model: BaseChatModel,
    gate: Gate,
    jev: Jev | None,
    signing_key: SigningKeyLookup,
    tracing: Tracing | None = None,
    idp_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    if settings.session_key is None:
        raise MissingSessionKey("DESKCLERK_SESSION_KEY is not set; put it in .env")
    session_key = settings.session_key.get_secret_value().encode()
    secure_cookies = settings.sign_in_redirect_url.startswith("https://")
    session_seconds = settings.session_hours * SECONDS_PER_HOUR
    cost_of(settings.price_model, Usage(0, 0, 0, 0))
    quota = Quota(settings.requests_per_minute)
    audit = Audit(settings.audit_text, settings.audit_key)
    spendings: dict[tuple[str, str], Spending] = {}
    ask_locks: dict[str, asyncio.Lock] = {}
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
    page_headers = {
        "Content-Security-Policy": _content_security_policy(page, vendor),
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_checkpointer(tokens.path, tokens.fernet) as checkpointer:
            app.state.checkpointer = checkpointer
            sweep = None
            if settings.retention_days > 0:
                await delete_old_threads(checkpointer, settings.retention_days)
                sweep = asyncio.create_task(
                    delete_old_threads_daily(checkpointer, settings.retention_days)
                )
            yield
            if sweep is not None:
                sweep.cancel()

    app = FastAPI(
        title="deskclerk", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    app.add_middleware(LimitRequestSize)

    scim_users = ScimUsers(tokens.path) if settings.scim_token is not None else None

    async def refuse_deactivated(user_id: str) -> None:
        if scim_users is not None and await run_in_threadpool(scim_users.is_deactivated, user_id):
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
                try:
                    discovered["endpoints"] = await oidc.discover(settings, client)
                except AuthUnavailable:
                    raise HTTPException(503, "the login server cannot be reached") from None
        return discovered["endpoints"]

    async def erase_when_idle(user_id: str) -> None:
        async with ask_locks.setdefault(user_id, asyncio.Lock()):
            connections.cancel_all_for(user_id)
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

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        failed = []
        try:
            await run_in_threadpool(tokens.check)
        except sqlite3.Error:
            failed.append("token store")
        if not hasattr(app.state, "checkpointer"):
            failed.append("checkpointer")
        try:
            await endpoints()
        except HTTPException:
            failed.append("login server discovery")
        if failed:
            return JSONResponse({"not_ready": failed}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(page, headers=page_headers)

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

    @app.get("/login")
    async def login() -> Response:
        begun = oidc.start(settings, await endpoints())
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
        request: Request, code: str | None = None, state: str | None = None
    ) -> Response:
        pending = cookies.unsign(session_key, "login", request.cookies.get(LOGIN_COOKIE, ""))
        if pending is None or code is None or state is None:
            raise HTTPException(400, SIGN_IN_FAILED)
        if not hmac.compare_digest(str(pending.get("state", "")).encode(), state.encode()):
            raise HTTPException(400, SIGN_IN_FAILED)
        found = await endpoints()
        async with idp_client() as client:
            try:
                verifier = str(pending.get("verifier", ""))
                token = await oidc.exchange_code(settings, found, client, code, verifier)
            except oidc.SignInRefused:
                raise HTTPException(400, SIGN_IN_FAILED) from None
            except oidc.NoIdToken:
                raise HTTPException(401, BROWSER_AUTH_REFUSED) from None
            except AuthUnavailable:
                raise HTTPException(503, "the login server cannot be reached") from None
        try:
            identity = await run_in_threadpool(
                identify, token, settings, signing_key, str(pending.get("nonce", ""))
            )
        except AuthError:
            raise HTTPException(401, BROWSER_AUTH_REFUSED) from None
        except AuthUnavailable:
            raise HTTPException(503, "the login server cannot be reached") from None
        await refuse_deactivated(identity.user_id)
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
    async def me(who: Caller = Depends(caller)) -> dict[str, str]:
        return {"user": who.identity.user_id, "name": who.identity.display_name}

    @app.get("/servers")
    async def list_servers(who: Caller = Depends(caller)) -> list[ServerOut]:
        user = who.identity.user_id
        return [
            ServerOut(
                name=server.name,
                url=server.url,
                connected=server.auth == "none"
                or await run_in_threadpool(tokens.get, user, server.name, "tokens") is not None,
            )
            for server in servers
            if server.open_to(who.identity.groups)
        ]

    @app.post("/servers/{name}/connect")
    async def connect(name: str, who: Caller = Depends(posting_caller)) -> ConnectOut:
        server = known_server(name, who.identity)
        try:
            url = await connections.begin(who.identity.user_id, server)
        except tuple(CONNECT_ERRORS) as error:
            raise HTTPException(CONNECT_ERRORS[type(error)], str(error)) from None
        except UPSTREAM_ERRORS:
            raise HTTPException(502, f"cannot start the sign-in to {name}") from None
        return ConnectOut(authorization_url=url)

    @app.get("/oauth/client.json")
    async def oauth_client_document() -> JSONResponse:
        document = connections.client_metadata().model_dump(mode="json", exclude_none=True)
        return JSONResponse(
            document | {"client_id": settings.client_metadata_url},
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/oauth/callback")
    async def oauth_callback(
        state: str,
        code: str | None = None,
        error: str | None = None,
        iss: str | None = None,
        who: Caller = Depends(caller),
    ) -> Response:
        if error is not None or code is None:
            try:
                connections.cancel(who.identity.user_id, state)
            except tuple(CONNECT_ERRORS) as refused:
                raise HTTPException(CONNECT_ERRORS[type(refused)], str(refused)) from None
            raise HTTPException(400, "the server did not give access, so nothing was connected")
        try:
            server_name = await connections.complete(who.identity.user_id, state, code, iss)
        except tuple(CONNECT_ERRORS) as failure:
            raise HTTPException(CONNECT_ERRORS[type(failure)], str(failure)) from None
        except UPSTREAM_ERRORS:
            raise HTTPException(502, "the server did not accept the sign-in") from None
        audit.connection(who.identity, "connect", server_name)
        return RedirectResponse("/", status_code=303)

    @app.post("/servers/{name}/disconnect", status_code=204)
    async def disconnect(name: str, who: Caller = Depends(posting_caller)) -> Response:
        server = known_server(name, who.identity)
        connections.cancel_for(who.identity.user_id, server.name)
        await run_in_threadpool(tokens.delete_connection, who.identity.user_id, server.name)
        audit.connection(who.identity, "disconnect", server.name)
        return Response(status_code=204)

    @app.post("/ask")
    async def ask(body: AskBody, who: Caller = Depends(posting_caller)) -> AskOut:
        return await take_turn(who.identity, body.conversation, body.request, None)

    @app.post("/ask/confirm")
    async def confirm(body: ConfirmBody, who: Caller = Depends(posting_caller)) -> AskOut:
        return await take_turn(who.identity, body.conversation, "", body.decisions)

    async def take_turn(
        identity: Identity,
        conversation_id: UUID | None,
        text: str,
        decisions: dict[str, Decision] | None,
    ) -> AskOut:
        entry = "ask" if decisions is None else "confirm"
        started = time.monotonic()
        conversation = conversation_id or uuid4()

        def refuse(
            status: int, detail: str, outcome: str, headers: dict[str, str] | None = None
        ) -> NoReturn:
            audit.asked(identity, conversation_id, text, started, outcome, None, 0.0, entry=entry)
            raise HTTPException(status, detail, headers)

        if not quota.allows(identity.user_id):
            refuse(
                429,
                f"at most {settings.requests_per_minute} asks a minute",
                "rate_limited",
                {"Retry-After": "60"},
            )
        checkpointer: AsyncSqliteSaver = app.state.checkpointer
        thread = thread_id(identity.user_id, conversation)
        async with ask_locks.setdefault(identity.user_id, asyncio.Lock()):
            await refuse_deactivated(identity.user_id)
            history = await saved_messages(checkpointer, thread)
            if conversation_id is not None and not history:
                refuse(404, "no such conversation", "unknown_conversation")
            if decisions is not None:
                text = last_request(history)
            month = month_of(datetime.now(UTC))
            if (identity.user_id, month) not in spendings:
                spendings[identity.user_id, month] = Spending(
                    settings.price_model,
                    settings.budget_usd,
                    spending_store,
                    identity.user_id,
                    month,
                )
            spending = spendings[identity.user_id, month]
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
                            trace.tool_span if trace else no_tool_span,
                            jev_tool_selection(jev, settings.tool_selection, spending.record_jev),
                            tool_output_chars=settings.tool_output_chars,
                            history_tokens=settings.history_tokens,
                            listing_seconds=settings.listing_seconds,
                            confirm_writes=True,
                            decisions=decisions,
                            executed=executed,
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
                )
                raise HTTPException(504, detail) from None
            except ask_failures as error:
                cost = spending.cost_usd - spent_before
                raise upstream_failure(
                    identity, conversation, text, started, cost, error, entry, executed
                ) from None
            except BaseExceptionGroup as group:
                failed = group.subgroup(ask_failures)
                if failed is None:
                    raise
                cost = spending.cost_usd - spent_before
                raise upstream_failure(
                    identity, conversation, text, started, cost, first_leaf(failed), entry, executed
                ) from None
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
        )
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
    ) -> HTTPException:
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
        )
        return HTTPException(502, detail)

    return app
