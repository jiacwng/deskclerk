# The web front of the agent. An employee signs in with the company's OpenID Connect provider,
# connects their MCP servers and asks the agent. Browsers hold a signed session cookie; other
# clients send a bearer token. Every dependency is passed to create_app.

import asyncio
import base64
import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import anthropic
import httpx
import httpx2
import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from langchain_core.language_models import BaseChatModel
from mcp.client.auth import OAuthFlowError
from mcp.client.streamable_http import StreamableHTTPError
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from deskclerk import cookies, oidc
from deskclerk.agent import Gate, run_agent
from deskclerk.audit import Audit
from deskclerk.auth import AuthError, AuthUnavailable, Identity, SigningKeyLookup, authenticate
from deskclerk.config import Settings
from deskclerk.connections import (
    AlreadyConnected,
    FlowExpired,
    NoSignInNeeded,
    NotConnected,
    OAuthConnections,
    UnknownState,
    WrongUser,
)
from deskclerk.instructions import system_prompt
from deskclerk.providers import BudgetExceeded, Spending, price_of
from deskclerk.quota import Quota
from deskclerk.servers import McpServer
from deskclerk.tokens import TokenStore

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
UPSTREAM_ERRORS = (
    NotConnected,
    MCPError,
    StreamableHTTPError,
    OAuthFlowError,
    httpx2.HTTPError,
)
BROWSER_AUTH_REFUSED = "the login server's access token was refused"


class MissingSessionKey(RuntimeError):
    pass


class AskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)


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


class AskOut(BaseModel):
    answer: str
    executed: list[ExecutedOut]
    blocked: list[BlockedOut]
    not_connected: list[str]
    step_limit_hit: bool
    cut_off: bool
    cost_usd: float


@dataclass(frozen=True)
class Caller:
    identity: Identity
    by_cookie: bool


def _session_payload(identity: Identity) -> dict[str, Any]:
    return {
        "user": identity.user_id,
        "client": identity.client_id,
    }


def _token_expiry(token: str) -> int:
    return int(jwt.decode(token, options={"verify_signature": False})["exp"])


def _identity_from(payload: dict[str, Any]) -> Identity | None:
    try:
        return Identity(
            user_id=str(payload["user"]),
            client_id=str(payload["client"]),
        )
    except (KeyError, TypeError):
        return None


# The page is one file with one script and one style block, so the policy allows exactly those
# two blocks by hash and nothing else.
def _content_security_policy(page: str) -> str:
    sources = []
    for tag in ("script", "style"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", page, re.DOTALL)
        if match is None:
            raise ValueError(f"index.html has no <{tag}> block")
        digest = base64.b64encode(hashlib.sha256(match.group(1).encode()).digest()).decode()
        sources.append(f"{tag}-src 'sha256-{digest}'")
    fixed = ["default-src 'none'", "connect-src 'self'", "base-uri 'none'"]
    fixed += ["form-action 'self'", "frame-ancestors 'none'"]
    return "; ".join([*fixed, *sources])


def create_app(
    settings: Settings,
    servers: list[McpServer],
    connections: OAuthConnections,
    tokens: TokenStore,
    model: BaseChatModel,
    gate: Gate,
    signing_key: SigningKeyLookup,
    idp_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    if settings.session_key is None:
        raise MissingSessionKey("DESKCLERK_SESSION_KEY is not set; put it in .env")
    session_key = settings.session_key.get_secret_value().encode()
    secure_cookies = settings.sign_in_redirect_url.startswith("https://")
    session_seconds = settings.session_hours * SECONDS_PER_HOUR
    price = price_of(settings)
    quota = Quota(settings.requests_per_minute)
    audit = Audit(settings.audit_text, settings.audit_key)
    spendings: dict[str, Spending] = {}
    ask_locks: dict[str, asyncio.Lock] = {}
    discovered: dict[str, oidc.Endpoints] = {}
    model_errors: tuple[type[Exception], ...] = (anthropic.APIError, *UPSTREAM_ERRORS)
    server_by_name = {server.name: server for server in servers}
    page = resources.files("deskclerk").joinpath("static/index.html").read_text(encoding="utf-8")
    page_headers = {
        "Content-Security-Policy": _content_security_policy(page),
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }

    app = FastAPI(title="deskclerk", docs_url=None, redoc_url=None, openapi_url=None)

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
            return Caller(identity, by_cookie=False)
        payload = cookies.unsign(session_key, "session", request.cookies.get(SESSION_COOKIE, ""))
        session_identity = None if payload is None else _identity_from(payload)
        if session_identity is None:
            raise HTTPException(401, "sign in at /login", BEARER)
        return Caller(session_identity, by_cookie=True)

    async def posting_caller(request: Request, who: Caller = Depends(caller)) -> Caller:
        if who.by_cookie and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
            raise HTTPException(403, f"send the header {CSRF_HEADER}: {CSRF_VALUE}")
        return who

    def known_server(name: str) -> McpServer:
        server = server_by_name.get(name)
        if server is None:
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

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(page, headers=page_headers)

    @app.get("/login")
    async def login() -> Response:
        begun = oidc.start(settings, await endpoints())
        pending = {"state": begun.state, "verifier": begun.verifier}
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
            except AuthUnavailable:
                raise HTTPException(503, "the login server cannot be reached") from None
        try:
            identity = await run_in_threadpool(
                authenticate, token, settings, signing_key, settings.audience
            )
        except AuthError:
            raise HTTPException(401, BROWSER_AUTH_REFUSED) from None
        except AuthUnavailable:
            raise HTTPException(503, "the login server cannot be reached") from None
        lifetime = min(session_seconds, _token_expiry(token) - int(time.time()))
        if lifetime <= 0:
            raise HTTPException(401, BROWSER_AUTH_REFUSED)
        session = cookies.sign(session_key, "session", _session_payload(identity), lifetime)
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(LOGIN_COOKIE, path="/login")
        response.set_cookie(
            SESSION_COOKIE,
            session,
            lifetime,
            path="/",
            httponly=True,
            samesite="lax",
            secure=secure_cookies,
        )
        return response

    @app.post("/logout", status_code=204)
    async def logout(_: Caller = Depends(posting_caller)) -> Response:
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/me")
    async def me(who: Caller = Depends(caller)) -> dict[str, str]:
        return {"user": who.identity.user_id}

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
        ]

    @app.post("/servers/{name}/connect")
    async def connect(name: str, who: Caller = Depends(posting_caller)) -> ConnectOut:
        server = known_server(name)
        try:
            url = await connections.begin(who.identity.user_id, server)
        except tuple(CONNECT_ERRORS) as error:
            raise HTTPException(CONNECT_ERRORS[type(error)], str(error)) from None
        except UPSTREAM_ERRORS:
            raise HTTPException(502, f"cannot start the sign-in to {name}") from None
        return ConnectOut(authorization_url=url)

    @app.get("/oauth/callback")
    async def oauth_callback(
        state: str,
        code: str | None = None,
        error: str | None = None,
        who: Caller = Depends(caller),
    ) -> Response:
        if error is not None or code is None:
            try:
                connections.cancel(who.identity.user_id, state)
            except tuple(CONNECT_ERRORS) as refused:
                raise HTTPException(CONNECT_ERRORS[type(refused)], str(refused)) from None
            raise HTTPException(400, "the server did not give access, so nothing was connected")
        try:
            server_name = await connections.complete(who.identity.user_id, state, code)
        except tuple(CONNECT_ERRORS) as failure:
            raise HTTPException(CONNECT_ERRORS[type(failure)], str(failure)) from None
        except UPSTREAM_ERRORS:
            raise HTTPException(502, "the server did not accept the sign-in") from None
        audit.connection(who.identity, "connect", server_name)
        return RedirectResponse("/", status_code=303)

    @app.post("/servers/{name}/disconnect", status_code=204)
    async def disconnect(name: str, who: Caller = Depends(posting_caller)) -> Response:
        server = known_server(name)
        connections.cancel_for(who.identity.user_id, server.name)
        await run_in_threadpool(tokens.delete_tokens, who.identity.user_id, server.name)
        audit.connection(who.identity, "disconnect", server.name)
        return Response(status_code=204)

    @app.post("/ask")
    async def ask(body: AskBody, who: Caller = Depends(posting_caller)) -> AskOut:
        identity = who.identity
        started = time.monotonic()
        if not quota.allows(identity.user_id):
            audit.asked(identity, body.request, started, "rate_limited", None, 0.0)
            raise HTTPException(
                429, f"at most {settings.requests_per_minute} asks a minute", {"Retry-After": "60"}
            )
        async with ask_locks.setdefault(identity.user_id, asyncio.Lock()):
            spending = spendings.setdefault(identity.user_id, Spending(price, settings.budget_usd))
            try:
                spending.check()
            except BudgetExceeded as error:
                audit.asked(identity, body.request, started, "over_budget", None, 0.0)
                raise HTTPException(402, str(error)) from None
            spent_before = spending.cost_usd
            try:
                result = await run_agent(
                    identity.user_id,
                    body.request,
                    system_prompt(datetime.now(UTC).date()),
                    servers,
                    connections,
                    model,
                    gate,
                    settings.step_limit,
                    spending.record,
                )
            except model_errors:
                cost = spending.cost_usd - spent_before
                raise upstream_failure(identity, body.request, started, cost) from None
            except BaseExceptionGroup as group:
                if group.subgroup(model_errors) is None:
                    raise
                cost = spending.cost_usd - spent_before
                raise upstream_failure(identity, body.request, started, cost) from None
        cost = price.cost(result.total_usage)
        audit.asked(identity, body.request, started, "answered", result, cost)
        return AskOut(
            answer=result.answer,
            executed=[ExecutedOut(**vars(call)) for call in result.executed],
            blocked=[BlockedOut(**vars(call)) for call in result.blocked],
            not_connected=list(result.not_connected),
            step_limit_hit=result.step_limit_hit,
            cut_off=result.cut_off,
            cost_usd=cost,
        )

    def upstream_failure(
        identity: Identity, request: str, started: float, cost: float
    ) -> HTTPException:
        audit.asked(identity, request, started, "upstream_error", None, cost)
        return HTTPException(502, "the model or an MCP server did not answer")

    return app
