# What every route group shares: the objects create_app builds once, the signed-in caller checks
# that guard the routes, and the cookie and header names the browser page and the tests use.

from dataclasses import dataclass, field

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from langchain_core.language_models import BaseChatModel
from starlette.concurrency import run_in_threadpool

from deskclerk import oidc
from deskclerk.agent import Gate
from deskclerk.audit import Audit
from deskclerk.auditstore import AuditStore
from deskclerk.auth import (
    AuthError,
    AuthUnavailable,
    Identity,
    SigningKeyLookup,
    authenticate,
)
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.database import Database
from deskclerk.employees import EmployeeStore
from deskclerk.jev import Jev
from deskclerk.metrics import Metrics
from deskclerk.quota import Quota
from deskclerk.servers import McpServer
from deskclerk.serverstore import ServerCatalog
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore

SESSION_COOKIE = "deskclerk_session"
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "deskclerk"
BEARER = {"WWW-Authenticate": "Bearer"}
MAX_REASON_CHARS = 200


def clip(text: str | None) -> str:
    return repr((text or "")[:MAX_REASON_CHARS])


def link_failure(code: str) -> RedirectResponse:
    return RedirectResponse(f"/?error={code}", status_code=303)


@dataclass(frozen=True)
class Caller:
    identity: Identity
    by_cookie: bool


@dataclass(frozen=True)
class Services:
    settings: Settings
    database: Database
    catalog: ServerCatalog
    connections: OAuthConnections
    tokens: TokenStore
    employees: EmployeeStore
    spending_store: SpendingStore
    sessions: SessionStore
    model: BaseChatModel
    gate: Gate
    jev: Jev | None
    audit: Audit
    audit_store: AuditStore
    metrics: Metrics
    quota: Quota
    signing_key: SigningKeyLookup
    session_key: bytes
    secure_cookies: bool
    session_seconds: int
    idp_transport: httpx.AsyncBaseTransport | None
    discovered: dict[str, oidc.Endpoints] = field(default_factory=dict)

    async def caller(self, request: Request) -> Caller:
        header = request.headers.get("authorization")
        if header is not None:
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise HTTPException(401, "send Authorization: Bearer <access token>", BEARER)
            try:
                identity = await run_in_threadpool(
                    authenticate, token, self.settings, self.signing_key, self.settings.audience
                )
            except AuthError as error:
                raise HTTPException(401, str(error), BEARER) from None
            except AuthUnavailable:
                raise HTTPException(503, "the login server cannot be reached") from None
            if not self.employees.recorded(identity):
                await run_in_threadpool(self.employees.record, identity)
            return Caller(identity, by_cookie=False)
        session_identity = await run_in_threadpool(
            self.sessions.lookup, request.cookies.get(SESSION_COOKIE, "")
        )
        if session_identity is None:
            raise HTTPException(401, "sign in at /login", BEARER)
        return Caller(session_identity, by_cookie=True)

    async def posting_caller(self, request: Request) -> Caller:
        who = await self.caller(request)
        if who.by_cookie and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
            raise HTTPException(403, f"send the header {CSRF_HEADER}: {CSRF_VALUE}")
        return who

    def is_admin(self, identity: Identity) -> bool:
        group = self.settings.admin_group
        return bool(group) and group in identity.groups

    async def admin_caller(self, request: Request) -> Caller:
        who = await self.caller(request)
        if not self.is_admin(who.identity):
            raise HTTPException(403, "this page is for the admins group")
        return who

    async def admin_posting_caller(self, request: Request) -> Caller:
        who = await self.posting_caller(request)
        if not self.is_admin(who.identity):
            raise HTTPException(403, "this page is for the admins group")
        return who

    def known_server(self, name: str, identity: Identity) -> McpServer:
        for server in self.catalog.offered:
            if server.name == name and server.open_to(identity.groups):
                return server
        raise HTTPException(404, f"no server named {name}")

    def idp_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.idp_transport, timeout=oidc.TIMEOUT_SECONDS)

    async def endpoints(self) -> oidc.Endpoints:
        if "endpoints" not in self.discovered:
            async with self.idp_client() as client:
                self.discovered["endpoints"] = await oidc.discover(self.settings, client)
        return self.discovered["endpoints"]
