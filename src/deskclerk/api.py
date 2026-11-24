# The web front of the agent. An employee signs in with the company's OpenID Connect provider,
# connects their MCP servers and asks the agent, which waits for the employee's confirmation before
# a call that changes something. Browsers hold a session cookie that names a row in the session
# table; other clients send a bearer token. Every dependency is passed to create_app, which builds
# the shared objects once and includes one router per group of routes from deskclerk.routes.

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from prometheus_client import CollectorRegistry

from deskclerk.agent import Gate
from deskclerk.audit import Audit
from deskclerk.auditstore import AuditStore, chain_key_of
from deskclerk.auth import SigningKeyLookup
from deskclerk.bodylimit import LimitRequestSize
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.conversations import open_checkpointer
from deskclerk.database import Database
from deskclerk.employees import EmployeeStore
from deskclerk.jev import Jev
from deskclerk.metrics import Metrics
from deskclerk.providers import Usage, cost_of
from deskclerk.quota import Quota
from deskclerk.retention import (
    delete_old_audit_rows,
    delete_old_audit_rows_daily,
    delete_old_threads,
    delete_old_threads_daily,
)
from deskclerk.routes import admin, conversations, health, servers, signin
from deskclerk.routes.conversations import ExecutedOut, FailedTurn
from deskclerk.routes.services import Services
from deskclerk.serverstore import ServerCatalog
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore

SECONDS_PER_HOUR = 3600


class MissingSessionKey(RuntimeError):
    pass


def create_app(
    settings: Settings,
    database: Database,
    catalog: ServerCatalog,
    connections: OAuthConnections,
    tokens: TokenStore,
    spending_store: SpendingStore,
    sessions: SessionStore,
    model: BaseChatModel,
    gate: Gate,
    jev: Jev | None,
    signing_key: SigningKeyLookup,
    registry: CollectorRegistry,
    idp_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    if settings.session_key is None:
        raise MissingSessionKey("DESKCLERK_SESSION_KEY is not set; put it in .env")
    cost_of(settings.price_model, Usage(0, 0, 0, 0))
    audit_store = AuditStore(database, chain_key_of(settings))
    services = Services(
        settings=settings,
        database=database,
        catalog=catalog,
        connections=connections,
        tokens=tokens,
        employees=EmployeeStore(database),
        spending_store=spending_store,
        sessions=sessions,
        model=model,
        gate=gate,
        jev=jev,
        audit=Audit(settings.audit_text, settings.audit_key, audit_store),
        audit_store=audit_store,
        metrics=Metrics(registry),
        quota=Quota(database, settings.requests_per_minute),
        signing_key=signing_key,
        session_key=settings.session_key.get_secret_value().encode(),
        secure_cookies=settings.sign_in_redirect_url.startswith("https://"),
        session_seconds=settings.session_hours * SECONDS_PER_HOUR,
        idp_transport=idp_transport,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with (
            database.serving(),
            connections.listening(),
            catalog.listening(),
        ):
            checkpointer = await open_checkpointer(database, tokens.fernet)
            app.state.checkpointer = checkpointer
            sweeps: list[asyncio.Task[None]] = []
            if settings.retention_days > 0:
                await delete_old_threads(database, checkpointer, settings.retention_days)
                await delete_old_audit_rows(audit_store, settings.retention_days)
                sweeps = [
                    asyncio.create_task(
                        delete_old_threads_daily(database, checkpointer, settings.retention_days)
                    ),
                    asyncio.create_task(
                        delete_old_audit_rows_daily(audit_store, settings.retention_days)
                    ),
                ]
            yield
            for sweep in sweeps:
                sweep.cancel()

    app = FastAPI(
        title="deskclerk", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    app.add_middleware(LimitRequestSize)

    @app.exception_handler(FailedTurn)
    async def failed_turn(_: Request, failure: FailedTurn) -> JSONResponse:
        calls = [ExecutedOut(**vars(call)).model_dump() for call in failure.executed]
        return JSONResponse({"detail": failure.detail, "calls": calls}, status_code=failure.status)

    app.include_router(health.router(services))
    if settings.admin_group:
        app.include_router(admin.router(services))
    app.include_router(signin.router(services))
    app.include_router(servers.router(services))
    app.include_router(conversations.router(services))
    return app
