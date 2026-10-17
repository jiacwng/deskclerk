# The HTTP API. Every request is answered as the employee named in its bearer token, and nothing
# in the request body can change who that is.
# Run locally: docker compose up -d --wait, load the data with needtoknow.ingest, then
# .venv/bin/uvicorn --app-dir src --factory needtoknow.api:create_default_app

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated

import jwt
import psycopg
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from needtoknow import agent, db, retrieval
from needtoknow.audit import Audit, log_to_stdout
from needtoknow.auth import (
    JWKS_TIMEOUT_SECONDS,
    AuthError,
    AuthUnavailable,
    Identity,
    SigningKeyLookup,
    authenticate,
    jwks_signing_key,
)
from needtoknow.config import Settings, load_settings
from needtoknow.embed import embed_query
from needtoknow.exposure import check_exposure
from needtoknow.graph import GraphGroups, graph_from_settings
from needtoknow.providers import BudgetExceeded, Spending, chat_model, price_of
from needtoknow.quota import Quota
from needtoknow.schema import check_schema
from needtoknow.screen import Screen, screen_from_settings

READY_TIMEOUT_SECONDS = 5

_bearer = HTTPBearer(auto_error=False)
_CHAT_PAGE = files("needtoknow").joinpath("static/chat.html").read_text(encoding="utf-8")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=retrieval.MAX_INPUT_CHARS)
    k: int = Field(default=5, ge=1, le=retrieval.MAX_RESULTS)


class SearchResponse(BaseModel):
    user: str
    results: list[retrieval.Hit]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=retrieval.MAX_INPUT_CHARS)


class AskResponse(BaseModel):
    user: str
    answer: str
    citations: list[str]
    refused: bool
    queries: list[agent.Query]
    withheld: int


def create_app(
    settings: Settings,
    signing_key: SigningKeyLookup,
    model: BaseChatModel,
    screen: Screen | None = None,
    graph: GraphGroups | None = None,
) -> FastAPI:
    check_exposure(settings)
    check_schema(settings)
    pool = db.app_pool(settings, settings.pool_size)
    quota = Quota(settings.requests_per_minute)
    audit = Audit(settings.audit_text)
    # The key set is cached for five minutes, so a readiness probe rarely calls the login server.
    signing_keys = jwt.PyJWKClient(settings.jwks_url, timeout=JWKS_TIMEOUT_SECONDS)

    # The first query would otherwise wait several seconds for the embedding model to load.
    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        embed_query("warm up")
        yield
        pool.close()

    api = FastAPI(title="needtoknow", lifespan=lifespan)
    spending = Spending(price_of(settings), settings.budget_usd)

    def current_identity(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Identity:
        if credentials is None:
            raise _unauthorized(None)
        try:
            identity = authenticate(
                credentials.credentials, settings, signing_key, settings.audience, graph
            )
        except AuthError:
            raise _unauthorized("invalid_token") from None
        except AuthUnavailable:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the login server cannot be reached",
            ) from None
        if not quota.allows(identity.user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="too many requests, wait a minute",
            )
        return identity

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/ready")
    def ready() -> JSONResponse:
        checks = {"database": "ok", "signing_keys": "ok"}
        try:
            with pool.connection(timeout=READY_TIMEOUT_SECONDS) as connection:
                connection.execute("SELECT 1")
        except psycopg.OperationalError:
            checks["database"] = "unreachable"
        try:
            signing_keys.get_jwk_set()
        except jwt.PyJWTError:
            checks["signing_keys"] = "unreachable"
        ready = all(check == "ok" for check in checks.values())
        code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(checks, status_code=code)

    @api.get("/", response_class=HTMLResponse)
    def chat_page() -> str:
        return _CHAT_PAGE

    @api.get("/config")
    def config() -> dict[str, str]:
        return {
            "issuer": settings.issuer,
            "client_id": settings.client_id,
            "scope": settings.login_scope,
        }

    # A plain def runs in FastAPI's thread pool, so the blocking database call and the embedding
    # do not stall other requests. A search, and each tool call of an answer, borrows a
    # connection from the pool for its own transaction.
    @api.post("/search")
    def search(
        request: SearchRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> SearchResponse:
        started = time.monotonic()
        with pool.connection() as connection:
            hits = retrieval.search(connection, identity.principals, request.query, request.k)
        documents = [hit.doc_id for hit in hits]
        audit.served("search", identity, request.query, started, documents=documents)
        return SearchResponse(user=identity.user_id, results=hits)

    @api.post("/ask")
    def ask(
        request: AskRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> AskResponse:
        started = time.monotonic()
        try:
            answer = agent.ask(
                request.question, pool, identity.principals, model, spending, screen=screen
            )
        except BudgetExceeded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the model budget for this server is spent",
            ) from None
        except psycopg.OperationalError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the database cannot be reached",
            ) from None
        audit.answered(identity, request.question, answer, started)
        return AskResponse(
            user=identity.user_id,
            answer=answer.text,
            citations=list(answer.citations),
            refused=answer.refused,
            queries=list(answer.queries),
            withheld=answer.withheld,
        )

    return api


def create_default_app() -> FastAPI:
    log_to_stdout()
    settings = load_settings()
    screen = screen_from_settings(settings)
    graph = graph_from_settings(settings)
    return create_app(settings, jwks_signing_key(settings), chat_model(settings), screen, graph)


def _unauthorized(error: str | None) -> HTTPException:
    # RFC 6750: a request without a token gets a bare challenge, a rejected token names the error.
    challenge = "Bearer" if error is None else f'Bearer error="{error}"'
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing bearer token" if error is None else "invalid bearer token",
        headers={"WWW-Authenticate": challenge},
    )
