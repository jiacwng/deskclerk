# The HTTP API. Every request is answered as the employee named in its bearer token, and nothing
# in the request body can change who that is.
# Run locally: docker compose up -d --wait, load the data with needtoknow.ingest, then
# .venv/bin/uvicorn --app-dir src --factory needtoknow.api:create_default_app

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from needtoknow import agent, db, retrieval
from needtoknow.auth import (
    AuthError,
    AuthUnavailable,
    Identity,
    SigningKeyLookup,
    authenticate,
    jwks_signing_key,
)
from needtoknow.config import Settings, load_settings
from needtoknow.embed import embed_query
from needtoknow.providers import BudgetExceeded, Spending, chat_model

_bearer = HTTPBearer(auto_error=False)
_CHAT_PAGE = files("needtoknow").joinpath("static/chat.html").read_text(encoding="utf-8")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=retrieval.MAX_RESULTS)


class SearchResponse(BaseModel):
    user: str
    results: list[retrieval.Hit]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    user: str
    answer: str
    citations: list[str]
    refused: bool
    queries: list[agent.Query]


def create_app(settings: Settings, signing_key: SigningKeyLookup, model: BaseChatModel) -> FastAPI:
    api = FastAPI(title="needtoknow", lifespan=_load_embedding_model)
    spending = Spending(settings.model, settings.budget_usd)

    def current_identity(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Identity:
        if credentials is None:
            raise _unauthorized(None)
        try:
            return authenticate(credentials.credentials, settings, signing_key)
        except AuthError:
            raise _unauthorized("invalid_token") from None
        except AuthUnavailable:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the login server cannot be reached",
            ) from None

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/", response_class=HTMLResponse)
    def chat_page() -> str:
        return _CHAT_PAGE

    @api.get("/config")
    def config() -> dict[str, str]:
        return {"issuer": settings.issuer, "client_id": settings.client_id}

    # A plain def runs in FastAPI's thread pool, so the blocking database call and the embedding
    # do not stall other requests. Each request opens its own connection.
    @api.post("/search")
    def search(
        request: SearchRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> SearchResponse:
        with db.connect_app(settings) as connection:
            hits = retrieval.search(connection, identity.principals, request.query, request.k)
        return SearchResponse(user=identity.user_id, results=hits)

    @api.post("/ask")
    def ask(
        request: AskRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> AskResponse:
        try:
            with db.connect_app(settings) as connection:
                answer = agent.ask(
                    request.question, connection, identity.principals, model, spending
                )
        except BudgetExceeded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the model budget for this server is spent",
            ) from None
        return AskResponse(
            user=identity.user_id,
            answer=answer.text,
            citations=list(answer.citations),
            refused=answer.refused,
            queries=list(answer.queries),
        )

    return api


def create_default_app() -> FastAPI:
    settings = load_settings()
    return create_app(settings, jwks_signing_key(settings), chat_model(settings))


# The first query would otherwise wait several seconds for the embedding model to load.
@asynccontextmanager
async def _load_embedding_model(api: FastAPI) -> AsyncIterator[None]:
    embed_query("warm up")
    yield


def _unauthorized(error: str | None) -> HTTPException:
    # RFC 6750: a request without a token gets a bare challenge, a rejected token names the error.
    challenge = "Bearer" if error is None else f'Bearer error="{error}"'
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing bearer token" if error is None else "invalid bearer token",
        headers={"WWW-Authenticate": challenge},
    )
