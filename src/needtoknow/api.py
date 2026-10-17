# The HTTP API. Every request is answered as the employee named in its bearer token, and nothing
# in the request body can change who that is.
# Run locally: docker compose up -d --wait, load the data with needtoknow.ingest, then
# .venv/bin/uvicorn --app-dir src --factory needtoknow.api:create_default_app

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from needtoknow.exposure import check_exposure
from needtoknow.providers import BudgetExceeded, Spending, chat_model, price_of
from needtoknow.quota import Quota
from needtoknow.screen import Screen, screen_from_settings

_bearer = HTTPBearer(auto_error=False)
BUDGET_SPENT = "the model budget for this server is spent"
DATABASE_DOWN = "the database cannot be reached"
# The client sends the conversation so far, so the server stores nothing between requests.
MAX_HISTORY_TURNS = 10
MAX_HISTORY_CHARS = MAX_HISTORY_TURNS * retrieval.MAX_INPUT_CHARS
_CHAT_PAGE = files("needtoknow").joinpath("static/chat.html").read_text(encoding="utf-8")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=retrieval.MAX_INPUT_CHARS)
    k: int = Field(default=5, ge=1, le=retrieval.MAX_RESULTS)


class SearchResponse(BaseModel):
    user: str
    results: list[retrieval.Hit]


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=retrieval.MAX_INPUT_CHARS)
    answer: str = Field(min_length=1)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=retrieval.MAX_INPUT_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @model_validator(mode="after")
    def _history_fits(self) -> "AskRequest":
        chars = 0
        for turn in self.history:
            chars += len(turn.question) + len(turn.answer)
        if chars > MAX_HISTORY_CHARS:
            raise ValueError(f"the history holds more than {MAX_HISTORY_CHARS} characters")
        return self

    def turns(self) -> list[agent.Turn]:
        return [agent.Turn(turn.question, turn.answer) for turn in self.history]


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
) -> FastAPI:
    check_exposure(settings)
    pool = db.app_pool(settings, settings.pool_size)
    quota = Quota(settings.requests_per_minute)

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
                credentials.credentials, settings, signing_key, settings.client_id
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

    @api.get("/", response_class=HTMLResponse)
    def chat_page() -> str:
        return _CHAT_PAGE

    @api.get("/config")
    def config() -> dict[str, str]:
        return {"issuer": settings.issuer, "client_id": settings.client_id}

    # A plain def runs in FastAPI's thread pool, so the blocking database call and the embedding
    # do not stall other requests. A search, and each tool call of an answer, borrows a
    # connection from the pool for its own transaction.
    @api.post("/search")
    def search(
        request: SearchRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> SearchResponse:
        with pool.connection() as connection:
            hits = retrieval.search(connection, identity.principals, request.query, request.k)
        return SearchResponse(user=identity.user_id, results=hits)

    @api.post("/ask")
    def ask(
        request: AskRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> AskResponse:
        try:
            answer = agent.ask(
                request.question,
                pool,
                identity.principals,
                model,
                spending,
                screen=screen,
                history=request.turns(),
            )
        except BudgetExceeded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=BUDGET_SPENT
            ) from None
        except psycopg.OperationalError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=DATABASE_DOWN
            ) from None
        return _ask_response(identity, answer)

    # The status is sent before the first event, so a failure after it arrives as an error event.
    @api.post("/ask/stream", response_class=EventSourceResponse)
    def ask_stream(
        request: AskRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> Iterator[ServerSentEvent]:
        events = agent.ask_stream(
            request.question,
            pool,
            identity.principals,
            model,
            spending,
            screen=screen,
            history=request.turns(),
        )
        try:
            for event in events:
                if isinstance(event, agent.ToolCall):
                    yield ServerSentEvent(event="tool", data=event)
                elif isinstance(event, agent.Token):
                    yield ServerSentEvent(event="token", data=event.text)
                else:
                    yield ServerSentEvent(event="answer", data=_ask_response(identity, event))
        except BudgetExceeded:
            yield ServerSentEvent(event="error", data={"detail": BUDGET_SPENT})
        except psycopg.OperationalError:
            yield ServerSentEvent(event="error", data={"detail": DATABASE_DOWN})

    return api


def create_default_app() -> FastAPI:
    settings = load_settings()
    screen = screen_from_settings(settings)
    return create_app(settings, jwks_signing_key(settings), chat_model(settings), screen)


def _ask_response(identity: Identity, answer: agent.Answer) -> AskResponse:
    return AskResponse(
        user=identity.user_id,
        answer=answer.text,
        citations=list(answer.citations),
        refused=answer.refused,
        queries=list(answer.queries),
        withheld=answer.withheld,
    )


def _unauthorized(error: str | None) -> HTTPException:
    # RFC 6750: a request without a token gets a bare challenge, a rejected token names the error.
    challenge = "Bearer" if error is None else f'Bearer error="{error}"'
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing bearer token" if error is None else "invalid bearer token",
        headers={"WWW-Authenticate": challenge},
    )
