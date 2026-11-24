# The routes that list, read and delete an employee's conversations and the ones that ask the agent:
# /ask and /ask/confirm answer as one JSON body or, when the client accepts it, as a stream of
# server-sent events. A turn is checked against the quota and the budget, run by the agent, then
# recorded in the audit log, the metrics and the spending store.

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, NoReturn
from uuid import UUID, uuid4

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.sse import KEEPALIVE_COMMENT, format_sse_event
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from deskclerk.agent import (
    AlwaysAllowRefused,
    CallEnded,
    CallStarted,
    ConversationThread,
    ConversationWaiting,
    Decision,
    Edit,
    ExecutedCall,
    Limits,
    McpServerFailed,
    NothingToConfirm,
    RunEvent,
    ToolAccess,
    WrongDecisions,
    no_tool_span,
    run_agent,
    waiting_calls,
)
from deskclerk.audit import FinishedTurn
from deskclerk.auth import Identity
from deskclerk.connections import UPSTREAM_ERRORS, first_leaf
from deskclerk.conversations import (
    NoCheckpointBefore,
    delete_conversation,
    delete_dropped_turns,
    fork_before_request,
    last_request,
    list_conversations,
    past_turns,
    record_turn,
    saved_messages,
    thread_id,
)
from deskclerk.instructions import system_prompt
from deskclerk.providers import BudgetExceeded, model_errors
from deskclerk.routes.services import Caller, Services
from deskclerk.screen import jev_screen
from deskclerk.spending import Spending, month_of

MAX_REQUEST_CHARS = 4000
STREAM_PING_SECONDS = 15.0
MODEL_ERRORS = model_errors()

_log = logging.getLogger(__name__)


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


class ExecutedOut(BaseModel):
    server: str
    tool: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    flagged: bool


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
    trusted: bool
    after_flag: bool


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
    flagged: bool


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
                flagged=False,
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
                flagged=event.flagged,
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


def _upstream_detail(error: BaseException) -> str:
    if isinstance(error, McpServerFailed):
        return f"the MCP server {error.server_name} did not answer ({error.cause})"
    if isinstance(error, MODEL_ERRORS):
        return f"the model did not answer ({type(error).__name__})"
    return f"an MCP server did not answer ({type(error).__name__})"


def router(services: Services) -> APIRouter:
    routes = APIRouter()
    settings = services.settings
    database = services.database
    catalog = services.catalog
    tokens = services.tokens
    connections = services.connections
    audit = services.audit
    metrics = services.metrics
    spending_store = services.spending_store
    ask_failures: tuple[type[Exception], ...] = (
        McpServerFailed,
        *MODEL_ERRORS,
        *UPSTREAM_ERRORS,
    )

    @routes.get("/conversations")
    async def conversation_list(
        who: Caller = Depends(services.caller),
    ) -> list[ConversationSummaryOut]:
        found = await list_conversations(database, who.identity.user_id)
        return [ConversationSummaryOut(**vars(summary)) for summary in found]

    @routes.delete("/conversations/{conversation_id}", status_code=204)
    async def remove_conversation(
        request: Request, conversation_id: UUID, who: Caller = Depends(services.posting_caller)
    ) -> Response:
        checkpointer: AsyncPostgresSaver = request.app.state.checkpointer
        user_id = who.identity.user_id
        async with database.user_lock(user_id):
            if not await saved_messages(checkpointer, thread_id(user_id, conversation_id)):
                raise HTTPException(404, "no such conversation")
            await delete_conversation(database, checkpointer, user_id, conversation_id)
        return Response(status_code=204)

    @routes.get("/conversations/{conversation_id}")
    async def conversation_turns(
        request: Request, conversation_id: UUID, who: Caller = Depends(services.caller)
    ) -> ConversationOut:
        checkpointer: AsyncPostgresSaver = request.app.state.checkpointer
        thread = thread_id(who.identity.user_id, conversation_id)
        history = await saved_messages(checkpointer, thread)
        if not history:
            raise HTTPException(404, "no such conversation")
        waiting = await waiting_calls(
            who.identity.user_id,
            [s for s in catalog.offered if s.open_to(who.identity.groups)],
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

    @routes.post("/ask", response_model=AskOut)
    async def ask(
        request: Request, body: AskBody, who: Caller = Depends(services.posting_caller)
    ) -> AskOut | StreamingResponse:
        return await answer_as_asked(
            request, who.identity, body.conversation, body.request, None, body.edit_of
        )

    @routes.post("/ask/confirm", response_model=AskOut)
    async def confirm(
        request: Request, body: ConfirmBody, who: Caller = Depends(services.posting_caller)
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
                request,
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
                    request,
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
        request: Request,
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

        async def refuse(
            status: int, detail: str, outcome: str, headers: dict[str, str] | None = None
        ) -> NoReturn:
            await run_in_threadpool(
                audit.asked,
                FinishedTurn(
                    identity,
                    conversation_id,
                    text,
                    started,
                    outcome,
                    None,
                    0.0,
                    entry=entry,
                    edit_of=edit_of,
                ),
            )
            metrics.ask_finished(outcome, started, 0.0, (), None)
            raise HTTPException(status, detail, headers)

        if not await services.quota.allows(identity.user_id):
            await refuse(
                429,
                f"at most {settings.requests_per_minute} asks a minute",
                "rate_limited",
                {"Retry-After": "60"},
            )
        checkpointer: AsyncPostgresSaver = request.app.state.checkpointer
        thread = thread_id(identity.user_id, conversation)
        async with database.user_lock(identity.user_id):
            history = await saved_messages(checkpointer, thread)
            if (conversation_id is not None or edit_of is not None) and not history:
                await refuse(404, "no such conversation", "unknown_conversation")
            fork_from = None
            if edit_of is not None:
                if edit_of >= len(past_turns(history)):
                    await refuse(422, "no such request to edit", "wrong_edit")
                try:
                    fork = await fork_before_request(checkpointer, thread, edit_of)
                except NoCheckpointBefore:
                    await refuse(422, "that request can no longer be edited", "wrong_edit")
                fork_from, history = fork.checkpoint_id, fork.messages
            if decisions is not None:
                text = last_request(history)
            month = month_of(datetime.now(UTC))
            spending = Spending(
                settings.price_model, settings.budget_usd, spending_store, identity.user_id, month
            )
            try:
                await run_in_threadpool(spending.check)
            except BudgetExceeded as error:
                await refuse(402, str(error), "over_budget")
            spent_before = await run_in_threadpool(spending.spent_usd)
            executed: list[ExecutedCall] = []
            try:
                async with asyncio.timeout(settings.ask_seconds):
                    result = await run_agent(
                        identity.user_id,
                        text,
                        system_prompt(datetime.now(UTC).date()),
                        ToolAccess(
                            [s for s in catalog.offered if s.open_to(identity.groups)],
                            connections,
                            tokens,
                        ),
                        ConversationThread(checkpointer, thread, history, fork_from),
                        services.model,
                        services.gate,
                        jev_screen(services.jev, spending.record_jev),
                        Limits(
                            settings.step_limit,
                            settings.tool_output_chars,
                            settings.history_tokens,
                            settings.listing_seconds,
                        ),
                        spending.record,
                        metrics.timed(no_tool_span),
                        decisions=decisions,
                        executed=executed,
                        emit=lambda event: emit(_stream_event(event)),
                        running=running,
                    )
            except ConversationWaiting:
                await refuse(
                    409, "this conversation waits for a confirmation", "conversation_waiting"
                )
            except NothingToConfirm:
                await refuse(409, "this conversation has nothing to confirm", "nothing_pending")
            except WrongDecisions:
                await refuse(422, "decide every pending call exactly once", "wrong_decisions")
            except AlwaysAllowRefused:
                await refuse(
                    422,
                    "only a call that does not delete or overwrite can be always allowed",
                    "wrong_decisions",
                )
            except asyncio.CancelledError:
                cost = await run_in_threadpool(spending.spent_usd) - spent_before
                metrics.ask_finished("cancelled", started, cost, executed, None)
                await run_in_threadpool(
                    audit.asked,
                    FinishedTurn(
                        identity,
                        conversation,
                        text,
                        started,
                        "cancelled",
                        None,
                        cost,
                        "the client stopped the request",
                        entry,
                        executed,
                        edit_of,
                    ),
                )
                await record_turn(database, identity.user_id, conversation, text, edit_of == 0)
                raise
            except TimeoutError:
                cost = await run_in_threadpool(spending.spent_usd) - spent_before
                detail = (
                    f"the ask took longer than {settings.ask_seconds} seconds "
                    "(set DESKCLERK_ASK_SECONDS to change it)"
                )
                await run_in_threadpool(
                    audit.asked,
                    FinishedTurn(
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
                    ),
                )
                metrics.ask_finished("timed_out", started, cost, executed, None)
                raise FailedTurn(504, detail, executed) from None
            except ask_failures as error:
                cost = await run_in_threadpool(spending.spent_usd) - spent_before
                raise await upstream_failure(
                    identity, conversation, text, started, cost, error, entry, executed, edit_of
                ) from None
            except BaseExceptionGroup as group:
                failed = group.subgroup(ask_failures)
                if failed is None:
                    if isinstance(group, Exception):
                        cost = await run_in_threadpool(spending.spent_usd) - spent_before
                        await unexpected_failure(
                            identity, conversation, text, started, cost, group, entry, executed,
                            edit_of,
                        )  # fmt: skip
                    raise
                cost = await run_in_threadpool(spending.spent_usd) - spent_before
                raise await upstream_failure(
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
            except Exception as error:
                cost = await run_in_threadpool(spending.spent_usd) - spent_before
                await unexpected_failure(
                    identity, conversation, text, started, cost, error, entry, executed, edit_of
                )
                raise
            await delete_dropped_turns(database, thread)
            await record_turn(database, identity.user_id, conversation, text, edit_of == 0)
        cost = await run_in_threadpool(spending.spent_usd) - spent_before
        outcome = "awaiting_confirmation" if result.pending else "answered"
        await run_in_threadpool(
            audit.asked,
            FinishedTurn(
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
            ),
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

    async def unexpected_failure(
        identity: Identity,
        conversation: UUID | None,
        request: str,
        started: float,
        cost: float,
        error: BaseException,
        entry: str,
        executed: list[ExecutedCall],
        edit_of: int | None,
    ) -> None:
        await run_in_threadpool(
            audit.asked,
            FinishedTurn(
                identity,
                conversation,
                request,
                started,
                "error",
                None,
                cost,
                type(error).__name__,
                entry,
                executed,
                edit_of,
            ),
        )
        metrics.ask_finished("error", started, cost, executed, None)

    async def upstream_failure(
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
        await run_in_threadpool(
            audit.asked,
            FinishedTurn(
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
            ),
        )
        metrics.ask_finished("upstream_error", started, cost, executed, None)
        return FailedTurn(502, detail, executed)

    return routes
