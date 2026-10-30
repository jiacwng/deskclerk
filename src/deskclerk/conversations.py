# Conversation memory: each conversation is a LangGraph thread whose checkpoints live in
# Postgres, encrypted with the token key. A thread id starts with the employee's id, so one
# employee can never load another's. Erasing an employee deletes every thread with their prefix.
# A conversations table beside them holds each conversation's title and last activity for the list.

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

from deskclerk.database import Database

CIPHER_NAME = "fernet"
UUID_LENGTH = 36
TITLE_CHARS = 60
LIST_LIMIT = 100


class FernetCipher:
    def __init__(self, fernet: Fernet) -> None:
        self._fernet = fernet

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return CIPHER_NAME, self._fernet.encrypt(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != CIPHER_NAME:
            raise ValueError(f"unknown cipher {ciphername}")
        return self._fernet.decrypt(ciphertext)


async def open_checkpointer(database: Database, fernet: Fernet) -> AsyncPostgresSaver:
    checkpointer = AsyncPostgresSaver(
        database.async_pool, serde=EncryptedSerializer(FernetCipher(fernet))
    )
    async with database.schema_lock():
        await checkpointer.setup()
    await asyncio.to_thread(
        database.create_tables,
        "CREATE TABLE IF NOT EXISTS conversations ("
        "user_id TEXT NOT NULL, id UUID NOT NULL, title TEXT NOT NULL, "
        "updated_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (user_id, id))",
    )
    return checkpointer


@dataclass(frozen=True)
class ConversationSummary:
    id: UUID
    title: str
    updated_at: datetime


def title_of(request: str) -> str:
    collapsed = " ".join(request.split())
    if len(collapsed) <= TITLE_CHARS:
        return collapsed
    return collapsed[:TITLE_CHARS] + "…"


async def record_turn(
    database: Database, user_id: str, conversation: UUID, request: str, retitle: bool = False
) -> None:
    async with database.async_pool.connection() as connection:
        await connection.execute(
            "INSERT INTO conversations (user_id, id, title, updated_at) VALUES (%s, %s, %s, now()) "
            "ON CONFLICT (user_id, id) DO UPDATE SET updated_at = now(), "
            "title = CASE WHEN %s THEN excluded.title ELSE conversations.title END",
            (user_id, conversation, title_of(request), retitle),
        )


async def list_conversations(database: Database, user_id: str) -> list[ConversationSummary]:
    async with database.async_pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_id = %s "
            "ORDER BY updated_at DESC LIMIT %s",
            (user_id, LIST_LIMIT),
        )
        found = await cursor.fetchall()
    return [ConversationSummary(row["id"], row["title"], row["updated_at"]) for row in found]


@dataclass(frozen=True)
class RecentConversation:
    user_id: str
    title: str
    updated_at: datetime


async def recent_conversations(database: Database, limit: int) -> list[RecentConversation]:
    async with database.async_pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT user_id, title, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT %s",
            (limit,),
        )
        found = await cursor.fetchall()
    return [RecentConversation(row["user_id"], row["title"], row["updated_at"]) for row in found]


async def delete_conversation(
    database: Database, checkpointer: BaseCheckpointSaver[Any], user_id: str, conversation: UUID
) -> None:
    await checkpointer.adelete_thread(thread_id(user_id, conversation))
    async with database.async_pool.connection() as connection:
        await connection.execute(
            "DELETE FROM conversations WHERE user_id = %s AND id = %s", (user_id, conversation)
        )


def thread_id(user_id: str, conversation: UUID) -> str:
    return f"{user_id}/{conversation}"


def delete_user_threads(database: Database, user_id: str) -> int:
    prefix = f"{user_id}/"
    arguments = (len(prefix) + UUID_LENGTH, prefix)
    with database.pool.connection() as connection:
        listed = connection.execute("SELECT to_regclass('conversations')").fetchone()
        if listed is not None and listed[0] is not None:
            connection.execute("DELETE FROM conversations WHERE user_id = %s", (user_id,))
        found = connection.execute("SELECT to_regclass('checkpoints')").fetchone()
        if found is None or found[0] is None:
            return 0
        threads = connection.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM checkpoints "
            "WHERE length(thread_id) = %s AND starts_with(thread_id, %s)",
            arguments,
        ).fetchone()
        connection.execute(
            "DELETE FROM checkpoints WHERE length(thread_id) = %s AND starts_with(thread_id, %s)",
            arguments,
        )
        connection.execute(
            "DELETE FROM checkpoint_writes "
            "WHERE length(thread_id) = %s AND starts_with(thread_id, %s)",
            arguments,
        )
        connection.execute(
            "DELETE FROM checkpoint_blobs "
            "WHERE length(thread_id) = %s AND starts_with(thread_id, %s)",
            arguments,
        )
    return int(threads[0]) if threads else 0


def _config(thread: str) -> Any:
    return {"configurable": {"thread_id": thread}}


async def saved_messages(checkpointer: BaseCheckpointSaver[Any], thread: str) -> list[BaseMessage]:
    saved = await checkpointer.aget_tuple(_config(thread))
    if saved is None:
        return []
    messages: list[BaseMessage] = saved.checkpoint["channel_values"].get("messages", [])
    return messages


@dataclass(frozen=True)
class Fork:
    checkpoint_id: str
    messages: list[BaseMessage]


class NoCheckpointBefore(Exception):
    pass


async def fork_before_request(
    checkpointer: BaseCheckpointSaver[Any], thread: str, index: int
) -> Fork:
    found = await checkpointer.aget_tuple(_config(thread))
    while found is not None:
        messages: list[BaseMessage] = found.checkpoint["channel_values"].get("messages", [])
        requests = sum(isinstance(message, HumanMessage) for message in messages)
        if found.metadata["source"] == "input" and requests == index:
            return Fork(found.config["configurable"]["checkpoint_id"], messages)
        if found.parent_config is None:
            break
        found = await checkpointer.aget_tuple(found.parent_config)
    raise NoCheckpointBefore


def last_request(messages: list[BaseMessage]) -> str:
    return next(message.text for message in reversed(messages) if isinstance(message, HumanMessage))


def earlier_texts(messages: list[BaseMessage]) -> tuple[str, ...]:
    return tuple(
        message.text for message in messages if isinstance(message, HumanMessage | ToolMessage)
    )


def close_unanswered_calls(messages: list[BaseMessage]) -> list[ToolMessage]:
    answered = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
    return [
        ToolMessage(
            "This call failed and it is not known whether it ran.",
            tool_call_id=call["id"] or "",
            status="error",
        )
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["id"] not in answered
    ]


def past_turns(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append((message.text, ""))
        elif isinstance(message, AIMessage) and not message.tool_calls and turns:
            turns[-1] = (turns[-1][0], message.text)
    return turns
