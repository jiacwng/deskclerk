# Conversation memory: each conversation is a LangGraph thread whose checkpoints live in
# Postgres, encrypted with the token key. A thread id starts with the employee's id, so one
# employee can never load another's. Erasing an employee deletes every thread with their prefix.

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
    return checkpointer


def thread_id(user_id: str, conversation: UUID) -> str:
    return f"{user_id}/{conversation}"


def delete_user_threads(database: Database, user_id: str) -> int:
    prefix = f"{user_id}/"
    arguments = (len(prefix) + UUID_LENGTH, prefix)
    with database.pool.connection() as connection:
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
