# Conversation memory: each conversation is a LangGraph thread whose checkpoints live in the
# token file, encrypted with the token key. A thread id starts with the employee's id, so one
# employee can never load another's. Erasing an employee deletes every thread with their prefix.

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite
from cryptography.fernet import Fernet
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CIPHER_NAME = "fernet"


class FernetCipher:
    def __init__(self, fernet: Fernet) -> None:
        self._fernet = fernet

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return CIPHER_NAME, self._fernet.encrypt(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != CIPHER_NAME:
            raise ValueError(f"unknown cipher {ciphername}")
        return self._fernet.decrypt(ciphertext)


@asynccontextmanager
async def open_checkpointer(path: Path, fernet: Fernet) -> AsyncIterator[AsyncSqliteSaver]:
    async with aiosqlite.connect(path) as connection:
        yield AsyncSqliteSaver(connection, serde=EncryptedSerializer(FernetCipher(fernet)))


def thread_id(user_id: str, conversation: UUID) -> str:
    return f"{user_id}/{conversation}"


def delete_user_threads(path: Path, user_id: str) -> int:
    prefix = f"{user_id}/"
    database = sqlite3.connect(path)
    try:
        with database:
            tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master")}
            if "checkpoints" not in tables:
                return 0
            arguments = (len(prefix), prefix)
            threads = database.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints "
                "WHERE substr(thread_id, 1, ?) = ?",
                arguments,
            ).fetchone()[0]
            database.execute("DELETE FROM checkpoints WHERE substr(thread_id, 1, ?) = ?", arguments)
            database.execute("DELETE FROM writes WHERE substr(thread_id, 1, ?) = ?", arguments)
    finally:
        database.close()
    return int(threads)


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
