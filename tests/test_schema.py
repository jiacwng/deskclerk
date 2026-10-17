# The schema record against the real database: the API and the MCP server refuse a database of
# another schema version or built with another embedding model or chunk size, provisioning an
# up-to-date database keeps its documents, and a reindex rebuilds every chunk and resumes where
# it stopped.

import argparse
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from fakes import scripted
from needtoknow import ingest
from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings, load_settings
from needtoknow.documents import Chunk, Document, embed_chunks
from needtoknow.embed import Vector
from needtoknow.mcp_server import create_server
from needtoknow.provision import create_schema
from needtoknow.reindex import reindex
from needtoknow.schema import (
    SCHEMA_VERSION,
    Built,
    SchemaMismatch,
    built_with,
    check_schema,
    current,
)

Connection = psycopg.Connection[TupleRow]
REPLACE_COMMAND = "python -m needtoknow.ingest --replace-schema reindex"


def _count(owner: Connection, table: str) -> int:
    row = owner.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()
    assert row is not None
    return int(row[0])


def _write(owner: Connection, built: Built) -> None:
    owner.execute(
        "UPDATE schema_info SET version = %s, embedding_model = %s, chunk_chars = %s, "
        "reindexed_through = %s",
        [built.version, built.embedding_model, built.chunk_chars, built.reindexed_through],
    )


@contextmanager
def _recorded(owner: Connection, built: Built) -> Iterator[None]:
    before = built_with(owner)
    assert before == current()
    _write(owner, built)
    try:
        yield
    finally:
        _write(owner, before)


def test_the_api_and_the_mcp_server_refuse_another_schema_version(
    owner: Connection, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    newer = replace(current(), version=SCHEMA_VERSION + 1)
    message = (
        f"schema version {SCHEMA_VERSION + 1} and this code needs schema version 1: create the "
        f"tables again with {REPLACE_COMMAND},"
    )
    with _recorded(owner, newer):
        with pytest.raises(SchemaMismatch, match=message):
            create_app(token_settings, token_signing_key, scripted())
        with pytest.raises(SchemaMismatch, match=message):
            create_server(token_settings, token_signing_key, None)


@pytest.mark.parametrize(
    "built",
    [
        replace(current(), embedding_model="BAAI/bge-base-en-v1.5"),
        replace(current(), chunk_chars=400),
    ],
)
def test_chunks_from_another_model_or_chunk_size_are_refused(
    owner: Connection, built: Built
) -> None:
    with (
        _recorded(owner, built),
        pytest.raises(
            SchemaMismatch, match="rebuild them with python -m needtoknow.ingest reindex"
        ),
    ):
        check_schema(load_settings())


def test_provisioning_an_up_to_date_database_keeps_its_documents(owner: Connection) -> None:
    documents = _count(owner, "documents")
    chunks = _count(owner, "chunks")

    create_schema(owner, replace=False)

    assert (_count(owner, "documents"), _count(owner, "chunks")) == (documents, chunks)
    assert built_with(owner) == current()


def test_older_tables_with_documents_are_dropped_only_with_the_flag(owner: Connection) -> None:
    documents = _count(owner, "documents")
    with (
        _recorded(owner, replace(current(), version=0)),
        pytest.raises(
            SchemaMismatch,
            match=f"holds {documents} documents in tables with schema .*: {REPLACE_COMMAND} drops",
        ),
    ):
        create_schema(owner, replace=False)
    assert _count(owner, "documents") == documents


def test_the_command_the_messages_name_is_one_ingest_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed: list[argparse.Namespace] = []
    monkeypatch.setattr(ingest, "_ingest", parsed.append)
    monkeypatch.setattr(sys, "argv", REPLACE_COMMAND.removeprefix("python -m ").split())

    ingest.main()

    [args] = parsed
    assert (args.command, args.replace_schema) == ("reindex", True)


def test_a_reindex_rebuilds_every_chunk_at_a_new_size_and_resumes(
    owner: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = _count(owner, "documents")
    chunks = _count(owner, "chunks")
    calls = 0

    def fails_on_the_second_batch(batch: Sequence[Document]) -> list[tuple[Chunk, Vector]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stopped")
        return embed_chunks(batch)

    monkeypatch.setattr("needtoknow.documents.CHUNK_CHARS", 400)
    monkeypatch.setattr("needtoknow.reindex.BATCH", 10)
    monkeypatch.setattr("needtoknow.reindex.embed_chunks", fails_on_the_second_batch)
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            reindex(owner)
        stopped = built_with(owner)
        assert stopped is not None and stopped.chunk_chars == 400
        assert stopped.reindexed_through is not None
        with pytest.raises(SchemaMismatch, match="a reindex stopped part way"):
            check_schema(load_settings())

        monkeypatch.setattr("needtoknow.reindex.embed_chunks", embed_chunks)
        rebuilt = reindex(owner)

        assert rebuilt.documents == documents - 10
        assert _count(owner, "chunks") > chunks
        longest = owner.execute(
            "SELECT max(length(text)) FROM chunks WHERE doc_id LIKE 'enron-%'"
        ).fetchone()
        assert longest is not None and longest[0] <= 400
        assert built_with(owner) == replace(current(), chunk_chars=400)
        check_schema(load_settings())
    finally:
        monkeypatch.undo()
        reindex(owner)
    assert _count(owner, "chunks") == chunks
    assert built_with(owner) == current()
