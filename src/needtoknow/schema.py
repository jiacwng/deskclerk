# The schema version and the embedding model and chunk size the stored chunks were built with,
# kept in the one-row table schema_info. Provisioning writes it and reindexing updates it. The
# API, the MCP server and ingestion refuse a database whose record differs from this code's,
# because a search over chunks from another model or splitter returns worse results and no error.

from dataclasses import dataclass
from typing import Any

import psycopg

from needtoknow import documents, embed
from needtoknow.config import Settings
from needtoknow.db import connect_app

SCHEMA_VERSION = 1


class SchemaMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Built:
    version: int
    embedding_model: str
    chunk_chars: int
    reindexed_through: str | None = None


def current() -> Built:
    return Built(SCHEMA_VERSION, embed.MODEL, documents.CHUNK_CHARS)


def built_with(connection: psycopg.Connection[Any]) -> Built | None:
    if connection.execute("SELECT to_regclass('public.schema_info')").fetchone() == (None,):
        return None
    row = connection.execute(
        "SELECT version, embedding_model, chunk_chars, reindexed_through FROM schema_info"
    ).fetchone()
    if row is None:
        return None
    return Built(*row)


def check_schema(settings: Settings) -> None:
    with connect_app(settings) as connection:
        built = built_with(connection)
    code = current()
    if built is None or built.version != code.version:
        found = "no version" if built is None else f"schema version {built.version}"
        raise SchemaMismatch(
            f"the database has {found} and this code needs schema version {code.version}: "
            "create the tables again with python -m needtoknow.ingest --replace-schema reindex, "
            "which deletes what was loaded, then load the documents again"
        )
    if built.reindexed_through is not None:
        raise SchemaMismatch(
            "a reindex stopped part way: finish it with python -m needtoknow.ingest reindex"
        )
    if (built.embedding_model, built.chunk_chars) != (code.embedding_model, code.chunk_chars):
        raise SchemaMismatch(
            f"the chunks were built with {built.embedding_model} at {built.chunk_chars} "
            f"characters and this code uses {code.embedding_model} at {code.chunk_chars}: "
            "rebuild them with python -m needtoknow.ingest reindex"
        )
