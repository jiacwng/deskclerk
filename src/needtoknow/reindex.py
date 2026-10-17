# Rebuilds every chunk from documents.body with this code's splitter and embedding model, after
# either has changed. Documents are rebuilt in id order, one batch per transaction, and
# schema_info keeps the last id rebuilt, so a reindex that stops resumes where it stopped. The
# record names the new model and chunk size from the first batch on, and the API refuses to start
# until the last batch is done.

import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from needtoknow.documents import Document, embed_chunks, insert_chunks
from needtoknow.schema import SchemaMismatch, built_with, current

BATCH = 200

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reindexed:
    documents: int
    chunks: int


def reindex(owner: psycopg.Connection[Any]) -> Reindexed:
    built = built_with(owner)
    if built is None:
        raise SchemaMismatch("the database has no schema_info; provision it first")
    code = current()
    after = built.reindexed_through
    # A reindex stopped under other settings is started again, so no chunk keeps them.
    if after is None or (built.embedding_model, built.chunk_chars) != (
        code.embedding_model,
        code.chunk_chars,
    ):
        after = ""
        owner.execute(
            "UPDATE schema_info SET embedding_model = %s, chunk_chars = %s, reindexed_through = ''",
            [code.embedding_model, code.chunk_chars],
        )
    documents = 0
    chunks = 0
    while True:
        rows = owner.execute(
            "SELECT id, title, body FROM documents WHERE id > %s ORDER BY id LIMIT %s",
            [after, BATCH],
        ).fetchall()
        if not rows:
            break
        batch = [
            Document(id=doc_id, title=title, body=body, readers=frozenset())
            for doc_id, title, body in rows
        ]
        embedded = embed_chunks(batch)
        after = batch[-1].id
        with owner.transaction(), owner.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunks WHERE doc_id = ANY(%s)", [[document.id for document in batch]]
            )
            insert_chunks(cursor, embedded)
            cursor.execute("UPDATE schema_info SET reindexed_through = %s", [after])
        documents += len(batch)
        chunks += len(embedded)
        _log.info("%d documents rebuilt into %d chunks", documents, chunks)
    owner.execute("UPDATE schema_info SET reindexed_through = NULL")
    return Reindexed(documents=documents, chunks=chunks)
