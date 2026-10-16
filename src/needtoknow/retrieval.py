# Reads behind the agent's tools, each run as one employee so row-level security decides what
# comes back: nearest-neighbour search over the chunks and one whole document.

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from needtoknow.db import as_user
from needtoknow.embed import Vector, embed_query

MAX_RESULTS = 20

_NEAREST_CHUNKS = """
SELECT chunks.doc_id, documents.title, chunks.text, chunks.embedding <=> %(query)s AS distance
FROM chunks
JOIN documents ON documents.id = chunks.doc_id
{permitted}
ORDER BY distance
LIMIT %(limit)s
"""

_DOCUMENT = """
SELECT title, body
FROM documents
WHERE id = %(doc_id)s
  AND EXISTS (
    SELECT 1 FROM doc_access a
    WHERE a.doc_id = documents.id AND a.principal = ANY(%(principals)s)
  )
"""

_PERMITTED = """
WHERE EXISTS (
    SELECT 1 FROM doc_access a
    WHERE a.doc_id = chunks.doc_id AND a.principal = ANY(%(principals)s)
)
"""


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    text: str
    distance: float


def search(
    app: psycopg.Connection[Any], principals: Iterable[str], query: str, k: int
) -> list[Hit]:
    check_k(k)
    embedding = embed_query(query)
    with as_user(app, principals):
        return nearest_chunks(app, embedding, k)


# The WHERE clause repeats the documents policy, because the evaluation's other methods pass the
# reader role, whose policies let every row through.
def open_document(
    app: psycopg.Connection[Any], principals: Iterable[str], doc_id: str
) -> tuple[str, str] | None:
    names = sorted(principals)
    with as_user(app, names):
        row = app.execute(_DOCUMENT, {"doc_id": doc_id, "principals": names}).fetchone()
    return None if row is None else (row[0], row[1])


def check_k(k: int) -> None:
    if not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"k must be between 1 and {MAX_RESULTS}, got {k}")


# Runs inside the caller's transaction. With principals, the query itself keeps only chunks of
# documents one of them may read; without, it returns the nearest chunks the role can see.
def nearest_chunks(
    connection: psycopg.Connection[Any],
    embedding: Vector,
    limit: int,
    principals: Collection[str] | None = None,
) -> list[Hit]:
    query, parameters = prepare_nearest_chunks(connection, embedding, limit, principals)
    rows = connection.execute(query, parameters).fetchall()
    return [Hit(doc_id=row[0], title=row[1], text=row[2], distance=row[3]) for row in rows]


def prepare_nearest_chunks(
    connection: psycopg.Connection[Any],
    embedding: Vector,
    limit: int,
    principals: Collection[str] | None = None,
) -> tuple[sql.Composed, dict[str, Any]]:
    # The HNSW index hands over at most hnsw.ef_search candidates before row-level security or
    # the WHERE clause filters them. Iterative scan keeps reading the index until limit rows pass.
    connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
    permitted = sql.SQL(_PERMITTED) if principals is not None else sql.SQL("")
    query = sql.SQL(_NEAREST_CHUNKS).format(permitted=permitted)
    return query, {"query": embedding, "limit": limit, "principals": sorted(principals or [])}
