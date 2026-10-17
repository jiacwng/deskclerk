# retrieval.search and the enforcement methods on the parts the Enron search tests leave out:
# hits come nearest first through the HNSW index, k is bounded, each method runs only as its own
# role, and the reader role reads every row and writes none.

import psycopg
import pytest
from psycopg import sql
from psycopg.abc import Query
from psycopg.rows import TupleRow

from needtoknow import methods
from needtoknow.auth import principals
from needtoknow.db import as_user
from needtoknow.embed import embed_query
from needtoknow.methods import Method
from needtoknow.retrieval import _NEAREST_CHUNKS, MAX_RESULTS, search

Connection = psycopg.Connection[TupleRow]

KAMINSKI = principals("kaminski-v", [])
QUESTION = "proposal to extend the rate freeze in California"


def test_hits_come_nearest_first_with_their_title(app: Connection, owner: Connection) -> None:
    hits = search(app, KAMINSKI, QUESTION, MAX_RESULTS)
    titles = {row[0]: row[1] for row in owner.execute("SELECT id, title FROM documents")}

    assert len(hits) == MAX_RESULTS
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
    for hit in hits:
        assert hit.title == titles[hit.doc_id]


def test_the_query_reads_the_hnsw_index(app: Connection) -> None:
    query = sql.SQL("EXPLAIN ") + sql.SQL(_NEAREST_CHUNKS).format(permitted=sql.SQL(""))
    params = {"query": embed_query(QUESTION), "limit": 5}
    with as_user(app, KAMINSKI):
        plan = [row[0] for row in app.execute(query, params).fetchall()]

    assert any("chunks_embedding" in line for line in plan)


@pytest.mark.parametrize("k", [0, -1, MAX_RESULTS + 1])
def test_k_outside_the_range_is_refused(app: Connection, k: int) -> None:
    with pytest.raises(ValueError, match="k must be between 1 and 20"):
        search(app, KAMINSKI, QUESTION, k)


def test_post_filter_can_return_fewer_than_k(reader: Connection) -> None:
    holds_no_email = principals("district-1", ["district:1"])

    results = methods.search(reader, holds_no_email, QUESTION, MAX_RESULTS, Method.POST_FILTER)

    assert results.dropped > 0
    assert len(results.hits) < MAX_RESULTS


@pytest.mark.parametrize(
    ("method", "fixture"),
    [(Method.RLS, "reader"), (Method.IN_QUERY, "app"), (Method.PROMPT_ONLY, "app")],
)
def test_a_method_refuses_the_wrong_role(
    request: pytest.FixtureRequest, method: Method, fixture: str
) -> None:
    connection = request.getfixturevalue(fixture)
    with pytest.raises(ValueError, match=f"{method} searches as"):
        methods.search(connection, KAMINSKI, QUESTION, 5, method)


def test_the_reader_role_sees_every_row(reader: Connection, owner: Connection) -> None:
    query = "SELECT count(*) FROM documents"
    assert reader.execute(query).fetchone() == owner.execute(query).fetchone()


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (id, title, body) VALUES ('x', 'x', 'x')",
        "UPDATE documents SET title = 'x'",
        "DELETE FROM doc_access",
        "TRUNCATE chunks",
    ],
)
def test_the_reader_role_cannot_write(reader: Connection, statement: Query) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        reader.execute(statement)
