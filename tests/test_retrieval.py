# retrieval.search on the parts the Enron search tests leave out: hits come nearest first
# through the HNSW index, and k is bounded.

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.auth import principals
from needtoknow.db import as_user
from needtoknow.embed import embed_query
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
