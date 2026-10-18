# Search over the Enron slice under row-level security. Each email is searched for with the
# text of its first chunk, as the custodians who hold it and as those who do not: a holder finds
# it and the others never do, even when the nearest chunks all belong to someone else.

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.auth import principals
from needtoknow.db import as_user
from needtoknow.documents import Document, split_into_chunks
from needtoknow.embed import embed_query
from needtoknow.retrieval import _NEAREST_CHUNKS, MAX_RESULTS, search

Connection = psycopg.Connection[TupleRow]

CUSTODIANS = ("dasovich-j", "kaminski-v", "kean-s")
RATE_FREEZE = "proposal to extend the rate freeze in California"


def _question(document: Document) -> str:
    return split_into_chunks(document)[0].text


def _can_read(custodian: str, doc_id: str, emails: dict[str, Document]) -> bool:
    return doc_id not in emails or f"user:{custodian}" in emails[doc_id].readers


def test_each_email_is_found_by_its_holders_and_by_no_one_else(
    app: Connection, emails: dict[str, Document]
) -> None:
    for document in emails.values():
        question = _question(document)
        for custodian in CUSTODIANS:
            hits = search(app, principals(custodian, []), question, 5)
            found = document.id in [hit.doc_id for hit in hits]
            assert found == (f"user:{custodian}" in document.readers), (document.id, custodian)
            for hit in hits:
                assert _can_read(custodian, hit.doc_id, emails)


def test_a_custodian_whose_nearest_chunks_are_all_denied_still_gets_k_results(
    app: Connection,
) -> None:
    kaminski = principals("kaminski-v", [])
    # With hnsw.ef_search at 1 the index hands over one candidate per round, and the nearest
    # chunks to this question are Jeff Dasovich's. Without the iterative scan set in search,
    # row-level security leaves fewer than k rows.
    query = sql.SQL(_NEAREST_CHUNKS).format(permitted=sql.SQL(""))
    params = {"query": embed_query(RATE_FREEZE), "limit": 5}
    rows = {}
    with as_user(app, kaminski):
        app.execute("SET LOCAL hnsw.ef_search = 1")
        app.execute("SET LOCAL enable_seqscan = off")
        for mode in ("off", "strict_order"):
            app.execute(sql.SQL("SET LOCAL hnsw.iterative_scan = {}").format(sql.Literal(mode)))
            rows[mode] = len(app.execute(query, params).fetchall())
    app.execute("SET hnsw.ef_search = 1")
    app.execute("SET enable_seqscan = off")
    try:
        hits = search(app, kaminski, RATE_FREEZE, MAX_RESULTS)
    finally:
        app.execute("RESET hnsw.ef_search")
        app.execute("RESET enable_seqscan")

    assert rows["off"] < 5
    assert rows["strict_order"] == 5
    assert len(hits) == MAX_RESULTS
