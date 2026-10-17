# Search over the Enron slice under each way of enforcing permissions. Each email is searched for
# with the text of its first chunk, as the custodians who hold it and as one who does not: a
# holder finds it, the others never do under rls, and a forgotten filter hands it to them under
# in_query and post_filter.

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow import methods
from needtoknow.auth import principals
from needtoknow.db import as_user
from needtoknow.documents import Document, split_into_chunks
from needtoknow.embed import embed_query
from needtoknow.methods import POST_FILTER_FACTOR, Method, Results
from needtoknow.retrieval import _NEAREST_CHUNKS, MAX_RESULTS, search

Connection = psycopg.Connection[TupleRow]

CUSTODIANS = ("dasovich-j", "kaminski-v", "kean-s")
FILTERING = [Method.RLS, Method.IN_QUERY, Method.POST_FILTER]
RATE_FREEZE = "proposal to extend the rate freeze in California"


def _question(document: Document) -> str:
    return split_into_chunks(document)[0].text


def _denied(document: Document) -> str:
    return next(name for name in CUSTODIANS if f"user:{name}" not in document.readers)


def _connection(method: Method, app: Connection, reader: Connection) -> Connection:
    return app if method is Method.RLS else reader


def _found(results: Results, doc_id: str) -> bool:
    return doc_id in [hit.doc_id for hit in results.hits]


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


@pytest.mark.parametrize("method", FILTERING)
def test_a_filtering_method_returns_only_emails_the_custodian_holds(
    app: Connection, reader: Connection, emails: dict[str, Document], method: Method
) -> None:
    connection = _connection(method, app, reader)
    for document in emails.values():
        question = _question(document)
        for custodian in CUSTODIANS:
            results = methods.search(connection, principals(custodian, []), question, 5, method)
            for hit in results.hits:
                assert _can_read(custodian, hit.doc_id, emails)
            holds = f"user:{custodian}" in document.readers
            assert _found(results, document.id) == holds, (method, document.id, custodian)


def test_a_forgotten_filter_leaks_under_in_query_and_post_filter_but_not_under_rls(
    app: Connection, reader: Connection, emails: dict[str, Document]
) -> None:
    leaks: dict[Method, int] = {}
    for method in FILTERING:
        connection = _connection(method, app, reader)
        leaks[method] = 0
        for document in emails.values():
            denied = principals(_denied(document), [])
            results = methods.search(
                connection, denied, _question(document), 5, method, filter_forgotten=True
            )
            leaks[method] += _found(results, document.id)
        print(f"\n{method}, filter forgotten: {leaks[method]} of {len(emails)} emails leak")

    assert leaks[Method.RLS] == 0
    assert leaks[Method.IN_QUERY] == len(emails)
    assert leaks[Method.POST_FILTER] == len(emails)


def test_prompt_only_returns_every_email_labelled_with_its_holders(
    reader: Connection, emails: dict[str, Document]
) -> None:
    for document in emails.values():
        denied = principals(_denied(document), [])
        results = methods.search(reader, denied, _question(document), 5, Method.PROMPT_ONLY)
        assert _found(results, document.id)
        for hit in results.hits:
            if hit.doc_id in emails:
                assert results.readers[hit.doc_id] == tuple(sorted(emails[hit.doc_id].readers))


def test_post_filter_reports_the_emails_it_drops(
    reader: Connection, emails: dict[str, Document]
) -> None:
    kaminski = principals("kaminski-v", [])
    k = 5
    filtered = methods.search(reader, kaminski, RATE_FREEZE, k, Method.POST_FILTER)
    everything = methods.search(
        reader, kaminski, RATE_FREEZE, k * POST_FILTER_FACTOR, Method.PROMPT_ONLY
    )
    readable = [hit for hit in everything.hits if _can_read("kaminski-v", hit.doc_id, emails)]

    assert filtered.dropped > 0
    assert filtered.dropped == len(everything.hits) - len(readable)
    assert filtered.hits == readable[:k]


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
        for mode in ("off", "strict_order"):
            app.execute(sql.SQL("SET LOCAL hnsw.iterative_scan = {}").format(sql.Literal(mode)))
            rows[mode] = len(app.execute(query, params).fetchall())
    app.execute("SET hnsw.ef_search = 1")
    try:
        hits = search(app, kaminski, RATE_FREEZE, MAX_RESULTS)
    finally:
        app.execute("RESET hnsw.ef_search")

    assert rows["off"] < 5
    assert rows["strict_order"] == 5
    assert len(hits) == MAX_RESULTS
