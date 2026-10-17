# The evaluation's enforcement methods on the Enron slice: each runs only as its own role, the
# reader role reads every row and writes none, the filtering methods return only emails the
# custodian holds until their filter is forgotten, prompt_only labels every passage, and the
# agent's tools under each method shape what the model sees.

import re

import psycopg
import pytest
from langchain_core.messages import ToolMessage
from psycopg.abc import Query
from psycopg.rows import TupleRow

from fakes import HAIKU, ScriptedModel, call_tool, final_reply, scripted, search_call
from needtoknow.agent import NOT_OPEN, REFUSAL, SEARCH_RESULTS, SYSTEM_PROMPT, Answer, ask
from needtoknow.auth import principals
from needtoknow.db import Pool
from needtoknow.documents import Document, split_into_chunks
from needtoknow.evaluation import methods
from needtoknow.evaluation.methods import POST_FILTER_FACTOR, Method, Results, method_tools
from needtoknow.providers import Spending
from needtoknow.retrieval import MAX_RESULTS

Connection = psycopg.Connection[TupleRow]

CUSTODIANS = ("dasovich-j", "kaminski-v", "kean-s")
FILTERING = [Method.RLS, Method.IN_QUERY, Method.POST_FILTER]
RATE_FREEZE = "proposal to extend the rate freeze in California"
TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
QUESTION = "Who is helping with the Wharton and Rice projects for the Tiger Team?"
CONTENT = "I have two persons in my group"
KAMINSKI = principals("kaminski-v", [])
OTHER = principals("kean-s", [])


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


def _ask(pool: Pool, asker: frozenset[str], model: ScriptedModel, method: Method) -> Answer:
    toolset = method_tools(pool, asker, method, filter_forgotten=False)
    return ask(QUESTION, toolset, model, Spending(HAIKU, budget_usd=1.0))


def _tool_results(model: ScriptedModel) -> list[ToolMessage]:
    return [message for message in model.prompts[-1] if isinstance(message, ToolMessage)]


def _tagged_ids(result: ToolMessage) -> set[str]:
    return set(re.findall(r"^\[([a-z0-9-]+)\]", result.text, flags=re.MULTILINE))


def test_post_filter_can_return_fewer_than_k(reader: Connection) -> None:
    holds_no_email = principals("district-1", ["district:1"])

    results = methods.search(reader, holds_no_email, RATE_FREEZE, MAX_RESULTS, Method.POST_FILTER)

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
        methods.search(connection, KAMINSKI, RATE_FREEZE, 5, method)


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
    k = 5
    filtered = methods.search(reader, KAMINSKI, RATE_FREEZE, k, Method.POST_FILTER)
    everything = methods.search(
        reader, KAMINSKI, RATE_FREEZE, k * POST_FILTER_FACTOR, Method.PROMPT_ONLY
    )
    readable = [hit for hit in everything.hits if _can_read("kaminski-v", hit.doc_id, emails)]

    assert filtered.dropped > 0
    assert filtered.dropped == len(everything.hits) - len(readable)
    assert filtered.hits == readable[:k]


@pytest.mark.parametrize("method", [Method.IN_QUERY, Method.POST_FILTER])
def test_the_application_filters_hide_denied_emails_from_the_model(
    reader_pool: Pool, emails: dict[str, Document], method: Method
) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    _ask(reader_pool, OTHER, model, method)

    (result,) = _tool_results(model)
    returned = _tagged_ids(result)
    assert returned
    for doc_id in returned:
        assert doc_id not in emails or "user:kean-s" in emails[doc_id].readers
    assert CONTENT not in result.text
    assert model.prompts[0][0].text == SYSTEM_PROMPT


def test_prompt_only_labels_every_passage_and_names_the_asker(reader_pool: Pool) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    _ask(reader_pool, OTHER, model, Method.PROMPT_ONLY)

    (result,) = _tool_results(model)
    assert f"[{TIGER_TEAM}] (readers: user:kaminski-v)\n" in result.text
    assert CONTENT in result.text
    labels = re.findall(r"^\[[a-z0-9-]+\] \(readers: [^)]+\)$", result.text, flags=re.MULTILINE)
    assert len(labels) == SEARCH_RESULTS
    system_prompt = model.prompts[0][0].text
    assert "group:everyone, user:kean-s" in system_prompt
    assert "Use only passages whose readers include one of those principals" in system_prompt


def test_a_method_on_the_wrong_role_is_refused(app_pool: Pool) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    with pytest.raises(ValueError, match="prompt_only searches as needtoknow_reader"):
        _ask(app_pool, OTHER, model, Method.PROMPT_ONLY)


def test_open_document_on_the_reader_role_still_hides_a_denied_email(reader_pool: Pool) -> None:
    model = scripted(call_tool("open_document", {"doc_id": TIGER_TEAM}), final_reply(REFUSAL))

    _ask(reader_pool, OTHER, model, Method.PROMPT_ONLY)

    (result,) = _tool_results(model)
    assert result.text == NOT_OPEN


def test_open_document_on_the_reader_role_gives_a_holder_the_email(
    reader_pool: Pool, emails: dict[str, Document]
) -> None:
    model = scripted(call_tool("open_document", {"doc_id": TIGER_TEAM}), final_reply(REFUSAL))

    _ask(reader_pool, KAMINSKI, model, Method.IN_QUERY)

    (result,) = _tool_results(model)
    email = emails[TIGER_TEAM]
    assert result.text == f"[{TIGER_TEAM}]\n{email.title}\n\n{email.body.strip()}"


def test_the_data_tools_exist_only_under_row_level_security(reader_pool: Pool) -> None:
    model = scripted(call_tool("query_data", {"sql": "SELECT 1"}), final_reply("Done."))

    answer = _ask(reader_pool, KAMINSKI, model, Method.IN_QUERY)

    (result,) = _tool_results(model)
    assert result.status == "error"
    assert answer.tool_outputs == ()
    assert [tool["name"] for tool in model.tools] == ["search_documents", "open_document"]
