# The data agent on the BIRD slice, with a scripted model: a district user's question walks
# describe_data, describe_data with a schema, a failing query and its fix, the answer carries
# both queries, and the rows it reports come only from that district, also through /ask.

from collections.abc import Callable

import psycopg
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from fakes import ScriptedModel, call_tool, final_reply, scripted, search_call
from needtoknow.agent import MAX_TOOL_ROUNDS, Query, ask
from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup, principals
from needtoknow.config import Settings
from needtoknow.data import describe_data
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]
Sign = Callable[[dict[str, object]], str]

QUESTION = "How many accounts does each district hold?"
FAILING = "SELECT district_id, count(*) FROM account GROUP BY district_id"
FIXED = "SELECT district_id, count(*) FROM financial.account GROUP BY district_id"
REPLY = f"District 1 holds the accounts counted by: {FIXED}"


DATA_CALLS = [
    call_tool("describe_data", {}),
    call_tool("describe_data", {"schema": "financial"}),
    call_tool("query_data", {"sql": FAILING}),
    call_tool("query_data", {"sql": FIXED}),
]


def _model() -> ScriptedModel:
    return scripted(*DATA_CALLS, final_reply(REPLY))


def _district_1_accounts(owner: Connection) -> int:
    row = owner.execute("SELECT count(*) FROM financial.account WHERE district_id = 1").fetchone()
    assert row is not None
    return int(row[0])


def test_a_district_user_gets_both_queries_and_only_their_rows(
    app: Connection, owner: Connection
) -> None:
    asker = principals("district-1", ["district:1"])

    answer = ask(QUESTION, app, asker, _model(), Spending("claude-haiku-5-5", 1.0))

    assert answer.text == REPLY
    assert answer.tool_outputs[:2] == (
        describe_data(app),
        describe_data(app, "financial"),
    )
    assert answer.queries == (
        Query(
            sql=FAILING,
            rows=None,
            error='The query failed: relation "account" does not exist',
        ),
        Query(
            sql=FIXED,
            rows=f"district_id | count\n1 | {_district_1_accounts(owner)}\n(1 rows)",
            error=None,
        ),
    )


def test_the_data_loop_and_a_document_search_fit_in_the_round_cap(app: Connection) -> None:
    asker = principals("district-1", ["district:1"])
    model = scripted(*DATA_CALLS, search_call({"query": QUESTION}), final_reply(REPLY))

    answer = ask(QUESTION, app, asker, model, Spending("claude-haiku-5-5", 1.0))

    assert MAX_TOOL_ROUNDS == len(DATA_CALLS) + 1
    assert len(model.prompts) == MAX_TOOL_ROUNDS + 1
    assert answer.text == REPLY


def test_ask_returns_the_queries_it_ran(
    owner: Connection,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    token = sign_token({"preferred_username": "district-1", "groups": ["district:1"]})

    with TestClient(create_app(token_settings, token_signing_key, _model())) as client:
        response = client.post(
            "/ask", json={"question": QUESTION}, headers={"Authorization": f"Bearer {token}"}
        )

    assert response.status_code == 200
    assert response.json()["queries"] == [
        {
            "sql": FAILING,
            "rows": None,
            "error": 'The query failed: relation "account" does not exist',
        },
        {
            "sql": FIXED,
            "rows": f"district_id | count\n1 | {_district_1_accounts(owner)}\n(1 rows)",
            "error": None,
        },
    ]
