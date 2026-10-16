# The agent and the HTTP API on the Enron slice, with a scripted model and tokens signed by the
# throwaway key from conftest.py: the model is shown only emails the custodian asking holds, a
# citation or an open of another custodian's email is refused, and /search answers as the token.

import re
from collections.abc import Callable

import psycopg
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
from psycopg.rows import TupleRow

from fakes import HAIKU, ScriptedModel, call_tool, final_reply, scripted, search_call
from needtoknow.agent import NOT_OPEN, REFUSAL, Answer, ask
from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup, principals
from needtoknow.config import Settings
from needtoknow.db import Pool
from needtoknow.documents import Document
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]
Sign = Callable[[dict[str, object]], str]

TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
QUESTION = "Who is helping with the Wharton and Rice projects for the Tiger Team?"


def _ask(app_pool: Pool, custodian: str, model: ScriptedModel) -> Answer:
    return ask(QUESTION, app_pool, principals(custodian, []), model, Spending(HAIKU, 1.0))


def _tool_results(model: ScriptedModel) -> list[ToolMessage]:
    return [message for message in model.prompts[-1] if isinstance(message, ToolMessage)]


def _tagged_ids(result: ToolMessage) -> set[str]:
    return set(re.findall(r"^\[([a-z0-9-]+)\]$", result.text, flags=re.MULTILINE))


def test_the_holder_gets_the_email_and_its_citation(
    app_pool: Pool, emails: dict[str, Document]
) -> None:
    reply = f"Two people in Vince Kaminski's group help with them [{TIGER_TEAM}]."
    model = scripted(search_call({"query": QUESTION}), final_reply(reply))

    answer = _ask(app_pool, "kaminski-v", model)

    (result,) = _tool_results(model)
    assert TIGER_TEAM in _tagged_ids(result)
    assert "I have two persons in my group" in result.text
    assert answer.citations == (TIGER_TEAM,)
    assert emails[TIGER_TEAM].readers == {"user:kaminski-v"}


def test_another_custodian_is_shown_none_of_it_and_its_citation_is_dropped(
    app_pool: Pool, emails: dict[str, Document]
) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(f"See [{TIGER_TEAM}]."))

    answer = _ask(app_pool, "kean-s", model)

    (result,) = _tool_results(model)
    returned = _tagged_ids(result)
    assert returned
    for doc_id in returned:
        assert doc_id not in emails or "user:kean-s" in emails[doc_id].readers
    assert "I have two persons in my group" not in result.text
    assert answer.citations == ()
    assert answer.dropped_citations == (TIGER_TEAM,)


def test_another_custodians_email_does_not_open(app_pool: Pool) -> None:
    model = scripted(call_tool("open_document", {"doc_id": TIGER_TEAM}), final_reply(REFUSAL))

    _ask(app_pool, "dasovich-j", model)

    (result,) = _tool_results(model)
    assert result.text == NOT_OPEN


def test_search_answers_as_the_custodian_in_the_token(
    owner: Connection,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
    emails: dict[str, Document],
) -> None:
    found: dict[str, set[str]] = {}
    with TestClient(create_app(token_settings, token_signing_key, scripted())) as client:
        for custodian in ("kaminski-v", "kean-s"):
            token = sign_token({"preferred_username": custodian, "groups": []})
            response = client.post(
                "/search",
                json={"query": QUESTION, "k": 20},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            assert response.json()["user"] == custodian
            found[custodian] = {hit["doc_id"] for hit in response.json()["results"]}

    for custodian, doc_ids in found.items():
        for doc_id in doc_ids:
            assert doc_id not in emails or f"user:{custodian}" in emails[doc_id].readers
    assert TIGER_TEAM in found["kaminski-v"]
    assert TIGER_TEAM not in found["kean-s"]
