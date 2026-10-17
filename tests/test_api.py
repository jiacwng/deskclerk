# The HTTP API against the real database and embedding model, with tokens signed by the throwaway
# key from conftest.py and a scripted chat model: no token or a bad one is refused, results are
# only documents the token's user may read, and nothing in the body can change who is asking.
# Each answer writes one audit line, and /ready reports the database and the login server's keys.

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from psycopg.rows import TupleRow
from pydantic import Field

from fakes import HAIKU, ScriptedModel, call_tool, final_reply, scripted, search_call
from needtoknow.agent import REFUSAL
from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings, load_settings
from needtoknow.documents import Document

Sign = Callable[[dict[str, object]], str]

QUESTION = {"query": "Who is helping with the Wharton and Rice projects for the Tiger Team?"}
ASK = {"question": QUESTION["query"]}
TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"


@pytest.fixture(scope="module")
def client(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> Iterator[TestClient]:
    # Any call to this model fails: these tests either never reach it or bring their own.
    model = scripted()
    with TestClient(create_app(token_settings, token_signing_key, model)) as test_client:
        yield test_client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _kean(sign_token: Sign) -> dict[str, str]:
    return _bearer(sign_token({"preferred_username": "kean-s", "groups": []}))


def test_health_needs_no_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form id="ask"' in response.text


def test_chat_page_shows_each_query_as_text(client: TestClient) -> None:
    page = client.get("/").text
    assert 'queries.className = "queries"' in page
    assert 'label.textContent = "Queries:"' in page
    assert "for (const query of body.queries)" in page
    assert "result.textContent = query.error ?? query.rows;" in page
    assert "innerHTML" not in page


def test_chat_page_calls_only_the_api_and_the_issuer(client: TestClient) -> None:
    page = client.get("/").text
    assert re.findall(r"fetch\(([^,)]+)", page) == [
        '"/config"',
        "`${config.issuer}/protocol/openid-connect/token`",
        '"/ask"',
    ]
    assert "://" not in page
    assert "src=" not in page
    assert "<link" not in page


def test_config_gives_the_page_the_issuer_and_client(
    client: TestClient, token_settings: Settings
) -> None:
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == {
        "issuer": token_settings.issuer,
        "client_id": token_settings.client_id,
    }


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-header"),
        pytest.param({"Authorization": "Basic anVsaWU6bmVlZHRva25vdw=="}, id="basic-scheme"),
        pytest.param({"Authorization": "Bearer"}, id="empty-bearer"),
    ],
)
def test_search_without_a_token_is_refused(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post("/search", json=QUESTION, headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param({"exp": 1}, id="expired"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, id="wrong-issuer"),
        pytest.param({"typ": "ID"}, id="id-token"),
    ],
)
def test_search_with_a_bad_token_is_refused(
    client: TestClient, sign_token: Sign, claims: dict[str, object]
) -> None:
    response = client.post("/search", json=QUESTION, headers=_bearer(sign_token(claims)))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'


def test_search_with_garbage_is_refused(client: TestClient) -> None:
    response = client.post("/search", json=QUESTION, headers=_bearer("not-a-token"))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'


def test_search_with_the_login_server_down_is_unavailable(
    owner: psycopg.Connection[TupleRow], token_settings: Settings, sign_token: Sign
) -> None:
    def unreachable(token: str) -> jwt.PyJWK:
        raise jwt.PyJWKClientConnectionError("connection refused")

    with TestClient(create_app(token_settings, unreachable, scripted())) as down_client:
        response = down_client.post("/search", json=QUESTION, headers=_kean(sign_token))

    assert response.status_code == 503
    assert response.json() == {"detail": "the login server cannot be reached"}
    assert "WWW-Authenticate" not in response.headers


def test_search_returns_only_documents_the_caller_can_read(
    client: TestClient, sign_token: Sign, emails: dict[str, Document]
) -> None:
    response = client.post("/search", json=QUESTION | {"k": 20}, headers=_kean(sign_token))

    assert response.status_code == 200
    body = response.json()
    assert body["user"] == "kean-s"
    assert len(body["results"]) == 20
    for result in body["results"]:
        assert set(result) == {"doc_id", "title", "text", "distance"}
        doc_id = result["doc_id"]
        assert doc_id not in emails or "user:kean-s" in emails[doc_id].readers


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"user": "kaminski-v"}, id="user"),
        pytest.param({"principals": ["user:kaminski-v"]}, id="principals"),
        pytest.param({"user_id": "kaminski-v", "groups": ["analyst"]}, id="user-id-and-groups"),
    ],
)
def test_identity_in_the_body_is_rejected(
    client: TestClient, sign_token: Sign, extra: dict[str, object]
) -> None:
    response = client.post("/search", json=QUESTION | extra, headers=_kean(sign_token))
    assert response.status_code == 422
    assert {error["type"] for error in response.json()["detail"]} == {"extra_forbidden"}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-query"),
        pytest.param({"query": ""}, id="empty-query"),
        pytest.param({"query": "x" * 2001}, id="long-query"),
        pytest.param({"query": 42}, id="query-not-a-string"),
        pytest.param(QUESTION | {"k": 0}, id="k-zero"),
        pytest.param(QUESTION | {"k": 21}, id="k-too-large"),
        pytest.param(QUESTION | {"k": "5"}, id="k-a-string"),
    ],
)
def test_invalid_body_is_refused(
    client: TestClient, sign_token: Sign, body: dict[str, object]
) -> None:
    response = client.post("/search", json=body, headers=_kean(sign_token))
    assert response.status_code == 422


def test_ask_without_a_token_is_refused(client: TestClient) -> None:
    response = client.post("/ask", json=ASK)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(ASK | {"user": "kaminski-v"}, id="user"),
        pytest.param(ASK | {"principals": ["user:kaminski-v"]}, id="principals"),
        pytest.param({}, id="no-question"),
        pytest.param({"question": ""}, id="empty-question"),
        pytest.param({"question": "x" * 2001}, id="long-question"),
        pytest.param({"question": 42}, id="question-not-a-string"),
    ],
)
def test_ask_with_an_invalid_body_is_refused(
    client: TestClient, sign_token: Sign, body: dict[str, object]
) -> None:
    response = client.post("/ask", json=body, headers=_kean(sign_token))
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("user", "reply", "expected"),
    [
        pytest.param(
            "kaminski-v",
            f"Two people in his group help [{TIGER_TEAM}].",
            {
                "user": "kaminski-v",
                "answer": f"Two people in his group help [{TIGER_TEAM}].",
                "citations": [TIGER_TEAM],
                "refused": False,
                "queries": [],
                "withheld": 0,
            },
            id="holder",
        ),
        pytest.param(
            "kean-s",
            REFUSAL,
            {
                "user": "kean-s",
                "answer": REFUSAL,
                "citations": [],
                "refused": True,
                "queries": [],
                "withheld": 0,
            },
            id="other-custodian",
        ),
    ],
)
def test_ask_answers_as_the_token_user(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
    user: str,
    reply: str,
    expected: dict[str, object],
) -> None:
    model = scripted(search_call({"query": ASK["question"]}), final_reply(reply))
    token = sign_token({"preferred_username": user, "groups": []})

    with TestClient(create_app(token_settings, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_bearer(token))

    assert response.status_code == 200
    assert response.json() == expected
    search_result = model.prompts[-1][-1].text
    assert (f"[{TIGER_TEAM}]" in search_result) == (user == "kaminski-v")


def test_ask_after_opening_a_document_keeps_the_response_shape(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    reply = f"Two people in his group help [{TIGER_TEAM}]."
    model = scripted(call_tool("open_document", {"doc_id": TIGER_TEAM}), final_reply(reply))
    token = sign_token({"preferred_username": "kaminski-v", "groups": []})

    with TestClient(create_app(token_settings, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_bearer(token))

    assert response.status_code == 200
    assert response.json() == {
        "user": "kaminski-v",
        "answer": reply,
        "citations": [TIGER_TEAM],
        "refused": False,
        "queries": [],
        "withheld": 0,
    }


def test_ask_with_the_budget_spent_is_unavailable(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    model = scripted(final_reply(REFUSAL))
    spent = token_settings.model_copy(update={"budget_usd": 0.0})

    with TestClient(create_app(spent, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_kean(sign_token))

    assert response.status_code == 503
    assert model.prompts == []


def test_ask_with_the_database_down_is_unavailable(
    unreachable_database: None,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    model = scripted(search_call(QUESTION), final_reply(REFUSAL))

    with TestClient(create_app(token_settings, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_kean(sign_token))

    assert response.status_code == 503
    assert response.json() == {"detail": "the database cannot be reached"}


def test_concurrent_searches_each_answer_as_their_own_caller(
    client: TestClient, sign_token: Sign, emails: dict[str, Document]
) -> None:
    callers = {
        "kean-s": _kean(sign_token),
        "kaminski-v": _bearer(sign_token({"preferred_username": "kaminski-v", "groups": []})),
    }

    def search_as(user: str) -> tuple[str, dict[str, object]]:
        response = client.post("/search", json=QUESTION | {"k": 10}, headers=callers[user])
        assert response.status_code == 200
        return user, response.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(search_as, ["kean-s", "kaminski-v"] * 20))

    found: dict[str, set[str]] = {"kean-s": set(), "kaminski-v": set()}
    for user, body in replies:
        assert body["user"] == user
        results = body["results"]
        assert isinstance(results, list)
        for result in results:
            doc_id = result["doc_id"]
            assert doc_id not in emails or f"user:{user}" in emails[doc_id].readers
            found[user].add(doc_id)
    assert len(replies) == 40
    assert TIGER_TEAM in found["kaminski-v"]
    assert TIGER_TEAM not in found["kean-s"]


class _WaitingModel(ScriptedModel):
    thinking: threading.Event = Field(default_factory=threading.Event)
    searched: threading.Event = Field(default_factory=threading.Event)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.thinking.set()
        self.searched.wait(timeout=60)
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_a_search_is_served_while_an_answer_waits_on_the_model(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    model = _WaitingModel(messages=iter([search_call(QUESTION), final_reply(REFUSAL)]))
    settings = token_settings.model_copy(update={"pool_size": 1})
    headers = _kean(sign_token)

    with (
        TestClient(create_app(settings, token_signing_key, model)) as client,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        answer = executor.submit(client.post, "/ask", json=ASK, headers=headers)
        assert model.thinking.wait(timeout=60)
        search = client.post("/search", json=QUESTION, headers=headers)
        model.searched.set()

        assert search.status_code == 200
        assert answer.result().status_code == 200


def test_the_api_settings_hold_no_provisioning_password(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NEEDTOKNOW_ADMIN_USER",
        "NEEDTOKNOW_ADMIN_PASSWORD",
        "NEEDTOKNOW_OWNER_PASSWORD",
        "NEEDTOKNOW_READER_PASSWORD",
    ):
        monkeypatch.setenv(name, "provisioning-secret")
    settings = load_settings()

    assert "provisioning-secret" not in repr(settings)
    assert not hasattr(settings, "admin_password")
    assert not hasattr(settings, "owner_password")
    assert not hasattr(settings, "reader_password")


def test_an_employee_over_the_quota_gets_429(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    model = scripted(final_reply(REFUSAL))
    one_a_minute = token_settings.model_copy(update={"requests_per_minute": 1})

    with TestClient(create_app(one_a_minute, token_signing_key, model)) as ask_client:
        first = ask_client.post("/ask", json=ASK, headers=_kean(sign_token))
        second = ask_client.post("/ask", json=ASK, headers=_kean(sign_token))

    assert first.status_code == 200
    assert second.status_code == 429


def _audit_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "needtoknow.audit"]


def _ask_as_kaminski(
    settings: Settings, signing_key: SigningKeyLookup, token: str
) -> dict[str, Any]:
    reply = f"Two people in his group help [{TIGER_TEAM}]."
    model = scripted(search_call({"query": ASK["question"]}), final_reply(reply))
    with TestClient(create_app(settings, signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_bearer(token))
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def test_an_answer_writes_one_audit_line_without_the_question(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="needtoknow.audit")
    token = sign_token({"preferred_username": "kaminski-v", "groups": []})

    _ask_as_kaminski(token_settings, token_signing_key, token)

    lines = _audit_lines(caplog)
    assert len(lines) == 1
    assert ASK["question"] not in lines[0]
    assert token not in lines[0]
    line = json.loads(lines[0])
    assert set(line) == {
        "time",
        "user",
        "client",
        "entry",
        "text_chars",
        "text_sha256",
        "text",
        "documents",
        "sql",
        "withheld",
        "refused",
        "exhausted",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
    }
    question = ASK["question"]
    assert line["user"] == "kaminski-v"
    assert line["client"] == token_settings.client_id
    assert line["entry"] == "ask"
    assert line["text_chars"] == len(question)
    assert line["text_sha256"] == hashlib.sha256(question.encode()).hexdigest()
    assert line["text"] is None
    assert line["documents"] == [TIGER_TEAM]
    assert line["sql"] == []
    assert line["withheld"] == 0
    assert line["refused"] is False
    assert line["exhausted"] is False
    assert line["input_tokens"] == 2000
    assert line["output_tokens"] == 200
    assert line["cost_usd"] == HAIKU.cost(2000, 200)
    assert line["latency_ms"] >= 0


def test_the_audit_line_holds_the_question_when_the_setting_asks(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="needtoknow.audit")
    with_text = token_settings.model_copy(update={"audit_text": True})
    token = sign_token({"preferred_username": "kaminski-v", "groups": []})

    _ask_as_kaminski(with_text, token_signing_key, token)

    lines = _audit_lines(caplog)
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == ASK["question"]


@pytest.mark.keycloak
def test_ready_when_the_database_and_the_keys_answer(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"database": "ok", "signing_keys": "ok"}


def test_not_ready_with_the_database_down(
    owner: psycopg.Connection[TupleRow],
    unreachable_database: None,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> None:
    with TestClient(create_app(token_settings, token_signing_key, scripted())) as down_client:
        response = down_client.get("/ready")

    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_not_ready_when_the_keys_cannot_be_fetched(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> None:
    no_keys = token_settings.model_copy(update={"jwks_url": "http://127.0.0.1:1/certs"})

    with TestClient(create_app(no_keys, token_signing_key, scripted())) as down_client:
        response = down_client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"database": "ok", "signing_keys": "unreachable"}
