# Checks the running docker compose stack over HTTP as three shipped employees and as nobody:
# each is served their own documents, a document held by one is never returned to another, and a
# request without a valid token is refused. Needs no model key. Run with: python -m pytest e2e

import httpx
import pytest

API = "http://localhost:8000"
TOKEN_URL = "http://localhost:8080/realms/needtoknow/protocol/openid-connect/token"
PASSWORD = "needtoknow"
QUESTION = "Tiger Team Wharton Rice"
KAMINSKI_ONLY = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
DASOVICH_ONLY = "enron-ef92b6783911316710bdd5c88c4291f4c751a73c3125043bc3f9464530c01e3e"


def _search(username: str) -> list[str]:
    form = {
        "grant_type": "password",
        "client_id": "needtoknow-test",
        "username": username,
        "password": PASSWORD,
        "scope": "openid",
    }
    login = httpx.post(TOKEN_URL, data=form, timeout=10)
    login.raise_for_status()
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    response = httpx.post(
        f"{API}/search", json={"query": QUESTION, "k": 20}, headers=headers, timeout=30
    )
    assert response.status_code == 200
    assert response.json()["user"] == username
    return [result["doc_id"] for result in response.json()["results"]]


def test_each_employee_is_served_their_own_documents() -> None:
    assert KAMINSKI_ONLY in _search("kaminski-v")
    assert DASOVICH_ONLY in _search("dasovich-j")


def test_a_document_held_by_one_employee_is_never_returned_to_another() -> None:
    assert DASOVICH_ONLY not in _search("kaminski-v")
    assert KAMINSKI_ONLY not in _search("dasovich-j")
    assert KAMINSKI_ONLY not in _search("kean-s")
    assert DASOVICH_ONLY not in _search("kean-s")


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}])
def test_a_request_without_a_valid_token_is_refused(headers: dict[str, str]) -> None:
    response = httpx.post(f"{API}/search", json={"query": QUESTION}, headers=headers, timeout=10)
    assert response.status_code == 401
