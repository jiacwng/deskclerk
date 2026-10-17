# Logs in as each user through the running Keycloak and checks that auth.py derives their
# principals from the real token, that the API searches as the user who holds the token, that
# only a token asked for with the MCP scope is good for the MCP server, and that the chat page
# client signs in with the authorization code flow and PKCE and never with a password.

import base64
import hashlib
import html
import re
import secrets
from typing import Any

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from fakes import scripted
from needtoknow.api import create_app
from needtoknow.auth import (
    AuthError,
    SigningKeyLookup,
    authenticate,
    jwks_signing_key,
    principals,
)
from needtoknow.config import load_settings
from needtoknow.mcp_server import SCOPE

pytestmark = pytest.mark.keycloak

SETTINGS = load_settings()
TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
PASSWORD = "needtoknow"  # noqa: S105, the development password in keycloak/needtoknow-realm.json
TEST_CLIENT = "needtoknow-test"
PAGE = "http://localhost:8000/"
TOKEN_URL = f"{SETTINGS.issuer}/protocol/openid-connect/token"


def _login(username: str, scope: str = "openid", client_id: str = TEST_CLIENT) -> httpx.Response:
    form = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": PASSWORD,
        "scope": scope,
    }
    return httpx.post(TOKEN_URL, data=form, timeout=10)


def _tokens(username: str, scope: str = "openid") -> dict[str, Any]:
    response = _login(username, scope)
    response.raise_for_status()
    tokens: dict[str, Any] = response.json()
    return tokens


# Does what the chat page and the browser do: open the login page with a PKCE challenge, post the
# login form, and read the code from the redirect back to the page.
def _authorization_code(browser: httpx.Client, username: str, challenge: str | None) -> str:
    state = secrets.token_urlsafe(16)
    query = {
        "response_type": "code",
        "client_id": SETTINGS.client_id,
        "redirect_uri": PAGE,
        "scope": "openid",
        "state": state,
    }
    if challenge is not None:
        query |= {"code_challenge": challenge, "code_challenge_method": "S256"}
    page = browser.get(f"{SETTINGS.issuer}/protocol/openid-connect/auth", params=query)
    page.raise_for_status()
    form = re.search(r'<form[^>]*\baction="([^"]+)"', page.text)
    assert form is not None, page.text
    # Keycloak marks its login cookies Secure. A browser sends them to http://localhost anyway,
    # httpx does not, so they go in the header by hand.
    cookies = "; ".join(f"{cookie.name}={cookie.value}" for cookie in page.cookies.jar)
    sent = browser.post(
        html.unescape(form[1]),
        data={"username": username, "password": PASSWORD},
        headers={"Cookie": cookies},
    )
    assert sent.status_code == 302, sent.text
    back = httpx.URL(sent.headers["location"])
    assert str(back.copy_with(query=None)) == PAGE
    assert back.params["state"] == state
    return back.params["code"]


def _code_flow(username: str) -> dict[str, Any]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    with httpx.Client(timeout=10) as browser:
        code = _authorization_code(browser, username, challenge)
        form = {
            "grant_type": "authorization_code",
            "client_id": SETTINGS.client_id,
            "code": code,
            "redirect_uri": PAGE,
            "code_verifier": verifier,
        }
        response = browser.post(TOKEN_URL, data=form)
    response.raise_for_status()
    tokens: dict[str, Any] = response.json()
    return tokens


@pytest.fixture(scope="module")
def signing_key() -> SigningKeyLookup:
    discovery = f"{SETTINGS.issuer}/.well-known/openid-configuration"
    try:
        httpx.get(discovery, timeout=10).raise_for_status()
    except httpx.HTTPError as error:
        pytest.fail(
            f"Keycloak does not answer at {discovery} ({error}). "
            "Start it with: docker compose up -d --wait keycloak"
        )
    return jwks_signing_key(SETTINGS)


def test_the_chat_page_signs_in_with_the_code_flow_and_pkce(
    signing_key: SigningKeyLookup,
) -> None:
    tokens = _code_flow("district-1")
    identity = authenticate(tokens["access_token"], SETTINGS, signing_key, SETTINGS.audience)
    assert identity.principals == principals("district-1", ["district:1"])


def test_the_id_token_of_the_chat_page_is_refused(signing_key: SigningKeyLookup) -> None:
    id_token = _code_flow("kean-s")["id_token"]
    with pytest.raises(AuthError, match="not an access token: typ is 'ID'"):
        authenticate(id_token, SETTINGS, signing_key, SETTINGS.audience)


def test_a_code_without_its_verifier_is_refused() -> None:
    challenge = base64.urlsafe_b64encode(hashlib.sha256(b"x" * 43).digest()).rstrip(b"=")
    with httpx.Client(timeout=10) as browser:
        code = _authorization_code(browser, "kean-s", challenge.decode())
        form = {
            "grant_type": "authorization_code",
            "client_id": SETTINGS.client_id,
            "code": code,
            "redirect_uri": PAGE,
        }
        response = browser.post(TOKEN_URL, data=form)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_grant"


def test_a_sign_in_without_pkce_is_refused() -> None:
    query = {
        "response_type": "code",
        "client_id": SETTINGS.client_id,
        "redirect_uri": PAGE,
        "scope": "openid",
        "state": "s",
    }
    with httpx.Client(timeout=10) as browser:
        response = browser.get(f"{SETTINGS.issuer}/protocol/openid-connect/auth", params=query)
    assert response.status_code == 302
    back = httpx.URL(response.headers["location"])
    assert back.params["error"] == "invalid_request"
    assert "code_challenge" in back.params["error_description"]


def test_the_chat_page_client_takes_no_password() -> None:
    response = _login("kean-s", client_id=SETTINGS.client_id)
    assert response.status_code in (400, 401)
    assert response.json()["error"] == "unauthorized_client"


def test_search_answers_as_the_token_holder(
    owner: psycopg.Connection[TupleRow], signing_key: SigningKeyLookup
) -> None:
    question = {"query": "Who is helping with the Wharton and Rice projects for the Tiger Team?"}
    with TestClient(create_app(SETTINGS, signing_key, scripted())) as client:
        found: dict[str, list[str]] = {}
        for username in ("kaminski-v", "kean-s"):
            headers = {"Authorization": f"Bearer {_tokens(username)['access_token']}"}
            response = client.post("/search", json=question, headers=headers)
            assert response.status_code == 200
            assert response.json()["user"] == username
            found[username] = [result["doc_id"] for result in response.json()["results"]]

    assert TIGER_TEAM in found["kaminski-v"]
    assert TIGER_TEAM not in found["kean-s"]


@pytest.mark.parametrize(
    ("username", "groups"),
    [
        ("kaminski-v", []),
        ("dasovich-j", []),
        ("kean-s", []),
        ("district-1", ["district:1"]),
        ("county-sierra", ["county:Sierra"]),
        ("analyst", ["analyst"]),
    ],
)
def test_token_gives_each_user_their_principals(
    username: str, groups: list[str], signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(
        _tokens(username)["access_token"], SETTINGS, signing_key, SETTINGS.audience
    )
    assert identity.user_id == username
    assert identity.principals == principals(username, groups)


def test_only_a_token_asked_for_with_the_mcp_scope_is_good_for_the_mcp_server(
    signing_key: SigningKeyLookup,
) -> None:
    plain = _tokens("district-1")["access_token"]
    with pytest.raises(AuthError, match="Audience doesn't match"):
        authenticate(plain, SETTINGS, signing_key, SETTINGS.mcp_url)

    token = _tokens("district-1", f"openid {SCOPE}")["access_token"]
    identity = authenticate(token, SETTINGS, signing_key, SETTINGS.mcp_url)
    assert identity.principals == principals("district-1", ["district:1"])
    assert SCOPE in identity.scopes
