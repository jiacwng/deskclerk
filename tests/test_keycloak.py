# Logs in as each user through the running Keycloak and checks that auth.py derives their
# principals from the real token, and that the API searches as the user who holds the token.

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

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

pytestmark = pytest.mark.keycloak

SETTINGS = load_settings()
TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
PASSWORD = "needtoknow"  # noqa: S105, the development password in keycloak/needtoknow-realm.json


def _open(url: str, form: dict[str, str] | None = None) -> bytes:
    data = None if form is None else urllib.parse.urlencode(form).encode()
    # The issuer comes from settings and is plain http in development.
    with urllib.request.urlopen(url, data=data, timeout=10) as response:  # noqa: S310
        body: bytes = response.read()
    return body


def _login(username: str) -> dict[str, Any]:
    form = {
        "grant_type": "password",
        "client_id": SETTINGS.client_id,
        "username": username,
        "password": PASSWORD,
        "scope": "openid",
    }
    tokens: dict[str, Any] = json.loads(
        _open(f"{SETTINGS.issuer}/protocol/openid-connect/token", form)
    )
    return tokens


@pytest.fixture(scope="module")
def signing_key() -> SigningKeyLookup:
    discovery = f"{SETTINGS.issuer}/.well-known/openid-configuration"
    try:
        _open(discovery)
    except urllib.error.URLError as error:
        pytest.fail(
            f"Keycloak does not answer at {discovery} ({error}). "
            "Start it with: docker compose up -d --wait keycloak"
        )
    return jwks_signing_key(SETTINGS)


def test_id_token_is_refused(signing_key: SigningKeyLookup) -> None:
    with pytest.raises(AuthError, match="not an access token"):
        authenticate(_login("kean-s")["id_token"], SETTINGS, signing_key)


def test_search_answers_as_the_token_holder(
    owner: psycopg.Connection[TupleRow], signing_key: SigningKeyLookup
) -> None:
    question = {"query": "Who is helping with the Wharton and Rice projects for the Tiger Team?"}
    with TestClient(create_app(SETTINGS, signing_key, scripted())) as client:
        found: dict[str, list[str]] = {}
        for username in ("kaminski-v", "kean-s"):
            headers = {"Authorization": f"Bearer {_login(username)['access_token']}"}
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
    identity = authenticate(_login(username)["access_token"], SETTINGS, signing_key)
    assert identity.user_id == username
    assert identity.principals == principals(username, groups)
