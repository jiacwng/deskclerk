# Checks the production stack of compose.prod.yaml, started from an empty checkout by
# e2e/run_prod.sh, over HTTPS through Caddy as an operator would use it: two groups and two users
# are created in the empty needtoknow realm through Keycloak's admin REST API, a folder of
# Enron emails is loaded with an access file that gives each group its own mailbox, each user
# searches and is served only that group's files, a request without a valid token is refused,
# and a file deleted from the folder is gone after the next load. Skipped unless the script sets
# E2E_PROD_DIR. Run with: bash e2e/run_prod.sh

import os
import re
import secrets
import ssl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.skipif("E2E_PROD_DIR" not in os.environ, reason="run e2e/run_prod.sh")

KEYCLOAK_ADMIN = "http://127.0.0.1:8080/auth"
REALM_ADMIN = f"{KEYCLOAK_ADMIN}/admin/realms/needtoknow"
CLIENT_ID = "operator-test"
MAILBOXES = ["dasovich-j", "kaminski-v"]
LOAD = [
    "files",
    "/data/files",
    "--access",
    "/data/access.jsonl",
    "--delete-missing",
]


@dataclass(frozen=True)
class Company:
    folder: Path
    secret: str
    password: str
    tls: ssl.SSLContext
    domain: str


def _subject(path: Path) -> str:
    match = re.search(r"^Subject: (.+)$", path.read_text(errors="replace"), re.MULTILINE)
    return match.group(1).strip() if match else ""


def _unique_file(mailbox: Path) -> Path:
    subjects = {path: _subject(path) for path in sorted(mailbox.glob("*.txt"))}
    for path, subject in subjects.items():
        if subject and list(subjects.values()).count(subject) == 1:
            return path
    raise AssertionError(f"{mailbox} holds no file with a subject of its own")


def _admin(method: str, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
    form = {
        "grant_type": "password",
        "client_id": "admin-cli",
        "username": os.environ["KC_BOOTSTRAP_ADMIN_USERNAME"],
        "password": os.environ["KC_BOOTSTRAP_ADMIN_PASSWORD"],
    }
    login = httpx.post(
        f"{KEYCLOAK_ADMIN}/realms/master/protocol/openid-connect/token", data=form, timeout=30
    )
    login.raise_for_status()
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    response = httpx.request(method, f"{REALM_ADMIN}{path}", headers=headers, json=json, timeout=30)
    response.raise_for_status()
    return response


def _created_id(response: httpx.Response) -> str:
    return response.headers["Location"].rsplit("/", 1)[1]


def _load(company: Company) -> str:
    command = [
        "docker",
        "compose",
        "-f",
        "compose.prod.yaml",
        "run",
        "--rm",
        "load",
        *LOAD,
    ]
    done = subprocess.run(
        command, cwd=company.folder, capture_output=True, text=True, timeout=900, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def _token(company: Company, username: str) -> str:
    form = {
        "grant_type": "password",
        "client_id": CLIENT_ID,
        "client_secret": company.secret,
        "username": username,
        "password": company.password,
    }
    response = httpx.post(
        f"https://{company.domain}/auth/realms/needtoknow/protocol/openid-connect/token",
        data=form,
        verify=company.tls,
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def _search(company: Company, username: str, query: str) -> list[str]:
    headers = {"Authorization": f"Bearer {_token(company, username)}"}
    response = httpx.post(
        f"https://{company.domain}/search",
        json={"query": query, "k": 20},
        headers=headers,
        verify=company.tls,
        timeout=60,
    )
    assert response.status_code == 200
    assert response.json()["user"] == username
    return [result["doc_id"] for result in response.json()["results"]]


@pytest.fixture(scope="module")
def company() -> Company:
    folder = Path(os.environ["E2E_PROD_DIR"])
    secret = secrets.token_hex(16)
    password = secrets.token_hex(16)
    for mailbox, username in zip(MAILBOXES, ["alice", "bob"], strict=True):
        group = _created_id(_admin("POST", "/groups", json={"name": mailbox}))
        user = _admin(
            "POST",
            "/users",
            json={
                "username": username,
                "enabled": True,
                "emailVerified": True,
                "email": f"{username}@example.com",
                "firstName": username,
                "lastName": "Tester",
                "credentials": [{"type": "password", "value": password, "temporary": False}],
            },
        )
        _admin("PUT", f"/users/{_created_id(user)}/groups/{group}")
    mappers = [
        {
            "name": "groups",
            "protocol": "openid-connect",
            "protocolMapper": "oidc-group-membership-mapper",
            "config": {"claim.name": "groups", "full.path": "true", "access.token.claim": "true"},
        },
        {
            "name": "audience",
            "protocol": "openid-connect",
            "protocolMapper": "oidc-audience-mapper",
            "config": {"included.client.audience": "needtoknow-api", "access.token.claim": "true"},
        },
    ]
    _admin(
        "POST",
        "/clients",
        json={
            "clientId": CLIENT_ID,
            "secret": secret,
            "publicClient": False,
            "standardFlowEnabled": False,
            "directAccessGrantsEnabled": True,
            "protocolMappers": mappers,
        },
    )
    company = Company(
        folder=folder,
        secret=secret,
        password=password,
        tls=ssl.create_default_context(cafile=os.environ["E2E_CA"]),
        domain=os.environ["NEEDTOKNOW_DOMAIN"],
    )
    loaded = _load(company)
    assert "deleted 0" in loaded
    return company


def _own_file(company: Company, mailbox: str) -> Path:
    return _unique_file(company.folder / "documents" / "files" / mailbox)


def test_each_user_is_served_only_the_files_of_their_group(company: Company) -> None:
    for username, mine, other in [("alice", *MAILBOXES), ("bob", *reversed(MAILBOXES))]:
        file = _own_file(company, mine)
        found = _search(company, username, _subject(file))
        assert f"file-{mine}/{file.name}" in found
        assert all(doc_id.startswith(f"file-{mine}/") for doc_id in found)
        theirs = _search(company, username, _subject(_own_file(company, other)))
        assert not [doc_id for doc_id in theirs if doc_id.startswith(f"file-{other}/")]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}])
def test_a_request_without_a_valid_token_is_refused(
    company: Company, headers: dict[str, str]
) -> None:
    response = httpx.post(
        f"https://{company.domain}/search",
        json={"query": "meeting"},
        headers=headers,
        verify=company.tls,
        timeout=30,
    )
    assert response.status_code == 401


def test_a_file_deleted_from_the_folder_is_gone_after_the_next_load(company: Company) -> None:
    removed = _own_file(company, MAILBOXES[0])
    subject = _subject(removed)
    assert f"file-{MAILBOXES[0]}/{removed.name}" in _search(company, "alice", subject)
    removed.unlink()
    assert "deleted 1" in _load(company)
    assert f"file-{MAILBOXES[0]}/{removed.name}" not in _search(company, "alice", subject)
    assert _search(company, "alice", _subject(_own_file(company, MAILBOXES[0])))
