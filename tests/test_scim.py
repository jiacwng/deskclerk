# SCIM offboarding against the real stores: the token gate, creating and finding users, and that
# deactivating or deleting a user erases that employee's data and nobody else's.

import logging
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from deskclerk.api import SESSION_COOKIE
from deskclerk.auth import Identity, SigningKeyLookup
from deskclerk.config import Settings
from test_agent import ScriptedModel, answer, script
from test_api import (
    FakeIdp,
    Sign,
    Web,
    as_user,
    ask,
    begin_sign_in,
    finish_sign_in,
    make_settings,
    web,
)
from test_erase import KEY, rows, seed

TOKEN = "a-scim-token-from-the-identity-provider"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
USERS = "/scim/v2/Users"
PATCH_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SEEDED = (3, 1, 2, 1)
ERASED = (0, 0, 0, 0)


def scim_site(
    token_settings: Settings,
    signing_key: SigningKeyLookup,
    path: Path,
    token: str | None = TOKEN,
) -> AbstractContextManager[Web]:
    secret = None if token is None else SecretStr(token)
    settings = make_settings(token_settings, scim_token=secret)
    return web(settings, [], script(), signing_key, path, key=KEY)


def provision(client: TestClient, user_name: str = "ana", external_id: str = "ana") -> str:
    created = client.post(
        USERS,
        headers=AUTH,
        json={"userName": user_name, "externalId": external_id, "active": True},
    )
    assert created.status_code == 201
    return str(created.json()["id"])


def employee_rows(path: Path, user: str) -> tuple[int, int, int, int]:
    tokens = path / "tokens.sqlite"
    return (
        len(rows(tokens, f"SELECT 1 FROM secrets WHERE user_id = '{user}'")),  # noqa: S608
        len(rows(tokens, f"SELECT 1 FROM sessions WHERE user_id = '{user}'")),  # noqa: S608
        len(rows(tokens, f"SELECT 1 FROM checkpoints WHERE thread_id LIKE '{user}/%'")),  # noqa: S608
        len(rows(path / "spending.sqlite", f"SELECT 1 FROM costs WHERE user_id = '{user}'")),  # noqa: S608
    )


def patch_active(client: TestClient, user_id: str, value: object) -> None:
    operation = {"op": "Replace", "path": "active", "value": value}
    response = client.patch(f"{USERS}/{user_id}", headers=AUTH, json={"Operations": [operation]})
    assert response.status_code == 200


def test_the_endpoints_are_absent_without_the_setting(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    with scim_site(token_settings, token_signing_key, tmp_path, None) as site:
        assert site.client.get(USERS, headers=AUTH).status_code == 404


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer"},
    ],
)
def test_a_missing_or_wrong_token_is_refused_in_the_scim_error_format(
    headers: dict[str, str],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
) -> None:
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        for method, path in [("GET", USERS), ("POST", USERS), ("GET", "/scim/v2/Schemas")]:
            response = site.client.request(method, path, headers=headers, json={})
            assert response.status_code == 401
            assert response.json() == {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:Error"],
                "status": "401",
                "detail": "send Authorization: Bearer <SCIM token>",
            }
            assert response.headers["content-type"].startswith("application/scim+json")


def test_users_are_created_found_by_filter_and_fetched(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        client = site.client
        user_id = provision(client, "Ana@Example.com", "oid-1")
        provision(client, "bob@example.com", "oid-2")

        def listed(filter_text: str) -> Any:
            return client.get(USERS, headers=AUTH, params={"filter": filter_text})

        by_name = listed('userName eq "ana@example.com"')
        by_external = listed('externalId eq "oid-2"')
        nobody = listed('userName eq "nobody"')
        unsupported = listed('name.givenName sw "A"')
        everyone = client.get(USERS, headers=AUTH)
        fetched = client.get(f"{USERS}/{user_id}", headers=AUTH)
        missing = client.get(f"{USERS}/nope", headers=AUTH)

    assert [r["externalId"] for r in by_name.json()["Resources"]] == ["oid-1"]
    assert [r["userName"] for r in by_external.json()["Resources"]] == ["bob@example.com"]
    assert (nobody.json()["totalResults"], nobody.json()["Resources"]) == (0, [])
    assert everyone.json()["totalResults"] == 2
    assert fetched.json()["id"] == user_id
    assert fetched.json()["active"] is True
    assert (missing.status_code, missing.json()["status"]) == (404, "404")
    assert unsupported.status_code == 400
    assert unsupported.json()["scimType"] == "invalidFilter"


def test_a_second_user_with_the_same_name_or_external_id_is_a_conflict(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        provision(site.client, "ana", "oid-1")
        same_name = site.client.post(
            USERS, headers=AUTH, json={"userName": "ANA", "externalId": "oid-2"}
        )
        same_external = site.client.post(
            USERS, headers=AUTH, json={"userName": "x", "externalId": "oid-1"}
        )
        no_external = site.client.post(USERS, headers=AUTH, json={"userName": "y"})
    assert (same_name.status_code, same_external.status_code) == (409, 409)
    assert same_name.json()["scimType"] == "uniqueness"
    assert (no_external.status_code, no_external.json()["scimType"]) == (400, "invalidValue")


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "Replace", "path": "active", "value": "False"},
        {"op": "replace", "path": "active", "value": False},
        {"op": "replace", "value": {"active": False}},
        {"op": "Replace", "value": {"active": "False"}},
    ],
)
def test_deactivating_through_patch_erases_only_that_employee(
    operation: dict[str, Any],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
) -> None:
    seed(tmp_path)
    assert employee_rows(tmp_path, "ana") == SEEDED
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        user_id = provision(site.client)
        response = site.client.patch(
            f"{USERS}/{user_id}",
            headers=AUTH,
            json={"schemas": [PATCH_SCHEMA], "Operations": [operation]},
        )
        assert response.status_code == 200
        assert response.json()["active"] is False
        assert site.client.get(f"{USERS}/{user_id}", headers=AUTH).json()["active"] is False

    assert employee_rows(tmp_path, "ana") == ERASED
    assert employee_rows(tmp_path, "anabel") == SEEDED


def test_deactivating_through_put_erases_the_employee(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    seed(tmp_path)
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        user_id = provision(site.client)
        other_id = site.client.put(
            f"{USERS}/{user_id}", headers=AUTH, json={"userName": "ana", "externalId": "other"}
        )
        assert employee_rows(tmp_path, "ana") == SEEDED
        response = site.client.put(
            f"{USERS}/{user_id}",
            headers=AUTH,
            json={"userName": "ana", "externalId": "ana", "active": False},
        )
    assert other_id.status_code == 400
    assert response.json()["active"] is False
    assert employee_rows(tmp_path, "ana") == ERASED
    assert employee_rows(tmp_path, "anabel") == SEEDED


def test_delete_erases_the_employee_and_removes_the_user(
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    seed(tmp_path)
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        user_id = provision(site.client)
        deleted = site.client.delete(f"{USERS}/{user_id}", headers=AUTH)
        again = site.client.get(f"{USERS}/{user_id}", headers=AUTH)

    assert deleted.status_code == 204
    assert again.status_code == 404
    assert employee_rows(tmp_path, "ana") == ERASED
    assert employee_rows(tmp_path, "anabel") == SEEDED
    lines = [r.getMessage() for r in caplog.records if r.name == "deskclerk.audit"]
    assert len(lines) == 1
    assert '"entry": "erase"' in lines[0]


def test_a_reactivated_user_can_sign_in_again(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    seed(tmp_path)
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        user_id = provision(site.client)
        patch_active(site.client, user_id, "False")
        patch_active(site.client, user_id, "True")
        assert site.client.get(f"{USERS}/{user_id}", headers=AUTH).json()["active"] is True
        session_id = site.sessions.create(Identity("ana", "deskclerk-api", "ana"), 3600)
        site.client.cookies.set(SESSION_COOKIE, session_id)
        assert site.client.get("/me").json() == {"user": "ana", "name": "ana"}


def test_discovery_endpoints_describe_the_user_resource_only(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        config = site.client.get("/scim/v2/ServiceProviderConfig", headers=AUTH).json()
        types = site.client.get("/scim/v2/ResourceTypes", headers=AUTH).json()
        schemas = site.client.get("/scim/v2/Schemas", headers=AUTH).json()
    assert config["patch"] == {"supported": True}
    assert config["bulk"]["supported"] is False
    assert [t["id"] for t in types["Resources"]] == ["User"]
    assert [s["id"] for s in schemas["Resources"]] == ["urn:ietf:params:scim:schemas:core:2.0:User"]


class HeldModel(ScriptedModel):
    entered: threading.Event = threading.Event()
    release: threading.Event = threading.Event()

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.entered.set()
        self.release.wait(10)
        return super()._generate(*args, **kwargs)


def test_a_deactivated_employee_is_refused_by_bearer_and_by_session_until_reactivated(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
) -> None:
    with scim_site(token_settings, token_signing_key, tmp_path) as site:
        user_id = provision(site.client)
        bearer = as_user(sign_token, "ana")
        assert site.client.get("/me", headers=bearer).status_code == 200
        patch_active(site.client, user_id, False)
        refused_bearer = site.client.get("/me", headers=bearer)
        site.client.cookies.set(
            SESSION_COOKIE, site.sessions.create(Identity("ana", "deskclerk-api", "ana"), 3600)
        )
        refused_cookie = site.client.get("/me")
        site.client.cookies.clear()
        other = site.client.get("/me", headers=as_user(sign_token, "anabel"))
        patch_active(site.client, user_id, True)
        allowed = site.client.get("/me", headers=bearer)
    assert (refused_bearer.status_code, refused_cookie.status_code) == (403, 403)
    assert (other.status_code, allowed.status_code) == (200, 200)


def test_a_deactivated_employee_cannot_sign_in(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(token_settings, scim_token=SecretStr(TOKEN))
    idp = FakeIdp(settings.issuer)
    with web(settings, [], script(), token_signing_key, tmp_path, idp.transport, KEY) as site:
        user_id = provision(site.client)
        patch_active(site.client, user_id, False)
        state, _, _ = begin_sign_in(site)
        response = finish_sign_in(site, idp, sign_token, state, {"sub": "ana"})
        assert (response.status_code, response.headers["location"]) == (
            303,
            "/?error=sign_in_refused",
        )
        assert SESSION_COOKIE not in site.client.cookies
    assert "account deactivated" in caplog.text
    assert rows(tmp_path / "tokens.sqlite", "SELECT 1 FROM sessions WHERE user_id = 'ana'") == []


def test_deactivating_waits_for_an_ask_in_flight_and_nothing_is_left_behind(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    held = HeldModel(messages=iter([answer("done")]))
    held.entered, held.release = threading.Event(), threading.Event()
    settings = make_settings(token_settings, scim_token=SecretStr(TOKEN))
    with web(settings, [], held, token_signing_key, tmp_path, key=KEY) as site:
        user_id = provision(site.client)
        asking = threading.Thread(target=ask, args=(site, as_user(sign_token, "ana")))
        asking.start()
        assert held.entered.wait(10)
        deactivating = threading.Thread(target=patch_active, args=(site.client, user_id, False))
        deactivating.start()
        time.sleep(0.5)
        still_waiting = deactivating.is_alive()
        held.release.set()
        asking.join(10)
        deactivating.join(10)
    assert still_waiting
    assert employee_rows(tmp_path, "ana") == ERASED
