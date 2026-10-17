# The Keycloak realm holds the users of the public data, each taken from the slices in
# tests/data, a chat page client that signs in with the code flow and PKCE, a test client that
# takes a password, both putting the groups and the API in the access token, and the scope and
# registration policies that let an MCP client register itself and get a token for the server.

import json
from pathlib import Path
from typing import Any

import pytest

from needtoknow.config import Settings
from needtoknow.mcp_server import SCOPE

ROOT = Path(__file__).parent.parent
REALM: dict[str, Any] = json.loads((ROOT / "keycloak" / "needtoknow-realm.json").read_text())
DATA = ROOT / "tests" / "data"
# The three largest Enron mailboxes, a district and a county of the BIRD slice, and an analyst.
DATA_USERS: dict[str, list[str]] = {
    "kaminski-v": [],
    "dasovich-j": [],
    "kean-s": [],
    "district-1": ["district:1"],
    "county-sierra": ["county:Sierra"],
    "analyst": ["analyst"],
}


def test_realm_groups_are_the_groups_of_the_data_users() -> None:
    data_groups = {group for groups in DATA_USERS.values() for group in groups}
    assert sorted(group["name"] for group in REALM["groups"]) == sorted(data_groups)


def test_the_data_users_are_mailboxes_a_district_and_a_county_of_the_slices() -> None:
    realm_users = {
        user["username"]: [group.removeprefix("/") for group in user["groups"]]
        for user in REALM["users"]
    }
    dump = (DATA / "bird" / "MINIDEV_postgresql" / "BIRD_dev.sql").read_text(encoding="utf-8")

    assert realm_users == DATA_USERS
    for custodian in ("kaminski-v", "dasovich-j", "kean-s"):
        assert (DATA / "enron" / "maildir" / custodian).is_dir()
    assert "\n2\t1\tPOPLATEK MESICNE\t" in dump
    assert "\tSierra\t" in dump


def test_every_user_can_log_in() -> None:
    for user in REALM["users"]:
        assert user["enabled"]
        assert [c["type"] for c in user["credentials"]] == ["password"]


def _client(client_id: str) -> dict[str, Any]:
    clients: list[dict[str, Any]] = [c for c in REALM["clients"] if c["clientId"] == client_id]
    assert len(clients) == 1
    return clients[0]


def test_the_chat_page_signs_in_with_the_code_flow_and_pkce_only() -> None:
    page = _client("needtoknow-api")
    assert page["publicClient"]
    assert page["standardFlowEnabled"]
    assert not page["directAccessGrantsEnabled"]
    assert page["attributes"]["pkce.code.challenge.method"] == "S256"
    assert page["redirectUris"] == ["http://localhost:8000/", "http://127.0.0.1:8000/"]


def test_only_the_test_client_takes_a_password() -> None:
    test = _client("needtoknow-test")
    assert test["publicClient"]
    assert test["directAccessGrantsEnabled"]
    assert not test["standardFlowEnabled"]
    password_clients = [c["clientId"] for c in REALM["clients"] if c["directAccessGrantsEnabled"]]
    assert password_clients == ["needtoknow-test"]


@pytest.mark.parametrize("client_id", ["needtoknow-api", "needtoknow-test"])
def test_each_client_puts_the_groups_and_the_api_in_the_access_token(client_id: str) -> None:
    client = _client(client_id)
    [groups] = _mappers(client, "oidc-group-membership-mapper")
    assert groups["claim.name"] == "groups"
    assert groups["full.path"] == "true"
    assert groups["access.token.claim"] == "true"
    [audience] = _mappers(client, "oidc-audience-mapper")
    assert audience["included.client.audience"] == "needtoknow-api"
    assert audience["access.token.claim"] == "true"


def _mappers(scope: dict[str, Any], kind: str) -> list[dict[str, str]]:
    return [m["config"] for m in scope["protocolMappers"] if m["protocolMapper"] == kind]


def test_the_mcp_scope_puts_the_server_url_and_the_groups_in_the_access_token() -> None:
    [scope] = [s for s in REALM["clientScopes"] if s["name"] == SCOPE]
    assert scope["attributes"]["include.in.token.scope"] == "true"

    [audience] = _mappers(scope, "oidc-audience-mapper")
    assert audience["included.custom.audience"] == Settings.model_fields["mcp_url"].default
    assert audience["access.token.claim"] == "true"
    [groups] = _mappers(scope, "oidc-group-membership-mapper")
    assert groups["claim.name"] == "groups"
    assert groups["full.path"] == "true"
    assert groups["access.token.claim"] == "true"


def test_every_client_may_ask_for_the_mcp_scope_and_none_gets_it_unasked() -> None:
    assert REALM["defaultOptionalClientScopes"] == [SCOPE]
    assert SCOPE not in REALM["defaultDefaultClientScopes"]


def test_the_default_scopes_give_the_subject_and_the_username() -> None:
    scopes = {s["name"]: s for s in REALM["clientScopes"]}
    assert set(REALM["defaultDefaultClientScopes"]) <= set(scopes)
    [username] = _mappers(scopes["profile"], "oidc-usermodel-attribute-mapper")
    assert username["claim.name"] == "preferred_username"
    assert _mappers(scopes["basic"], "oidc-sub-mapper")


def test_anonymous_registration_is_limited_to_clients_on_this_machine() -> None:
    components = REALM["components"]
    policies = {
        p["providerId"]: p["config"]
        for p in components[
            "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"
        ]
        if p["subType"] == "anonymous"
    }
    assert set(policies) == {
        "trusted-hosts",
        "max-clients",
        "scope",
        "registration-web-origins",
        "consent-required",
        "allowed-client-templates",
        "allowed-protocol-mappers",
    }
    hosts = policies["trusted-hosts"]
    assert hosts["trusted-hosts"] == ["localhost", "127.0.0.1"]
    assert hosts["host-sending-registration-request-must-match"] == ["true"]
    assert hosts["client-uris-must-match"] == ["true"]
    assert policies["allowed-protocol-mappers"]["allowed-protocol-mapper-types"] == []
    assert policies["allowed-client-templates"]["allow-default-scopes"] == ["true"]


def test_the_login_page_names_the_project_and_warns_it_is_for_development() -> None:
    assert REALM["displayName"] == (
        "needtoknow (development only, every user's password is needtoknow)"
    )
