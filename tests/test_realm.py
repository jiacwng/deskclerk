# The Keycloak realm holds the users of the public data, each taken from the slices in
# tests/data, and a client that puts their groups and itself in the access token.

import json
from pathlib import Path
from typing import Any

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


def test_client_puts_group_names_in_the_access_token() -> None:
    clients = [c for c in REALM["clients"] if c["clientId"] == "needtoknow-api"]
    assert len(clients) == 1
    client = clients[0]
    assert client["publicClient"]
    assert client["directAccessGrantsEnabled"]

    mappers = [
        m
        for m in client["protocolMappers"]
        if m["protocolMapper"] == "oidc-group-membership-mapper"
    ]
    assert len(mappers) == 1
    config = mappers[0]["config"]
    assert config["claim.name"] == "groups"
    assert config["full.path"] == "false"
    assert config["access.token.claim"] == "true"


def test_client_puts_itself_in_the_access_token_audience() -> None:
    client = next(c for c in REALM["clients"] if c["clientId"] == "needtoknow-api")
    mappers = [
        m for m in client["protocolMappers"] if m["protocolMapper"] == "oidc-audience-mapper"
    ]
    assert len(mappers) == 1
    config = mappers[0]["config"]
    assert config["included.client.audience"] == "needtoknow-api"
    assert config["access.token.claim"] == "true"
