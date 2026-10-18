# The production stack in compose.prod.yaml and its realm template: only Caddy is public and
# Keycloak's admin console is on the loopback address, Keycloak runs in production mode, every
# password is required from the environment, the app and the MCP server get only the settings
# they read and none of the admin, owner or bootstrap passwords, the realm holds no users and no
# password client, and the addresses the app checks tokens against are the ones Keycloak stamps
# into them.

import json
import re
from pathlib import Path
from typing import Any

import yaml

from needtoknow.mcp_server import SCOPE

ROOT = Path(__file__).parent.parent
COMPOSE: dict[str, Any] = yaml.safe_load((ROOT / "compose.prod.yaml").read_text())
SERVICES: dict[str, Any] = COMPOSE["services"]
REALM: dict[str, Any] = json.loads(
    (ROOT / "deploy" / "realm" / "needtoknow-realm.json").read_text()
)
REFERENCE = re.compile(r"^\$\{(\w+)(:\?[^}]*)?\}$")


def _environment(service: str) -> dict[str, str]:
    environment: dict[str, str] = SERVICES[service]["environment"]
    return environment


def test_only_caddy_is_public_and_keycloak_is_on_the_loopback_address() -> None:
    published = [name for name, service in SERVICES.items() if "ports" in service]
    assert published == ["caddy", "keycloak"]
    assert SERVICES["caddy"]["ports"] == ["80:80", "443:443", "443:443/udp"]
    assert SERVICES["keycloak"]["ports"] == ["127.0.0.1:8080:8080"]
    assert _environment("keycloak")["KC_HOSTNAME_ADMIN"] == "http://localhost:8080/auth"


def test_keycloak_runs_in_production_mode_behind_the_proxy() -> None:
    keycloak = SERVICES["keycloak"]
    environment = _environment("keycloak")
    assert keycloak["command"] == ["start", "--import-realm"]
    assert environment["KC_HOSTNAME"] == "https://${NEEDTOKNOW_DOMAIN}/auth"
    assert environment["KC_HTTP_RELATIVE_PATH"] == "/auth"
    assert environment["KC_PROXY_HEADERS"] == "xforwarded"
    assert environment["KC_DB_URL"] == "jdbc:postgresql://db:5432/keycloak"
    assert keycloak["volumes"] == ["./deploy/realm:/opt/keycloak/data/import:ro"]


def test_every_password_is_required_from_the_environment() -> None:
    required = set(re.findall(r"\$\{(\w+):\?", (ROOT / "compose.prod.yaml").read_text()))
    passwords = 0
    for name in SERVICES:
        for variable, value in SERVICES[name].get("environment", {}).items():
            if not variable.endswith("PASSWORD"):
                continue
            reference = REFERENCE.match(value)
            assert reference, f"{name}: {variable} is {value!r}"
            assert reference.group(1) in required
            passwords += 1
    assert passwords == 9


def test_the_app_and_the_mcp_server_get_no_admin_owner_or_bootstrap_password() -> None:
    forbidden = ("NEEDTOKNOW_ADMIN_PASSWORD", "NEEDTOKNOW_OWNER_PASSWORD", "KC_")
    for name in ("app", "mcp"):
        assert "env_file" not in SERVICES[name]
        for variable in _environment(name):
            assert not variable.startswith(forbidden)
        public = _environment(name)["NEEDTOKNOW_PUBLIC_RELATIONS"]
        assert public == "${NEEDTOKNOW_PUBLIC_RELATIONS:-}"
    assert "ANTHROPIC_API_KEY" in _environment("app")
    assert "ANTHROPIC_API_KEY" not in _environment("mcp")


def test_caddy_serves_no_keycloak_admin_path() -> None:
    caddyfile = (ROOT / "deploy" / "Caddyfile").read_text()
    assert "@keycloak path /auth/realms/needtoknow/* /auth/resources/*" in caddyfile
    assert "handle /auth/* {\n\t\trespond 404\n\t}" in caddyfile


def test_the_app_and_the_mcp_server_trust_the_tokens_keycloak_issues() -> None:
    issuer = _environment("keycloak")["KC_HOSTNAME"] + "/realms/needtoknow"
    for name in ("app", "mcp"):
        environment = _environment(name)
        assert environment["NEEDTOKNOW_DB_HOST"] == "db"
        assert environment["NEEDTOKNOW_ISSUER"] == issuer
        assert environment["NEEDTOKNOW_JWKS_URL"].startswith("http://keycloak:8080/auth/realms/")
        assert environment["NEEDTOKNOW_MCP_URL"] == "https://${NEEDTOKNOW_DOMAIN}/mcp"


def test_the_realm_holds_no_users_and_no_password_client() -> None:
    assert "users" not in REALM
    assert "groups" not in REALM
    assert [client["clientId"] for client in REALM["clients"]] == ["needtoknow-api"]


def test_the_chat_page_signs_in_on_the_domain_with_the_code_flow_and_pkce() -> None:
    [page] = REALM["clients"]
    assert page["publicClient"]
    assert page["standardFlowEnabled"]
    assert not page["directAccessGrantsEnabled"]
    assert page["attributes"]["pkce.code.challenge.method"] == "S256"
    assert page["redirectUris"] == ["https://${NEEDTOKNOW_DOMAIN}/"]
    assert page["webOrigins"] == ["https://${NEEDTOKNOW_DOMAIN}"]


def test_the_mcp_scope_puts_the_public_mcp_url_in_the_audience() -> None:
    [scope] = [s for s in REALM["clientScopes"] if s["name"] == SCOPE]
    [audience] = [m["config"] for m in scope["protocolMappers"] if m["name"] == "audience"]
    assert audience["included.custom.audience"] == _environment("mcp")["NEEDTOKNOW_MCP_URL"]
    assert REALM["defaultOptionalClientScopes"] == [SCOPE]


def test_any_host_may_register_a_client_that_redirects_to_its_own_machine() -> None:
    policies = REALM["components"][
        "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"
    ]
    [hosts] = [p["config"] for p in policies if p["providerId"] == "trusted-hosts"]
    assert hosts["trusted-hosts"] == ["localhost", "127.0.0.1"]
    assert hosts["host-sending-registration-request-must-match"] == ["false"]
    assert hosts["client-uris-must-match"] == ["true"]
    [limit] = [p["config"] for p in policies if p["providerId"] == "max-clients"]
    assert limit["max-clients"] == ["50"]


def test_the_groups_claim_carries_each_group_s_full_path() -> None:
    scopes = [REALM["clients"][0], *REALM["clientScopes"]]
    mappers = []
    for scope in scopes:
        for mapper in scope.get("protocolMappers", []):
            if mapper["protocolMapper"] == "oidc-group-membership-mapper":
                mappers.append(mapper["config"])
    assert len(mappers) == 2
    assert all(mapper["full.path"] == "true" for mapper in mappers)
