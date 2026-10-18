# auth.py turns a signed access token into principals and refuses every token it should.
# Tokens are signed with the throwaway RSA key from conftest.py, so no Keycloak server is needed.

import time
from collections.abc import Callable

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from conftest import TOKEN_KEY
from needtoknow.auth import (
    AuthError,
    AuthUnavailable,
    SigningKeyLookup,
    authenticate,
    discover_jwks_url,
    jwks_signing_key,
)
from needtoknow.config import Settings

Sign = Callable[[dict[str, object]], str]
OTHER_SERVICE = "https://agent.example.com"


def test_valid_token_gives_user_groups_and_everyone(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(
        sign_token({}), token_settings, token_signing_key, token_settings.audience
    )

    assert identity.user_id == "district-1"
    assert identity.principals == {"user:district-1", "group:everyone", "group:district:1"}


def test_keycloak_group_paths_keep_same_named_subgroups_apart(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = sign_token({"groups": ["/district:1", "/sales/managers", "/support/managers"]})

    identity = authenticate(token, token_settings, token_signing_key, token_settings.audience)

    assert identity.principals == {
        "user:district-1",
        "group:everyone",
        "group:district:1",
        "group:sales/managers",
        "group:support/managers",
    }


def test_missing_groups_gives_user_and_everyone(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(
        sign_token({"groups": None}), token_settings, token_signing_key, token_settings.audience
    )
    assert identity.principals == {"user:district-1", "group:everyone"}


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        pytest.param({"exp": -60}, "expired", id="expired"),
        pytest.param({"exp": None}, "exp", id="no-expiry"),
        pytest.param({"iat": 60}, "not yet valid", id="issued-in-the-future"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, "issuer", id="wrong-issuer"),
        pytest.param({"aud": None}, '"aud"', id="no-audience"),
        pytest.param({"aud": "account"}, "Audience doesn't match", id="wrong-audience"),
        pytest.param({"typ": "ID"}, "typ is 'ID'", id="keycloak-id-token"),
        pytest.param({"token_use": "id"}, "token_use is 'id'", id="cognito-id-token"),
        pytest.param({"scope": None}, "grants no scope", id="no-scope"),
        pytest.param({"nbf": 60}, "not yet valid", id="not-before-in-the-future"),
        pytest.param({"preferred_username": None}, "preferred_username", id="no-username"),
        pytest.param({"preferred_username": "a,b"}, "comma", id="comma-in-username"),
        pytest.param({"groups": ["district:1,analyst"]}, "comma", id="comma-in-group"),
        pytest.param({"groups": "district:1"}, "groups must be a list", id="groups-not-a-list"),
        pytest.param({"groups": ["everyone"]}, "named 'everyone'", id="everyone-group"),
        pytest.param({"groups": ["/everyone"]}, "named 'everyone'", id="everyone-group-path"),
        pytest.param({"scope": 7}, "scope must be a string", id="scope-not-text"),
    ],
)
def test_bad_claims_are_refused(
    claims: dict[str, object],
    reason: str,
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> None:
    now = int(time.time())
    for name in ("exp", "iat", "nbf"):
        offset = claims.get(name)
        if isinstance(offset, int):
            claims = claims | {name: now + offset}
    with pytest.raises(AuthError, match=reason):
        authenticate(sign_token(claims), token_settings, token_signing_key, token_settings.audience)


def test_audience_list_naming_the_api_is_accepted(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = sign_token({"aud": ["account", token_settings.audience]})
    identity = authenticate(token, token_settings, token_signing_key, token_settings.audience)
    assert identity.user_id == "district-1"


def test_a_token_for_another_service_is_refused(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = sign_token({"aud": OTHER_SERVICE})
    with pytest.raises(AuthError, match="Audience doesn't match"):
        authenticate(token, token_settings, token_signing_key, token_settings.audience)


def test_another_audience_accepts_any_registered_client_and_reads_the_scopes(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    claims: dict[str, object] = {
        "aud": OTHER_SERVICE,
        "azp": "a-registered-client",
        "scope": "openid profile needtoknow-agent",
    }
    identity = authenticate(sign_token(claims), token_settings, token_signing_key, OTHER_SERVICE)
    assert identity.user_id == "district-1"
    assert identity.scopes == {"openid", "profile", "needtoknow-agent"}
    assert identity.client_id == "a-registered-client"


def test_a_scope_list_with_a_number_is_refused(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    with pytest.raises(AuthError, match="scope must be a string or a list of strings"):
        authenticate(
            sign_token({"scope": ["openid", 7]}),
            token_settings,
            token_signing_key,
            token_settings.audience,
        )


def test_clock_skew_within_thirty_seconds_is_accepted(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    now = int(time.time())
    issued_ahead = sign_token({"iat": now + 10})
    expired_just_now = sign_token({"iat": now - 300, "exp": now - 10})

    for token in (issued_ahead, expired_just_now):
        identity = authenticate(token, token_settings, token_signing_key, token_settings.audience)
        assert identity.user_id == "district-1"


def test_real_key_client_reports_a_closed_port_as_unavailable(
    sign_token: Sign, token_settings: Settings
) -> None:
    # Nothing listens on port 9, so the connection is refused at once.
    closed = token_settings.model_copy(update={"issuer": "http://127.0.0.1:9/realms/needtoknow"})
    lookup = jwks_signing_key(f"{closed.issuer}/protocol/openid-connect/certs")
    with pytest.raises(AuthUnavailable):
        authenticate(sign_token({}), closed, lookup, token_settings.audience)


def _discovery(requests: list[str], body: dict[str, object]) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(answer))


def test_the_key_set_address_comes_from_the_issuer_s_openid_configuration(
    token_settings: Settings,
) -> None:
    okta = token_settings.model_copy(
        update={"issuer": "https://company.okta.com/oauth2/default", "jwks_url": None}
    )
    requests: list[str] = []
    client = _discovery(requests, {"jwks_uri": "https://company.okta.com/oauth2/default/v1/keys"})

    assert discover_jwks_url(okta, client) == "https://company.okta.com/oauth2/default/v1/keys"
    assert requests == ["https://company.okta.com/oauth2/default/.well-known/openid-configuration"]


def test_a_configured_key_set_address_is_used_without_asking_the_issuer(
    token_settings: Settings,
) -> None:
    requests: list[str] = []
    client = _discovery(requests, {"jwks_uri": "https://elsewhere.test/keys"})

    assert discover_jwks_url(token_settings, client) == token_settings.jwks_url
    assert requests == []


def test_an_openid_configuration_without_jwks_uri_is_unavailable(token_settings: Settings) -> None:
    unset = token_settings.model_copy(update={"jwks_url": None})

    with pytest.raises(AuthUnavailable, match="gives no jwks_uri"):
        discover_jwks_url(unset, _discovery([], {"jwks_uri": 7}))
    with pytest.raises(AuthUnavailable, match="cannot read jwks_uri"):
        discover_jwks_url(unset, _discovery([], {"issuer": unset.issuer}))


def test_bad_signature_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode({"preferred_username": "district-1"}, other_key, algorithm="RS256")
    with pytest.raises(AuthError, match="Signature verification failed"):
        authenticate(token, token_settings, token_signing_key, token_settings.audience)


def test_symmetric_algorithm_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = jwt.encode(
        {"preferred_username": "district-1"}, "a-shared-secret-of-32-bytes-long", "HS256"
    )
    with pytest.raises(AuthError, match="invalid token"):
        authenticate(token, token_settings, token_signing_key, token_settings.audience)


def test_garbage_is_refused(token_settings: Settings, token_signing_key: SigningKeyLookup) -> None:
    with pytest.raises(AuthError, match="invalid token"):
        authenticate("not-a-token", token_settings, token_signing_key, token_settings.audience)


def test_made_up_key_ids_fetch_the_key_set_at_most_once_a_minute(
    monkeypatch: pytest.MonkeyPatch, token_settings: Settings
) -> None:
    key = RSAAlgorithm.to_jwk(TOKEN_KEY.public_key(), as_dict=True) | {"kid": "real"}
    fetches: list[str] = []

    def fetch(client: jwt.PyJWKClient) -> dict[str, object]:
        fetches.append(client.uri)
        return {"keys": [key]}

    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch)
    lookup = jwks_signing_key(f"{token_settings.issuer}/protocol/openid-connect/certs")
    claims = {"sub": "x"}

    for forged in ("forged-1", "forged-2", "forged-3"):
        token = jwt.encode(claims, TOKEN_KEY, algorithm="RS256", headers={"kid": forged})
        with pytest.raises(jwt.PyJWTError):
            lookup(token)
    real = jwt.encode(claims, TOKEN_KEY, algorithm="RS256", headers={"kid": "real"})

    assert lookup(real).key_id == "real"
    assert len(fetches) == 2


def _company_settings(
    token_settings: Settings, issuer: str, audience: str, user_claim: str
) -> Settings:
    return token_settings.model_copy(
        update={"issuer": issuer, "audience": audience, "user_claim": user_claim}
    )


ENTRA_TENANT = "72f988bf-86f1-41af-91ab-2d7cd011db47"
ENTRA_API = "6e74172b-be56-4843-9ff4-e66a39bb12e3"
ENTRA_ISSUER = f"https://login.microsoftonline.com/{ENTRA_TENANT}/v2.0"
ENTRA_USER = "d1ad9ce7-b322-4221-ab74-1e1011e1bbcb"
ENTRA_GROUP = "f0b6e7d1-0d94-4b5c-a1a6-5c0b5e7d3f21"


# The claims of an Entra ID v2.0 access token, as Microsoft documents them.
def entra_claims() -> dict[str, object]:
    now = int(time.time())
    return {
        "aud": ENTRA_API,
        "iss": ENTRA_ISSUER,
        "iat": now,
        "nbf": now,
        "exp": now + 3600,
        "azp": "c44b4083-3bb0-49c1-b47d-974e53cbdf3c",
        "azpacr": "0",
        "name": "Abe Lincoln",
        "oid": ENTRA_USER,
        "preferred_username": "abeli@contoso.com",
        "scp": "access_as_user",
        "sub": "HKZpfaHyWadeOouYlitjrI-KffTm222X5rrV3xDqfKQ",
        "tid": ENTRA_TENANT,
        "ver": "2.0",
        "groups": [ENTRA_GROUP],
    }


def entra_settings(token_settings: Settings) -> Settings:
    return _company_settings(token_settings, ENTRA_ISSUER, ENTRA_API, "oid")


def test_an_entra_access_token_names_the_user_by_object_id(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = entra_settings(token_settings)
    token = jwt.encode(entra_claims(), TOKEN_KEY, algorithm="RS256")

    identity = authenticate(token, settings, token_signing_key, settings.audience)

    assert identity.user_id == ENTRA_USER
    assert identity.principals == {f"user:{ENTRA_USER}", "group:everyone", f"group:{ENTRA_GROUP}"}
    assert identity.scopes == {"access_as_user"}


def test_an_entra_id_token_for_the_same_app_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = entra_settings(token_settings)
    claims = entra_claims()
    del claims["scp"]
    token = jwt.encode(claims | {"nonce": "123523"}, TOKEN_KEY, algorithm="RS256")

    with pytest.raises(AuthError, match="not an access token: it grants no scope"):
        authenticate(token, settings, token_signing_key, settings.audience)


OKTA_ISSUER = "https://company.okta.com/oauth2/default"


def test_an_okta_access_token_names_the_user_by_subject(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = _company_settings(token_settings, OKTA_ISSUER, "api://default", "sub")
    now = int(time.time())
    claims = {
        "ver": 1,
        "jti": "AT.0mP4JKAZX1iACIT4vbEDF7LpvDVjxypPMf0D7uX39RE",
        "iss": OKTA_ISSUER,
        "aud": "api://default",
        "iat": now,
        "exp": now + 3600,
        "cid": "0oaoesxtxmPf08QHk0h7",
        "uid": "00uoesxtxlCnQyMBF0h7",
        "scp": ["openid", "profile"],
        "auth_time": now,
        "sub": "alice@company.com",
        "groups": ["Everyone", "Finance"],
    }
    token = jwt.encode(claims, TOKEN_KEY, algorithm="RS256")

    identity = authenticate(token, settings, token_signing_key, settings.audience)

    assert identity.user_id == "alice@company.com"
    assert identity.principals == {
        "user:alice@company.com",
        "group:everyone",
        "group:Everyone",
        "group:Finance",
    }
    assert identity.scopes == {"openid", "profile"}


def test_the_groups_claim_is_read_from_the_configured_name(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = token_settings.model_copy(update={"groups_claim": "roles"})
    token = sign_token({"groups": None, "roles": ["analyst"]})

    identity = authenticate(token, settings, token_signing_key, settings.audience)

    assert identity.principals == {"user:district-1", "group:everyone", "group:analyst"}


def test_an_es256_token_is_accepted(sign_token: Sign, token_settings: Settings) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    public = jwt.PyJWK.from_dict(ECAlgorithm.to_jwk(key.public_key(), as_dict=True))
    claims = jwt.decode(sign_token({}), options={"verify_signature": False})
    token = jwt.encode(claims, key, algorithm="ES256")

    identity = authenticate(token, token_settings, lambda _: public, token_settings.audience)

    assert identity.user_id == "district-1"
