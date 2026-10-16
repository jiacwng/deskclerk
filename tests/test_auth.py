# auth.py turns a signed access token into principals and refuses every token it should.
# Tokens are signed with the throwaway RSA key from conftest.py, so no Keycloak server is needed.

import time
from collections.abc import Callable

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from conftest import TOKEN_KEY
from needtoknow.auth import (
    AuthError,
    AuthUnavailable,
    SigningKeyLookup,
    authenticate,
    jwks_signing_key,
)
from needtoknow.config import JWKS_PATH, Settings

Sign = Callable[[dict[str, object]], str]


def test_valid_token_gives_user_groups_and_everyone(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(sign_token({}), token_settings, token_signing_key)

    assert identity.user_id == "district-1"
    assert identity.principals == {"user:district-1", "group:everyone", "group:district:1"}


def test_missing_groups_gives_user_and_everyone(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(sign_token({"groups": None}), token_settings, token_signing_key)
    assert identity.principals == {"user:district-1", "group:everyone"}


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        pytest.param({"exp": int(time.time()) - 60}, "expired", id="expired"),
        pytest.param({"exp": None}, "exp", id="no-expiry"),
        pytest.param({"iat": int(time.time()) + 60}, "not yet valid", id="issued-in-the-future"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, "issuer", id="wrong-issuer"),
        pytest.param({"aud": None}, '"aud"', id="no-audience"),
        pytest.param({"aud": "account"}, "Audience doesn't match", id="wrong-audience"),
        pytest.param({"azp": "other-client"}, "issued to 'other-client'", id="wrong-azp"),
        pytest.param({"typ": "ID"}, "not an access token", id="id-token"),
        pytest.param({"preferred_username": None}, "preferred_username", id="no-username"),
        pytest.param({"preferred_username": "a,b"}, "comma", id="comma-in-username"),
        pytest.param({"groups": ["district:1,analyst"]}, "comma", id="comma-in-group"),
        pytest.param({"groups": "district:1"}, "groups must be a list", id="groups-not-a-list"),
    ],
)
def test_bad_claims_are_refused(
    claims: dict[str, object],
    reason: str,
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> None:
    with pytest.raises(AuthError, match=reason):
        authenticate(sign_token(claims), token_settings, token_signing_key)


def test_audience_list_naming_the_api_is_accepted(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = sign_token({"aud": ["account", token_settings.client_id]})
    assert authenticate(token, token_settings, token_signing_key).user_id == "district-1"


def test_clock_skew_within_thirty_seconds_is_accepted(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    now = int(time.time())
    issued_ahead = sign_token({"iat": now + 10})
    expired_just_now = sign_token({"iat": now - 300, "exp": now - 10})

    assert authenticate(issued_ahead, token_settings, token_signing_key).user_id == "district-1"
    assert authenticate(expired_just_now, token_settings, token_signing_key).user_id == "district-1"


def test_unreachable_key_server_is_unavailable_not_invalid(
    sign_token: Sign, token_settings: Settings
) -> None:
    def signing_key(token: str) -> jwt.PyJWK:
        raise jwt.PyJWKClientConnectionError("connection refused")

    with pytest.raises(AuthUnavailable, match="cannot fetch the signing keys"):
        authenticate(sign_token({}), token_settings, signing_key)


def test_real_key_client_reports_a_closed_port_as_unavailable(
    sign_token: Sign, token_settings: Settings
) -> None:
    # Nothing listens on port 9, so the connection is refused at once.
    closed = token_settings.model_copy(
        update={
            "issuer": "http://127.0.0.1:9/realms/needtoknow",
            "jwks_url": f"http://127.0.0.1:9/realms/needtoknow{JWKS_PATH}",
        }
    )
    with pytest.raises(AuthUnavailable):
        authenticate(sign_token({}), closed, jwks_signing_key(closed))


def test_bad_signature_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode({"preferred_username": "district-1"}, other_key, algorithm="RS256")
    with pytest.raises(AuthError, match="Signature verification failed"):
        authenticate(token, token_settings, token_signing_key)


def test_symmetric_algorithm_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = jwt.encode(
        {"preferred_username": "district-1"}, "a-shared-secret-of-32-bytes-long", "HS256"
    )
    with pytest.raises(AuthError, match="invalid token"):
        authenticate(token, token_settings, token_signing_key)


def test_garbage_is_refused(token_settings: Settings, token_signing_key: SigningKeyLookup) -> None:
    with pytest.raises(AuthError, match="invalid token"):
        authenticate("not-a-token", token_settings, token_signing_key)


def test_made_up_key_ids_fetch_the_key_set_at_most_once_a_minute(
    monkeypatch: pytest.MonkeyPatch, token_settings: Settings
) -> None:
    key = RSAAlgorithm.to_jwk(TOKEN_KEY.public_key(), as_dict=True) | {"kid": "real"}
    fetches: list[str] = []

    def fetch(client: jwt.PyJWKClient) -> dict[str, object]:
        fetches.append(client.uri)
        return {"keys": [key]}

    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch)
    lookup = jwks_signing_key(token_settings)
    claims = {"sub": "x"}

    for forged in ("forged-1", "forged-2", "forged-3"):
        token = jwt.encode(claims, TOKEN_KEY, algorithm="RS256", headers={"kid": forged})
        with pytest.raises(jwt.PyJWTError):
            lookup(token)
    real = jwt.encode(claims, TOKEN_KEY, algorithm="RS256", headers={"kid": "real"})

    assert lookup(real).key_id == "real"
    assert len(fetches) == 2
