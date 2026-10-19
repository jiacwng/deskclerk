# Validates an OpenID Connect access token and names the employee it was issued to.

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from deskclerk.config import Settings

SigningKeyLookup = Callable[[str], jwt.PyJWK]

JWKS_TIMEOUT_SECONDS = 5
JWKS_REFETCH_SECONDS = 60
CLOCK_SKEW_SECONDS = 30


class AuthError(Exception):
    pass


class AuthUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    user_id: str
    client_id: str


def discover_jwks_url(settings: Settings, client: httpx.Client) -> str:
    discovery = f"{settings.issuer}/.well-known/openid-configuration"
    try:
        response = client.get(discovery)
        response.raise_for_status()
        jwks_uri = response.json()["jwks_uri"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise AuthUnavailable(f"cannot read jwks_uri from {discovery}: {error}") from None
    if not isinstance(jwks_uri, str) or not jwks_uri:
        raise AuthUnavailable(f"{discovery} gives no jwks_uri")
    return jwks_uri


# PyJWKClient fetches the key set again for every key id it does not know, before any signature
# is checked, so a token with a made-up key id would make each request call the login server. A
# refetch for an unknown key id happens at most once a minute; known ids come from the cache.
def jwks_signing_key(jwks_url: str) -> SigningKeyLookup:
    client = jwt.PyJWKClient(jwks_url, timeout=JWKS_TIMEOUT_SECONDS)
    lock = threading.Lock()
    last_refetch = [float("-inf")]

    def lookup(token: str) -> jwt.PyJWK:
        key_id = jwt.get_unverified_header(token).get("kid")
        known = {key.key_id for key in client.get_jwk_set().keys}
        if isinstance(key_id, str) and key_id in known:
            return client.get_signing_key(key_id)
        with lock:
            if time.monotonic() - last_refetch[0] < JWKS_REFETCH_SECONDS:
                raise jwt.PyJWKClientError(f"unknown signing key id {key_id!r}")
            last_refetch[0] = time.monotonic()
        return client.get_signing_key_from_jwt(token)

    return lookup


def authenticate(
    token: str,
    settings: Settings,
    signing_key: SigningKeyLookup,
    audience: str,
) -> Identity:
    try:
        claims = jwt.decode(
            token,
            signing_key(token),
            algorithms=["RS256", "ES256"],
            audience=audience,
            issuer=settings.issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", settings.user_claim]},
        )
    except jwt.PyJWKClientConnectionError as error:
        raise AuthUnavailable(f"cannot fetch the signing keys: {error}") from error
    except jwt.PyJWTError as error:
        raise AuthError(f"invalid token: {error}") from error

    _refuse_id_token(claims)
    username = claims[settings.user_claim]
    if not isinstance(username, str) or not username:
        raise AuthError(f"{settings.user_claim} must be a non-empty string")
    client = claims.get("azp", claims.get("cid", ""))
    if not isinstance(client, str):
        raise AuthError("azp must be a string")

    return Identity(user_id=username, client_id=client)


# An ID token is signed by the same keys and can name the API as its audience. Only access tokens
# carry the granted scopes: scope in Keycloak and RFC 9068, scp in Entra ID and Okta.
def _refuse_id_token(claims: dict[str, Any]) -> None:
    if claims.get("typ", "Bearer") != "Bearer":
        raise AuthError(f"not an access token: typ is {claims['typ']!r}")
    if claims.get("token_use", "access") != "access":
        raise AuthError(f"not an access token: token_use is {claims['token_use']!r}")
    if "scope" not in claims and "scp" not in claims:
        raise AuthError("not an access token: it grants no scope")
