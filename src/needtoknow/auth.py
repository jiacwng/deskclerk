# Validates a Keycloak access token and turns it into the principals the database filters on.

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import jwt

from needtoknow.config import Settings

SigningKeyLookup = Callable[[str], jwt.PyJWK]

EVERYONE = "group:everyone"

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
    principals: frozenset[str]


def principals(user_id: str, groups: Iterable[str]) -> frozenset[str]:
    return frozenset({f"user:{user_id}", EVERYONE, *(f"group:{g}" for g in groups)})


# PyJWKClient fetches the key set again for every key id it does not know, before any signature
# is checked, so a token with a made-up key id would make each request call the login server. A
# refetch for an unknown key id happens at most once a minute; known ids come from the cache.
def jwks_signing_key(settings: Settings) -> SigningKeyLookup:
    client = jwt.PyJWKClient(settings.jwks_url, timeout=JWKS_TIMEOUT_SECONDS)
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


def authenticate(token: str, settings: Settings, signing_key: SigningKeyLookup) -> Identity:
    # aud names the API the token is meant for, azp the client that asked for it; here both are
    # the needtoknow-api client. typ separates an access token from an ID token.
    try:
        claims = jwt.decode(
            token,
            signing_key(token),
            algorithms=["RS256"],
            audience=settings.client_id,
            issuer=settings.issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", "azp", "typ", "preferred_username"]},
        )
    except jwt.PyJWKClientConnectionError as error:
        raise AuthUnavailable(f"cannot fetch the signing keys: {error}") from error
    except jwt.PyJWTError as error:
        raise AuthError(f"invalid token: {error}") from error

    if claims["typ"] != "Bearer":
        raise AuthError(f"not an access token: typ is {claims['typ']!r}")
    if claims["azp"] != settings.client_id:
        raise AuthError(f"token was issued to {claims['azp']!r}, not {settings.client_id!r}")

    username = claims["preferred_username"]
    groups = claims.get("groups", [])
    if not isinstance(username, str) or not username:
        raise AuthError("preferred_username must be a non-empty string")
    if not isinstance(groups, list) or not all(isinstance(g, str) and g for g in groups):
        raise AuthError("groups must be a list of non-empty strings")
    # The database receives the principals joined by commas.
    for name in [username, *groups]:
        if "," in name:
            raise AuthError(f"{name!r} contains a comma")

    return Identity(user_id=username, principals=principals(username, groups))
