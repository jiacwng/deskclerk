# Validates an OpenID Connect access token and turns it into the principals the database
# filters on.

import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from needtoknow.config import Settings
from needtoknow.graph import GraphError, GraphGroups

SigningKeyLookup = Callable[[str], jwt.PyJWK]

EVERYONE_GROUP = "everyone"
EVERYONE = f"group:{EVERYONE_GROUP}"

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
    scopes: frozenset[str]
    client_id: str


def principals(user_id: str, groups: Iterable[str]) -> frozenset[str]:
    return frozenset({f"user:{user_id}", EVERYONE, *(f"group:{g}" for g in groups)})


def discover_jwks_url(settings: Settings, client: httpx.Client) -> str:
    if settings.jwks_url is not None:
        return settings.jwks_url
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
    graph: GraphGroups | None = None,
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
    scope = claims.get("scope", claims.get("scp", ""))
    if not isinstance(username, str) or not username:
        raise AuthError(f"{settings.user_claim} must be a non-empty string")
    if isinstance(scope, list) and all(isinstance(s, str) for s in scope):
        scope = " ".join(scope)
    if not isinstance(scope, str):
        raise AuthError("scope must be a string or a list of strings")
    client = claims.get("azp", claims.get("cid", ""))
    if not isinstance(client, str):
        raise AuthError("azp must be a string")
    groups = _groups(claims, settings.groups_claim, graph)
    # The database receives the principals joined by commas.
    for name in [username, *groups]:
        if "," in name:
            raise AuthError(f"{name!r} contains a comma")

    return Identity(
        user_id=username,
        principals=principals(username, groups),
        scopes=frozenset(scope.split()),
        client_id=client,
    )


# An ID token is signed by the same keys and can name the API as its audience, as Keycloak's
# does. Keycloak marks its tokens with typ, Cognito with token_use, and only access tokens carry
# the granted scopes: scope in Keycloak and RFC 9068, scp in Entra ID and Okta.
def _refuse_id_token(claims: dict[str, Any]) -> None:
    if claims.get("typ", "Bearer") != "Bearer":
        raise AuthError(f"not an access token: typ is {claims['typ']!r}")
    if claims.get("token_use", "access") != "access":
        raise AuthError(f"not an access token: token_use is {claims['token_use']!r}")
    if "scope" not in claims and "scp" not in claims:
        raise AuthError("not an access token: it grants no scope")


def _groups(claims: dict[str, Any], claim: str, graph: GraphGroups | None) -> list[str]:
    overflow = claims.get("_claim_names")
    if claim not in claims and isinstance(overflow, dict) and claim in overflow:
        if graph is None:
            raise AuthError(
                "the token leaves out the groups because the user is in too many; set "
                "NEEDTOKNOW_GRAPH_TENANT_ID, NEEDTOKNOW_GRAPH_CLIENT_ID and "
                "NEEDTOKNOW_GRAPH_CLIENT_SECRET to read them from Microsoft Graph"
            )
        try:
            return graph.groups_of(_object_id(claims))
        except GraphError as error:
            raise AuthUnavailable(str(error)) from error
    groups = claims.get(claim, [])
    if not isinstance(groups, list) or not all(isinstance(g, str) and g for g in groups):
        raise AuthError(f"{claim} must be a list of non-empty strings")
    # Keycloak sends each group's full path, /parent/child, so two groups of the same name under
    # different parents stay apart; a top-level /analyst becomes analyst.
    names = [group.removeprefix("/") for group in groups]
    if EVERYONE_GROUP in names:
        raise AuthError(f"a group named {EVERYONE_GROUP!r} would read every public document")
    return names


def _object_id(claims: dict[str, Any]) -> str:
    try:
        return str(uuid.UUID(claims.get("oid")))
    except (AttributeError, TypeError, ValueError):
        raise AuthError("oid must be the user's object id to read their groups") from None
