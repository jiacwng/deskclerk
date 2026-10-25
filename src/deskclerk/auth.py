# Names the employee an OpenID Connect token was issued to: authenticate checks the access token a
# bearer caller sends, identify checks the ID token that browser sign-in receives, and
# logout_target checks the logout token the login server sends when it ends a session.

import hmac
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from deskclerk.config import Settings

logger = logging.getLogger(__name__)

SigningKeyLookup = Callable[[str], jwt.PyJWK]

JWKS_TIMEOUT_SECONDS = 5
JWKS_REFETCH_SECONDS = 60
CLOCK_SKEW_SECONDS = 30
BACKCHANNEL_LOGOUT_EVENT = "http://schemas.openid.net/event/backchannel-logout"


class AuthError(Exception):
    pass


class AuthUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    user_id: str
    client_id: str
    display_name: str
    sid: str | None = None
    subject: str | None = None
    groups: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LogoutTarget:
    sid: str | None
    sub: str | None


def _display_name(claims: dict[str, Any], user_id: str) -> str:
    for name in ("preferred_username", "email"):
        value = claims.get(name)
        if isinstance(value, str) and value:
            return value
    return user_id


def _groups(claims: dict[str, Any], settings: Settings) -> frozenset[str]:
    value = claims.get(settings.groups_claim)
    if isinstance(value, list):
        return frozenset(group for group in value if isinstance(group, str) and group)
    overage = claims.get("_claim_names")
    if isinstance(overage, dict) and settings.groups_claim in overage:
        logger.warning(
            "the token has too many groups to list (group overage), so %s is treated as in no "
            "group; emit only the groups assigned to the application",
            claims.get(settings.user_claim),
        )
    return frozenset()


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

    return Identity(
        user_id=username,
        client_id=client,
        display_name=_display_name(claims, username),
        groups=_groups(claims, settings),
    )


def identify(
    id_token: str,
    settings: Settings,
    signing_key: SigningKeyLookup,
    nonce: str,
) -> Identity:
    try:
        claims = jwt.decode(
            id_token,
            signing_key(id_token),
            algorithms=["RS256", "ES256"],
            audience=settings.client_id,
            issuer=settings.issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", "nonce", settings.user_claim]},
        )
    except jwt.PyJWKClientConnectionError as error:
        raise AuthUnavailable(f"cannot fetch the signing keys: {error}") from error
    except jwt.PyJWTError as error:
        raise AuthError(f"invalid token: {error}") from error

    sent_nonce = claims["nonce"]
    if not isinstance(sent_nonce, str) or not hmac.compare_digest(
        sent_nonce.encode(), nonce.encode()
    ):
        raise AuthError("nonce does not match the sign-in")
    if isinstance(claims["aud"], list) and len(claims["aud"]) > 1:
        if claims.get("azp") != settings.client_id:
            raise AuthError("azp must be this client when the token has several audiences")
    username = claims[settings.user_claim]
    if not isinstance(username, str) or not username:
        raise AuthError(f"{settings.user_claim} must be a non-empty string")

    sid, subject = claims.get("sid"), claims.get("sub")
    return Identity(
        user_id=username,
        client_id=settings.client_id,
        display_name=_display_name(claims, username),
        sid=sid if isinstance(sid, str) and sid else None,
        subject=subject if isinstance(subject, str) and subject else None,
        groups=_groups(claims, settings),
    )


def logout_target(
    logout_token: str, settings: Settings, signing_key: SigningKeyLookup
) -> LogoutTarget:
    try:
        claims = jwt.decode(
            logout_token,
            signing_key(logout_token),
            algorithms=["RS256", "ES256"],
            audience=settings.client_id,
            issuer=settings.issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", "jti"]},
        )
    except jwt.PyJWKClientConnectionError as error:
        raise AuthUnavailable(f"cannot fetch the signing keys: {error}") from error
    except jwt.PyJWTError as error:
        raise AuthError(f"invalid token: {error}") from error

    events = claims.get("events")
    if not isinstance(events, dict) or BACKCHANNEL_LOGOUT_EVENT not in events:
        raise AuthError("not a logout token: the back-channel logout event is missing")
    if "nonce" in claims:
        raise AuthError("not a logout token: it carries a nonce")
    sid, sub = claims.get("sid"), claims.get("sub")
    sid = sid if isinstance(sid, str) and sid else None
    sub = sub if isinstance(sub, str) and sub else None
    if sid is None and sub is None:
        raise AuthError("a logout token must name a sid or a sub")
    return LogoutTarget(sid, sub)


# An ID token is signed by the same keys and can name the API as its audience. Only access tokens
# carry the granted scopes: scope in Keycloak and RFC 9068, scp in Entra ID and Okta.
def _refuse_id_token(claims: dict[str, Any]) -> None:
    if claims.get("typ", "Bearer") != "Bearer":
        raise AuthError(f"not an access token: typ is {claims['typ']!r}")
    if claims.get("token_use", "access") != "access":
        raise AuthError(f"not an access token: token_use is {claims['token_use']!r}")
    if "scope" not in claims and "scp" not in claims:
        raise AuthError("not an access token: it grants no scope")
