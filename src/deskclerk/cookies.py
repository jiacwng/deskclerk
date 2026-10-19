# Signs and checks the values of the cookies the web front sets, with HMAC-SHA256 under a key the
# operator holds. A value carries its own expiry, and each purpose has its own derived key, so a
# sign-in cookie cannot be replayed as a session.

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _derived_key(key: bytes, purpose: str) -> bytes:
    return hmac.new(key, purpose.encode(), hashlib.sha256).digest()


def _tag(key: bytes, purpose: str, body: bytes) -> bytes:
    return hmac.new(_derived_key(key, purpose), body, hashlib.sha256).digest()


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(key: bytes, purpose: str, payload: dict[str, Any], lifetime_seconds: int) -> str:
    body = json.dumps({**payload, "exp": int(time.time()) + lifetime_seconds}).encode()
    return f"{_encode(body)}.{_encode(_tag(key, purpose, body))}"


def unsign(key: bytes, purpose: str, value: str) -> dict[str, Any] | None:
    body_text, _, tag_text = value.partition(".")
    try:
        body = _decode(body_text)
        tag = _decode(tag_text)
        if not hmac.compare_digest(tag, _tag(key, purpose, body)):
            return None
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("exp", 0) < time.time():
        return None
    return payload
