# Reads an Entra ID user's groups from Microsoft Graph. Entra leaves the groups out of a token
# for a user in more than 200 of them, so the API asks Graph with its own application
# credentials (client credentials grant, permission GroupMember.Read.All) and gets the same
# group object ids the token would have carried.

import threading
import time

import httpx
from pydantic import SecretStr

from deskclerk.config import Settings

LOGIN = "https://login.microsoftonline.com"
GRAPH = "https://graph.microsoft.com/v1.0"
TIMEOUT_SECONDS = 5
# An application token is asked for again this long before Entra says it expires.
TOKEN_MARGIN_SECONDS = 60
CACHE_SECONDS = 300
CACHED_USERS = 10_000


class GraphError(Exception):
    pass


class GraphGroups:
    def __init__(
        self, client: httpx.Client, tenant_id: str, client_id: str, client_secret: SecretStr
    ) -> None:
        self._client = client
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._lock = threading.Lock()
        self._token = ""
        self._token_expires = 0.0
        self._groups: dict[str, tuple[float, tuple[str, ...]]] = {}

    # Each user's groups are kept for a few minutes, as a token carries them until it expires, so
    # a burst of requests asks Graph once. The oldest user is dropped once the cache is full.
    def groups_of(self, object_id: str) -> list[str]:
        now = time.monotonic()
        with self._lock:
            cached = self._groups.get(object_id)
            if cached is not None and now < cached[0]:
                return list(cached[1])
        groups = self._read_groups(object_id)
        with self._lock:
            self._groups.pop(object_id, None)
            if len(self._groups) >= CACHED_USERS:
                del self._groups[next(iter(self._groups))]
            self._groups[object_id] = (now + CACHE_SECONDS, tuple(groups))
        return groups

    def _read_groups(self, object_id: str) -> list[str]:
        url: str | None = f"{GRAPH}/users/{object_id}/transitiveMemberOf/microsoft.graph.group"
        params: dict[str, str] | None = {"$select": "id"}
        groups: list[str] = []
        try:
            headers = {"Authorization": f"Bearer {self._application_token()}"}
            while url is not None:
                response = self._client.get(url, params=params, headers=headers)
                response.raise_for_status()
                page = response.json()
                for group in page["value"]:
                    groups.append(group["id"])
                url = page.get("@odata.nextLink")
                params = None
                if url is not None and not url.startswith(f"{GRAPH}/"):
                    raise ValueError(f"next page {url!r} is not on {GRAPH}")
        except (httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError) as error:
            raise GraphError(f"cannot read the groups from Microsoft Graph: {error}") from None
        return groups

    def _application_token(self) -> str:
        with self._lock:
            if time.monotonic() < self._token_expires:
                return self._token
            response = self._client.post(
                f"{LOGIN}/{self._tenant_id}/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret.get_secret_value(),
                    "scope": "https://graph.microsoft.com/.default",
                },
            )
            response.raise_for_status()
            body = response.json()
            self._token = body["access_token"]
            self._token_expires = time.monotonic() + body["expires_in"] - TOKEN_MARGIN_SECONDS
            return self._token


def graph_from_settings(settings: Settings) -> GraphGroups | None:
    if (
        settings.graph_tenant_id is None
        or settings.graph_client_id is None
        or settings.graph_client_secret is None
    ):
        return None
    return GraphGroups(
        httpx.Client(timeout=TIMEOUT_SECONDS),
        settings.graph_tenant_id,
        settings.graph_client_id,
        settings.graph_client_secret,
    )
