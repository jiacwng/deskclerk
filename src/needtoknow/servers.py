# Reads the MCP servers the company admin lists in a TOML file and checks each entry.

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

LOCAL_HOSTS = {"localhost", "127.0.0.1"}


class ServersError(Exception):
    pass


@dataclass(frozen=True)
class McpServer:
    name: str
    url: str
    client_id: str | None
    client_secret: str | None


class _ServerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    client_id: str | None = None
    client_secret_env: str | None = None

    @field_validator("name")
    @classmethod
    def _name_is_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a server needs a name")
        return value

    @field_validator("url")
    @classmethod
    def _url_is_https_or_local(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme == "https" and parsed.hostname:
            return value
        if parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS:
            return value
        raise ValueError(f"{value!r} must be https, or http on localhost or 127.0.0.1")

    @model_validator(mode="after")
    def _secret_needs_client_id(self) -> "_ServerEntry":
        if self.client_secret_env is not None and self.client_id is None:
            raise ValueError(f"server {self.name!r} sets client_secret_env without client_id")
        return self


class _ServersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: list[_ServerEntry] = []

    @field_validator("server")
    @classmethod
    def _names_are_unique(cls, entries: list[_ServerEntry]) -> list[_ServerEntry]:
        seen: set[str] = set()
        for entry in entries:
            if entry.name in seen:
                raise ValueError(f"server name {entry.name!r} is listed twice")
            seen.add(entry.name)
        return entries


def load_servers(path: Path, environ: Mapping[str, str] = os.environ) -> list[McpServer]:
    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except OSError as error:
        raise ServersError(f"cannot read {path}: {error}") from None
    except tomllib.TOMLDecodeError as error:
        raise ServersError(f"{path} is not valid TOML: {error}") from None
    try:
        parsed = _ServersFile.model_validate(raw)
    except ValidationError as error:
        raise ServersError(f"{path} is not a valid server list: {error}") from None

    servers = []
    for entry in parsed.server:
        secret = None
        if entry.client_secret_env is not None:
            secret = environ.get(entry.client_secret_env)
            if secret is None:
                raise ServersError(
                    f"server {entry.name!r} reads its client secret from the environment "
                    f"variable {entry.client_secret_env}, which is not set"
                )
        servers.append(McpServer(entry.name, entry.url, entry.client_id, secret))
    return servers
