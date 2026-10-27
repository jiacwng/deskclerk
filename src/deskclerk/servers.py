# Reads the MCP servers the company admin lists in a TOML or mcpServers JSON file
# and checks each entry.

import json
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

LOCAL_HOSTS = {"localhost", "127.0.0.1"}
ENV_REFERENCE = re.compile(r"\$\{env:([^}]+)\}")
SERVER_NAME = re.compile(r"[A-Za-z0-9-]+(_[A-Za-z0-9-]+)*")


class ServersError(Exception):
    pass


@dataclass(frozen=True)
class McpServer:
    name: str
    url: str
    client_id: str | None
    client_secret: str | None
    auth: Literal["oauth", "none"] = "oauth"
    tools: tuple[str, ...] | None = None
    headers: tuple[tuple[str, str], ...] = ()
    client_cert: str | None = None
    client_key: str | None = None
    groups: tuple[str, ...] | None = None
    always_allow: bool = True
    confirm_tools: tuple[str, ...] = ()

    def open_to(self, groups: frozenset[str]) -> bool:
        return self.groups is None or not groups.isdisjoint(self.groups)


class _ServerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    plain_http: bool = False
    url: str
    client_id: str | None = None
    client_secret_env: str | None = None
    auth: Literal["oauth", "none"] = "oauth"
    tools: list[str] | None = None
    headers: dict[str, str] = {}
    client_cert: str | None = None
    client_key: str | None = None
    groups: list[str] | None = None
    always_allow: bool = True
    confirm_tools: list[str] = []

    @field_validator("name")
    @classmethod
    def _name_is_safe_in_a_tool_name(cls, value: str) -> str:
        if SERVER_NAME.fullmatch(value) is None:
            raise ValueError(
                f"server name {value!r} must use only letters, digits and hyphens, with single "
                "underscores between them, such as google-drive or jira_cloud"
            )
        return value

    @field_validator("tools", "confirm_tools")
    @classmethod
    def _tool_names_are_named_and_unique(cls, names: list[str] | None) -> list[str] | None:
        seen: set[str] = set()
        for name in names or []:
            if not name.strip():
                raise ValueError("a tool needs a name")
            if name in seen:
                raise ValueError(f"tool name {name!r} is listed twice")
            seen.add(name)
        return names

    @field_validator("groups")
    @classmethod
    def _groups_are_named(cls, names: list[str] | None) -> list[str] | None:
        if names is not None and (not names or any(not name.strip() for name in names)):
            raise ValueError("groups lists at least one group name, each with text")
        return names

    @field_validator("url")
    @classmethod
    def _url_is_https_or_allowed_http(cls, value: str, info: ValidationInfo) -> str:
        parsed = urlparse(value)
        if parsed.scheme == "https" and parsed.hostname:
            return value
        if parsed.scheme == "http" and parsed.hostname:
            if info.data.get("plain_http") or parsed.hostname in LOCAL_HOSTS:
                return value
        raise ValueError(
            f"{value!r} must be https, or http on localhost or 127.0.0.1 "
            "(set plain_http = true to allow http for this server)"
        )

    @model_validator(mode="after")
    def _certificate_and_key_come_together(self) -> "_ServerEntry":
        if (self.client_cert is None) != (self.client_key is None):
            raise ValueError(f"server {self.name!r} needs client_cert and client_key together")
        return self

    @model_validator(mode="after")
    def _oauth_owns_the_authorization_header(self) -> "_ServerEntry":
        if self.auth == "oauth" and any(name.lower() == "authorization" for name in self.headers):
            raise ValueError(
                f'server {self.name!r} sets an Authorization header with auth = "oauth"'
            )
        return self

    @model_validator(mode="after")
    def _secret_needs_client_id(self) -> "_ServerEntry":
        if self.client_secret_env is not None and self.client_id is None:
            raise ValueError(f"server {self.name!r} sets client_secret_env without client_id")
        return self

    @model_validator(mode="after")
    def _no_sign_in_needs_no_client(self) -> "_ServerEntry":
        if self.auth == "none" and (self.client_id is not None or self.client_secret_env):
            raise ValueError(f'server {self.name!r} has auth = "none" and a client')
        return self


class _McpJsonFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mcpServers: dict[str, dict[str, Any]] = {}


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


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except OSError as error:
        raise ServersError(f"cannot read {path}: {error}") from None
    except tomllib.TOMLDecodeError as error:
        raise ServersError(f"{path} is not valid TOML: {error}") from None


def _read_mcp_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            raw = json.load(file)
    except OSError as error:
        raise ServersError(f"cannot read {path}: {error}") from None
    except ValueError as error:
        raise ServersError(f"{path} is not valid JSON: {error}") from None
    try:
        file_model = _McpJsonFile.model_validate(raw)
    except ValidationError as error:
        raise ServersError(f"{path} is not a valid server list: {error}") from None
    servers = []
    for name, body in file_model.mcpServers.items():
        if "command" in body or body.get("type", "http") != "http":
            raise ServersError(
                f"server {name!r} is not an HTTP server; "
                "deskclerk connects to MCP servers over HTTP only"
            )
        fields = {key: value for key, value in body.items() if key != "type"}
        servers.append({**fields, "name": name})
    return {"server": servers}


def _parse(path: Path) -> _ServersFile:
    if path.suffix == ".json":
        raw = _read_mcp_json(path)
    else:
        raw = _read_toml(path)
    try:
        return _ServersFile.model_validate(raw)
    except ValidationError as error:
        raise ServersError(f"{path} is not a valid server list: {error}") from None


def secret_variables(path: Path) -> list[str]:
    names = []
    for entry in _parse(path).server:
        if entry.client_secret_env is not None:
            names.append(entry.client_secret_env)
        for value in entry.headers.values():
            names.extend(ENV_REFERENCE.findall(value))
    return names


def load_servers(path: Path, environ: Mapping[str, str] = os.environ) -> list[McpServer]:
    parsed = _parse(path)
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
        tools = None if entry.tools is None else tuple(entry.tools)
        headers = tuple(
            (name, _fill_environment(entry.name, name, value, environ))
            for name, value in entry.headers.items()
        )
        for file in (entry.client_cert, entry.client_key):
            if file is not None:
                _check_readable(entry.name, file)
        servers.append(
            McpServer(
                entry.name,
                entry.url,
                entry.client_id,
                secret,
                entry.auth,
                tools,
                headers,
                entry.client_cert,
                entry.client_key,
                None if entry.groups is None else tuple(entry.groups),
                entry.always_allow,
                tuple(entry.confirm_tools),
            )
        )
    return servers


def _fill_environment(server: str, header: str, value: str, environ: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable not in environ:
            raise ServersError(
                f"server {server!r} header {header!r} reads the environment variable "
                f"{variable}, which is not set"
            )
        return environ[variable]

    return ENV_REFERENCE.sub(replace, value)


def _check_readable(server: str, file: str) -> None:
    try:
        with open(file, "rb"):
            pass
    except OSError as error:
        raise ServersError(f"server {server!r} cannot read {file}: {error.strerror}") from None
