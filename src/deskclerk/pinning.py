# Pins each MCP tool by a hash of its definition the first time an employee uses the server (trust
# on first use, as Invariant Labs' mcp-scan does). A tool whose definition changed and a tool added
# later are withheld from the model. A tool the employee always allows is stored with the hash of
# its definition and stops counting as allowed when that hash no longer matches.

import hashlib
import json
from dataclasses import asdict, dataclass

from pydantic import TypeAdapter

from deskclerk.connections import ToolInfo
from deskclerk.tokens import TokenStore

_PINS = TypeAdapter(dict[str, str])


@dataclass(frozen=True)
class MenuEntry:
    name: str
    title: str
    read_only: bool


_MENU = TypeAdapter(list[MenuEntry])


@dataclass(frozen=True)
class PinnedTools:
    kept: tuple[ToolInfo, ...]
    changed: tuple[str, ...]


def definition_hash(tool: ToolInfo) -> str:
    definition = asdict(tool)
    if not tool.title:
        del definition["title"]
    canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def pin_tools(
    store: TokenStore, user_id: str, server_name: str, listed: list[ToolInfo]
) -> PinnedTools:
    stored = store.get(user_id, server_name, "tool_pins")
    if stored is None:
        if listed:
            pins = {tool.name: definition_hash(tool) for tool in listed}
            store.put(user_id, server_name, "tool_pins", _PINS.dump_json(pins).decode())
        return PinnedTools(tuple(listed), ())
    pins = _PINS.validate_json(stored)
    kept: list[ToolInfo] = []
    changed: list[str] = []
    for tool in listed:
        if pins.get(tool.name) == definition_hash(tool):
            kept.append(tool)
        else:
            changed.append(tool.name)
    return PinnedTools(tuple(kept), tuple(changed))


def allow_tool(store: TokenStore, user_id: str, server_name: str, tool: ToolInfo) -> None:
    allowed = _stored_allowances(store, user_id, server_name)
    allowed[tool.name] = definition_hash(tool)
    store.put(user_id, server_name, "always_allowed", _PINS.dump_json(allowed).decode())


def allowed_tools(
    store: TokenStore, user_id: str, server_name: str, offered: tuple[ToolInfo, ...]
) -> tuple[ToolInfo, ...]:
    stored = _stored_allowances(store, user_id, server_name)
    allowed = [tool for tool in offered if stored.get(tool.name) == definition_hash(tool)]
    still_valid = {tool.name: stored[tool.name] for tool in allowed}
    if still_valid != stored:
        store.put(user_id, server_name, "always_allowed", _PINS.dump_json(still_valid).decode())
    return tuple(allowed)


def _stored_allowances(store: TokenStore, user_id: str, server_name: str) -> dict[str, str]:
    stored = store.get(user_id, server_name, "always_allowed")
    return {} if stored is None else _PINS.validate_json(stored)


def save_menu(
    store: TokenStore, user_id: str, server_name: str, kept: tuple[ToolInfo, ...]
) -> None:
    menu = [MenuEntry(tool.name, tool.title, tool.read_only) for tool in kept]
    store.put(user_id, server_name, "tool_menu", _MENU.dump_json(menu).decode())


def read_menu(store: TokenStore, user_id: str, server_name: str) -> list[MenuEntry]:
    stored = store.get(user_id, server_name, "tool_menu")
    return [] if stored is None else _MENU.validate_json(stored)
