# Pins each MCP tool by a hash of its definition the first time an employee uses the server (trust
# on first use, as Invariant Labs' mcp-scan does). A tool whose definition changed and a tool added
# later are withheld from the model.

import hashlib
import json
from dataclasses import asdict, dataclass

from pydantic import TypeAdapter

from deskclerk.connections import ToolInfo
from deskclerk.tokens import TokenStore

_PINS = TypeAdapter(dict[str, str])


@dataclass(frozen=True)
class PinnedTools:
    kept: tuple[ToolInfo, ...]
    changed: tuple[str, ...]


def definition_hash(tool: ToolInfo) -> str:
    canonical = json.dumps(asdict(tool), sort_keys=True, separators=(",", ":"))
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
