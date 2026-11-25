# The gate for tool calls that are not read-only: the arguments must fit the tool's own schema
# and a destructive call may only name ids the agent has seen. The first check that refuses ends
# the gate.

import re
from typing import Any

from jsonschema import Draft202012Validator, SchemaError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for
from referencing.exceptions import Unresolvable

from deskclerk.agent import Gate, ProposedCall, Verdict


def write_check() -> Gate:
    async def gate(call: ProposedCall) -> Verdict:
        schema_reason = _schema_problem(call)
        if schema_reason:
            return Verdict(False, f"schema check: {schema_reason}")
        if call.tool.destructive:
            grounding_reason = _ungrounded_id(call)
            if grounding_reason:
                return Verdict(False, f"grounding check: {grounding_reason}")
        return Verdict(True, "deterministic checks passed")

    return gate


def _schema_problem(call: ProposedCall) -> str:
    schema = call.tool.input_schema
    try:
        validator_class = validator_for(schema, default=Draft202012Validator)
        validator_class.check_schema(schema)
    except SchemaError as broken:
        return f"the tool's own schema is invalid: {broken}"
    try:
        worst = best_match(validator_class(schema).iter_errors(call.arguments))
    except Unresolvable as broken:
        return f"the tool's own schema has a reference that cannot be resolved: {broken}"
    message: str = worst.message if worst else ""
    return message


def _ungrounded_id(call: ProposedCall) -> str:
    seen = [call.request, *call.observed]
    for name, value in _identifiers(call).items():
        whole_token = re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])")
        if not value or not any(whole_token.search(text) for text in seen):
            return f"acting on an id it has not seen: {name}={value}"
    return ""


# An identifier is a string argument, at any depth, whose key is id, *_id, *Id, ids, *_ids or *Ids
# (a list's items take the list's key) or whose schema has format uuid, and it must appear as a
# whole token. The tool's own schema decides, so the rule holds for any server.
def _identifiers(call: ProposedCall) -> dict[str, str]:
    found: dict[str, str] = {}
    _collect_identifiers(call.arguments, call.tool.input_schema, "", "", found)
    return found


def _collect_identifiers(
    value: Any, schema: Any, label: str, key: str, found: dict[str, str]
) -> None:
    spec: dict[str, Any] = schema if isinstance(schema, dict) else {}
    if isinstance(value, str):
        if _is_identifier_key(key) or spec.get("format") == "uuid":
            found[label] = value
    elif isinstance(value, dict):
        properties = spec.get("properties")
        for name, item in value.items():
            item_schema = properties.get(name) if isinstance(properties, dict) else None
            child_label = f"{label}.{name}" if label else name
            _collect_identifiers(item, item_schema, child_label, name, found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_identifiers(item, spec.get("items"), f"{label}[{index}]", key, found)


def _is_identifier_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in ("id", "ids") or lowered.endswith(("_id", "_ids")) or key.endswith(("Id", "Ids"))
    )
