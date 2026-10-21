# The gate for tool calls that are not read-only: the arguments must fit the tool's own schema
# and a destructive call may only name ids the agent has seen. The first check that refuses ends
# the gate.

import re
from typing import Any

from jsonschema import Draft202012Validator, SchemaError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

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
    worst = best_match(validator_class(schema).iter_errors(call.arguments))
    message: str = worst.message if worst else ""
    return message


def _ungrounded_id(call: ProposedCall) -> str:
    seen = [call.request, *call.observed]
    for name, value in _identifiers(call).items():
        whole_token = re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])")
        if not value or not any(whole_token.search(text) for text in seen):
            return f"acting on an id it has not seen: {name}={value}"
    return ""


# An identifier is a string argument named id, *_id or *Id (note_id, userId) or whose schema has
# format uuid, and it must appear as a whole token. The tool's own schema decides, so the rule
# holds for any server.
def _identifiers(call: ProposedCall) -> dict[str, str]:
    properties: dict[str, Any] = call.tool.input_schema.get("properties", {})
    found: dict[str, str] = {}
    for name, value in call.arguments.items():
        if not isinstance(value, str):
            continue
        spec = properties.get(name)
        named_as_id = name.lower() == "id" or name.lower().endswith("_id") or name.endswith("Id")
        is_uuid = isinstance(spec, dict) and spec.get("format") == "uuid"
        if named_as_id or is_uuid:
            found[name] = value
    return found
