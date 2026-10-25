# The audit log: one JSON line per ask or confirmation and one per connect or disconnect, naming
# who did it, which tools the agent called, which calls wait for the employee and what was decided.
# An erase of an employee is one line with the row counts removed and no content.
# The request, each call's arguments and results, and a gate's reason are logged only when
# DESKCLERK_AUDIT_TEXT is set; otherwise only the request's length and, when DESKCLERK_AUDIT_KEY is
# set, its HMAC-SHA256. Tokens are never logged.

import hashlib
import hmac
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import SecretStr

from deskclerk.agent import AgentResult, BlockedCall, DecidedCall, ExecutedCall, PendingCall
from deskclerk.auth import Identity

_log = logging.getLogger(__name__)


def hmac_hex(key: SecretStr, text: str) -> str:
    return hmac.new(key.get_secret_value().encode(), text.encode(), hashlib.sha256).hexdigest()


class Audit:
    def __init__(self, include_text: bool, key: SecretStr | None) -> None:
        self.include_text = include_text
        self.key = key

    def asked(
        self,
        identity: Identity,
        conversation: UUID | None,
        request: str,
        started: float,
        outcome: str,
        result: AgentResult | None,
        cost_usd: float,
        cause: str | None = None,
        entry: str = "ask",
    ) -> None:
        line = self._line(identity, entry)
        line.update(
            {
                "conversation": None if conversation is None else str(conversation),
                "outcome": outcome,
                "text_chars": len(request),
                "text_hmac": self._hmac(request),
                "executed": [] if result is None else [self._executed(c) for c in result.executed],
                "blocked": [] if result is None else [self._blocked(c) for c in result.blocked],
                "pending": [] if result is None else [self._pending(c) for c in result.pending],
                "decided": [] if result is None else [self._decided(c) for c in result.decided],
                "not_connected": [] if result is None else list(result.not_connected),
                "unavailable": [] if result is None else list(result.unavailable),
                "step_limit_hit": False if result is None else result.step_limit_hit,
                "cut_off": False if result is None else result.cut_off,
                "withheld": 0 if result is None else result.withheld,
                "changed_tools": [] if result is None else list(result.changed_tools),
                "input_tokens": 0 if result is None else result.total_usage.input_tokens,
                "output_tokens": 0 if result is None else result.total_usage.output_tokens,
                "cost_usd": cost_usd,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
        )
        if cause is not None:
            line["cause"] = cause
        if self.include_text:
            line["text"] = request
        _log.info(json.dumps(line))

    def connection(self, identity: Identity, action: str, server_name: str) -> None:
        line = self._line(identity, action)
        line["server"] = server_name
        _log.info(json.dumps(line))

    def erased(self, user_id: str, secrets: int, threads: int, costs: int) -> None:
        line = {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "user": user_id,
            "entry": "erase",
            "secrets": secrets,
            "threads": threads,
            "costs": costs,
        }
        _log.info(json.dumps(line))

    def _line(self, identity: Identity, entry: str) -> dict[str, Any]:
        return {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "user": identity.user_id,
            "client": identity.client_id,
            "entry": entry,
        }

    def _executed(self, call: ExecutedCall) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "server": call.server,
            "tool": call.tool,
            "is_error": call.is_error,
        }
        if self.include_text:
            entry.update({"arguments": call.arguments, "result": call.result})
        return entry

    def _blocked(self, call: BlockedCall) -> dict[str, Any]:
        entry: dict[str, Any] = {"server": call.server, "tool": call.tool}
        if self.include_text:
            entry.update({"arguments": call.arguments, "reason": call.reason})
        return entry

    def _pending(self, call: PendingCall) -> dict[str, Any]:
        entry: dict[str, Any] = {"server": call.server, "tool": call.tool}
        if self.include_text:
            entry["arguments"] = call.arguments
        return entry

    def _decided(self, call: DecidedCall) -> dict[str, Any]:
        return {"server": call.server, "tool": call.tool, "decision": call.decision}

    def _hmac(self, text: str) -> str | None:
        if self.key is None:
            return None
        return hmac_hex(self.key, text)


# The server entry point calls this once: the process writes its audit lines to stdout and the
# operator ships them, as a twelve-factor app does.
def log_to_stdout() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False
