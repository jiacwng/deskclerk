# The audit log: one JSON line per answer, naming who asked, from which client, which documents
# were used, and what it cost. The question and the SQL are logged only when
# NEEDTOKNOW_AUDIT_TEXT is set; otherwise only the question's length, and its HMAC-SHA256 when
# NEEDTOKNOW_AUDIT_KEY is set. Tokens and document text are never logged.

import hashlib
import hmac
import json
import logging
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic import SecretStr

from needtoknow.auth import Identity

_log = logging.getLogger(__name__)


class Audit:
    def __init__(self, include_text: bool, key: SecretStr | None) -> None:
        self.include_text = include_text
        self.key = key

    def answered(
        self,
        identity: Identity,
        question: str,
        started: float,
        documents: Iterable[str],
        sql: Iterable[str],
        withheld: int,
        refused: bool,
        exhausted: bool,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        line = {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "user": identity.user_id,
            "client": identity.client_id,
            "entry": "ask",
            "text_chars": len(question),
            "text_hmac": self._hmac(question),
            "text": question if self.include_text else None,
            "documents": list(documents),
            "sql": list(sql) if self.include_text else None,
            "withheld": withheld,
            "refused": refused,
            "exhausted": exhausted,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
        _log.info(json.dumps(line))

    def _hmac(self, text: str) -> str | None:
        if self.key is None:
            return None
        key = self.key.get_secret_value().encode()
        return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()


# The server entry point calls this once: the process writes its audit lines to stdout and the
# operator ships them, as a twelve-factor app does.
def log_to_stdout() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False
