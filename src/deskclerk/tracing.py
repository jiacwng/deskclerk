# Optional Langfuse tracing: when the Langfuse keys are set, each ask becomes one trace with the
# employee as user id and every model call recorded. Text is masked unless the audit log keeps it.

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from deskclerk.audit import hmac_hex

OMITTED = "[omitted]"
DEFAULT_HOST = "https://cloud.langfuse.com"


class TracingNotInstalled(RuntimeError):
    pass


def omit_text(*, data: Any, **kwargs: Any) -> Any:
    if isinstance(data, str):
        return OMITTED
    if isinstance(data, dict):
        return {key: omit_text(data=value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [omit_text(data=item) for item in data]
    return data


def keep_text(*, data: Any, **kwargs: Any) -> Any:
    return data


@dataclass(frozen=True)
class AskTrace:
    config: RunnableConfig
    _span: Any
    _client: Any

    def finish(self, withheld: int) -> None:
        self._span.update(metadata={"withheld": withheld})

    @contextmanager
    def tool_span(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> Iterator[Callable[[str], None]]:
        with self._client.start_as_current_observation(
            name=f"{server}__{tool}", as_type="tool", input=arguments
        ) as span:
            yield lambda output: span.update(output=output)


class Tracing:
    def __init__(
        self, client: Any, host: str, include_text: bool, audit_key: SecretStr | None
    ) -> None:
        self._client = client
        self.host = host
        self._include_text = include_text
        self._audit_key = audit_key

    def _traced_user(self, user_id: str) -> str | None:
        if self._include_text:
            return user_id
        if self._audit_key is not None:
            return hmac_hex(self._audit_key, user_id)
        return None

    @contextmanager
    def ask(self, user_id: str) -> Iterator[AskTrace]:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        with self._client.start_as_current_observation(name="ask", as_type="span") as span:
            with propagate_attributes(user_id=self._traced_user(user_id)):
                yield AskTrace({"callbacks": [CallbackHandler()]}, span, self._client)

    def flush(self) -> None:
        self._client.flush()


def tracing_from_environment(include_text: bool, audit_key: SecretStr | None) -> Tracing | None:
    if not os.environ.get("LANGFUSE_SECRET_KEY") or not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        raise TracingNotInstalled(
            "the LANGFUSE keys are set but langfuse is not installed; "
            "install it with pip install 'deskclerk[tracing]'"
        ) from None
    host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or DEFAULT_HOST
    client = Langfuse(base_url=host, mask=keep_text if include_text else omit_text)
    return Tracing(client, host, include_text, audit_key)
