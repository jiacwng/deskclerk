# Optional Langfuse tracing: when the Langfuse keys are set, each ask becomes one trace with the
# employee as user id and every model call recorded. Text is masked unless the audit log keeps it.

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

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

    def finish(self, withheld: int) -> None:
        self._span.update(metadata={"withheld": withheld})


class Tracing:
    def __init__(self, client: Any, host: str) -> None:
        self._client = client
        self.host = host

    @contextmanager
    def ask(self, user_id: str) -> Iterator[AskTrace]:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        with self._client.start_as_current_observation(name="ask", as_type="span") as span:
            with propagate_attributes(user_id=user_id):
                yield AskTrace({"callbacks": [CallbackHandler()]}, span)

    def flush(self) -> None:
        self._client.flush()


def tracing_from_environment(include_text: bool) -> Tracing | None:
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
    return Tracing(client, host)
