# Langfuse tracing without the remote service: off without keys, a specific error when the package
# is missing, text masked unless the audit log keeps it, and the employee on the trace.

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from deskclerk.audit import hmac_hex
from deskclerk.tracing import (
    OMITTED,
    AskTrace,
    Tracing,
    TracingNotInstalled,
    omit_text,
    tracing_from_environment,
)


class FakeLangfuse:
    created: list[dict[str, Any]] = []

    def __init__(self, **options: Any) -> None:
        self.created.append(options)


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)


@pytest.fixture
def fake_package(monkeypatch: pytest.MonkeyPatch) -> type[FakeLangfuse]:
    package = ModuleType("langfuse")
    package.Langfuse = FakeLangfuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", package)
    FakeLangfuse.created = []
    return FakeLangfuse


@pytest.mark.parametrize("missing", ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"])
def test_one_missing_key_means_no_tracing(
    keys: None, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.delenv(missing)
    assert tracing_from_environment(False, None) is None


def test_keys_without_the_package_name_the_extra(
    keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", None)
    with pytest.raises(TracingNotInstalled, match=r"pip install \'\.\[tracing\]\'"):
        tracing_from_environment(False, None)


def test_the_client_masks_text_unless_the_audit_log_keeps_it(
    keys: None, fake_package: type[FakeLangfuse]
) -> None:
    assert tracing_from_environment(False, None) is not None
    assert tracing_from_environment(True, None) is not None
    hiding, keeping = (options["mask"] for options in fake_package.created)
    secret = {"messages": ["a salary of 90000"], "tokens": 12}
    assert hiding(data=secret) == {"messages": [OMITTED], "tokens": 12}
    assert hiding(data=[1, 2.5, None, True]) == [1, 2.5, None, True]
    assert keeping(data=secret) == secret


class FakeObservation:
    def __init__(self, name: str, as_type: str, input: Any) -> None:
        self.name = name
        self.as_type = as_type
        self.input = omit_text(data=input)
        self.output: Any = None

    def update(self, *, output: Any) -> None:
        self.output = omit_text(data=output)


class MaskingClient:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []

    @contextmanager
    def start_as_current_observation(
        self, *, name: str, as_type: str, input: Any = None
    ) -> Iterator[FakeObservation]:
        observation = FakeObservation(name, as_type, input)
        self.observations.append(observation)
        yield observation


def test_a_tool_call_is_a_tool_observation_named_by_server_and_tool_with_text_masked() -> None:
    client = MaskingClient()
    trace = AskTrace({}, None, client)
    with trace.tool_span("notes", "write_note", {"text": "salary 90000", "pages": 2}) as record:
        record("saved salary 90000")
    (observation,) = client.observations
    assert (observation.name, observation.as_type) == ("notes__write_note", "tool")
    assert observation.input == {"text": OMITTED, "pages": 2}
    assert observation.output == OMITTED


@pytest.mark.parametrize(
    ("include_text", "audit_key", "expected"),
    [
        (True, None, "ada"),
        (True, SecretStr("k" * 32), "ada"),
        (False, SecretStr("k" * 32), hmac_hex(SecretStr("k" * 32), "ada")),
        (False, None, None),
    ],
)
def test_the_employee_is_plain_hashed_or_absent_on_the_trace(
    include_text: bool, audit_key: SecretStr | None, expected: str | None
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_tracer = provider.get_tracer("test")
    updates: list[dict[str, Any]] = []

    class OtelSpan:
        def update(self, **fields: Any) -> None:
            updates.append(fields)

    class OtelClient:
        @contextmanager
        def start_as_current_observation(self, *, name: str, as_type: str) -> Iterator[OtelSpan]:
            with otel_tracer.start_as_current_span(name):
                yield OtelSpan()

    with Tracing(OtelClient(), "host", include_text, audit_key).ask("ada") as trace:
        assert len(trace.config["callbacks"]) == 1  # type: ignore[arg-type]
        trace.finish(2)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes.get("user.id") == expected
    assert updates == [{"metadata": {"withheld": 2}}]
