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

from deskclerk.tracing import (
    OMITTED,
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
    assert tracing_from_environment(False) is None


def test_keys_without_the_package_name_the_extra(
    keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", None)
    with pytest.raises(TracingNotInstalled, match=r"deskclerk\[tracing\]"):
        tracing_from_environment(False)


def test_the_client_masks_text_unless_the_audit_log_keeps_it(
    keys: None, fake_package: type[FakeLangfuse]
) -> None:
    assert tracing_from_environment(False) is not None
    assert tracing_from_environment(True) is not None
    hiding, keeping = (options["mask"] for options in fake_package.created)
    secret = {"messages": ["a salary of 90000"], "tokens": 12}
    assert hiding(data=secret) == {"messages": [OMITTED], "tokens": 12}
    assert keeping(data=secret) == secret


def test_the_host_is_the_base_url_then_the_cloud(
    keys: None, fake_package: type[FakeLangfuse], monkeypatch: pytest.MonkeyPatch
) -> None:
    tracing = tracing_from_environment(False)
    assert tracing is not None
    assert tracing.host == "https://cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.com")
    tracing = tracing_from_environment(False)
    assert tracing is not None
    assert tracing.host == "https://langfuse.example.com"


def test_masking_leaves_numbers_and_none() -> None:
    assert omit_text(data=[1, 2.5, None, True]) == [1, 2.5, None, True]


def test_an_ask_trace_carries_the_employee_and_the_withheld_count() -> None:
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

    with Tracing(OtelClient(), "host").ask("ada") as trace:
        assert len(trace.config["callbacks"]) == 1  # type: ignore[arg-type]
        trace.finish(2)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["user.id"] == "ada"
    assert updates == [{"metadata": {"withheld": 2}}]
