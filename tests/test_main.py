# The server entry point refuses to start without the injection screen unless it is allowed.

from pathlib import Path
from typing import Any

import pytest
import uvicorn
from cryptography.fernet import Fernet

from deskclerk import __main__ as entry


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DESKCLERK_TOKEN_FILE", str(tmp_path / "tokens.sqlite"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DESKCLERK_ALLOW_UNSCREENED", raising=False)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    started: list[Any] = []
    monkeypatch.setattr(entry, "load_servers", lambda path: [])
    monkeypatch.setattr(entry, "discover_jwks_url", lambda settings, client: "http://idp/jwks")
    monkeypatch.setattr(entry, "jwks_signing_key", lambda url: lambda token: None)
    monkeypatch.setattr(entry, "chat_model", lambda settings: object())
    monkeypatch.setattr(entry, "create_app", lambda *args: "the app")
    monkeypatch.setattr(entry, "log_to_stdout", lambda: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: started.append(app))
    return started


@pytest.mark.usefixtures("environment")
def test_no_screen_key_and_no_allowance_exits_with_the_reason(served: list[Any]) -> None:
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == (
        "deskclerk: TYPESAFE_API_KEY is not set, so tool outputs cannot be screened for prompt "
        "injection. Set it, or set DESKCLERK_ALLOW_UNSCREENED=true to start without the screen."
    )
    assert served == []


@pytest.mark.usefixtures("environment")
def test_the_allowance_starts_the_server_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, served: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DESKCLERK_ALLOW_UNSCREENED", "true")
    entry.main()
    assert served == ["the app"]
    assert "tool outputs are not screened for prompt injection" in capsys.readouterr().err


@pytest.mark.usefixtures("environment")
def test_a_screen_key_starts_the_server_without_the_allowance(
    monkeypatch: pytest.MonkeyPatch, served: list[Any]
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "a-key")
    monkeypatch.setattr(entry, "jev_from_environment", lambda: object())
    entry.main()
    assert served == ["the app"]
