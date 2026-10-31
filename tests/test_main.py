# The server entry point starts without the injection screen and makes the keys that are missing.

from pathlib import Path
from typing import Any

import pytest
import uvicorn
from cryptography.fernet import Fernet

from deskclerk import __main__ as entry
from deskclerk.database import Database


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, database: Database) -> None:
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    started: list[Any] = []
    monkeypatch.setattr(entry, "load_servers", lambda path: [])
    monkeypatch.setattr(entry, "discover_jwks_url", lambda settings, client: "http://idp/jwks")
    monkeypatch.setattr(entry, "jwks_signing_key", lambda url: lambda token: None)
    monkeypatch.setattr(entry, "chat_model", lambda settings: object())
    monkeypatch.setattr(entry, "create_app", lambda *args: "the app")
    monkeypatch.setattr(entry, "log_to_stdout", lambda: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, access_log: started.append(app))
    return started


@pytest.mark.usefixtures("environment")
def test_no_screen_key_starts_the_server_with_a_warning(
    served: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    entry.main()
    assert served == ["the app"]
    assert "tool outputs are not screened for prompt injection" in capsys.readouterr().err


@pytest.mark.usefixtures("environment")
def test_the_first_start_makes_the_session_key_and_logs_its_file_not_its_value(
    served: list[Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry.main()
    logged = capsys.readouterr().err
    key = (tmp_path / "data" / "session_key").read_text().strip()
    assert "generated session_key" in logged
    assert key not in logged


@pytest.mark.usefixtures("environment")
def test_a_screen_key_starts_the_server(monkeypatch: pytest.MonkeyPatch, served: list[Any]) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "a-key")
    monkeypatch.setattr(entry, "jev_from_environment", lambda: object())
    entry.main()
    assert served == ["the app"]


def test_a_secret_file_reaches_the_settings_before_they_load(
    monkeypatch: pytest.MonkeyPatch, served: list[Any], tmp_path: Path, database: Database
) -> None:
    key_file = tmp_path / "key"
    key_file.write_text(Fernet.generate_key().decode() + "\n")
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY", "placeholder")
    monkeypatch.delenv("DESKCLERK_TOKEN_KEY")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY_FILE", str(key_file))
    entry.main()
    assert served == ["the app"]


def test_a_missing_secret_file_exits_with_the_reason(
    monkeypatch: pytest.MonkeyPatch, served: list[Any], tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "absent"))
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert "absent" in str(stopped.value.code)
    assert served == []


@pytest.mark.usefixtures("environment")
def test_the_server_runs_with_the_access_log_off(
    monkeypatch: pytest.MonkeyPatch, served: list[Any]
) -> None:
    runs: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: runs.append(options))
    entry.main()
    assert runs[0]["access_log"] is False
