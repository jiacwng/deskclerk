# Settings refuse a model named without its provider, and the audience defaults to the client.

import pytest
from pydantic import SecretStr, ValidationError

from deskclerk.config import Settings


@pytest.mark.parametrize("model_name", ["claude-haiku-5-5", ":claude-haiku-5-5", "anthropic:"])
def test_the_model_needs_a_provider_and_a_name(model_name: str) -> None:
    with pytest.raises(ValidationError, match="DESKCLERK_MODEL is provider:model"):
        Settings(model=model_name)


def test_the_audience_is_the_client_unless_set() -> None:
    assert Settings(client_id="page").audience == "page"
    assert Settings(client_id="page", audience="api://default").audience == "api://default"


def test_a_short_session_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(session_key=SecretStr("short"))
    assert Settings(session_key=SecretStr("k" * 32)).session_key is not None


def test_deskclerk_listens_on_localhost_port_8000_by_default() -> None:
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_the_listen_address_and_port_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DESKCLERK_HOST", "10.1.2.3")
    monkeypatch.setenv("DESKCLERK_PORT", "9000")
    settings = Settings()
    assert settings.host == "10.1.2.3"
    assert settings.port == 9000


@pytest.mark.parametrize("port", [0, 70000])
def test_a_port_outside_1_to_65535_is_refused(port: int) -> None:
    with pytest.raises(ValidationError):
        Settings(port=port)


def test_both_callback_urls_default_to_localhost() -> None:
    settings = Settings()
    assert settings.sign_in_redirect_url == "http://localhost:8000/login/callback"
    assert settings.redirect_url == "http://localhost:8000/oauth/callback"


def test_both_callback_urls_come_from_the_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKCLERK_PUBLIC_URL", "https://x.example/")
    settings = Settings()
    assert settings.sign_in_redirect_url == "https://x.example/login/callback"
    assert settings.redirect_url == "https://x.example/oauth/callback"


def test_a_public_url_without_a_scheme_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKCLERK_PUBLIC_URL", "x.example")
    with pytest.raises(ValidationError, match="DESKCLERK_PUBLIC_URL"):
        Settings()


def test_the_old_sign_in_redirect_setting_has_no_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKCLERK_SIGN_IN_REDIRECT_URL", "https://old.example/login/callback")
    assert Settings().sign_in_redirect_url == "http://localhost:8000/login/callback"
