# Settings read from DESKCLERK_* environment variables by pydantic-settings: how employees sign
# in, the model and the budget, the audit log and the request quota, the MCP server list, the
# encrypted store of each employee's MCP tokens and the session of the web front.

from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SESSION_KEY_CHARS = 32
MAX_PORT = 65535


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DESKCLERK_", frozen=True)

    issuer: str = "http://localhost:8080/realms/deskclerk"
    client_id: str = "deskclerk-api"
    audience: str
    # The claim that names the employee. Keycloak's username is readable; on Entra ID use oid
    # and on Okta sub, the identifiers that never change when a person is renamed.
    user_claim: str = "preferred_username"
    model: str = "anthropic:claude-haiku-5-5"
    budget_usd: float = 2.0
    audit_text: bool = False
    audit_key: SecretStr | None = None
    requests_per_minute: int = 20
    servers_file: Path = Path("servers.toml")
    token_file: Path = Path("data/tokens.sqlite")
    token_key: SecretStr | None = None
    redirect_url: str = "http://localhost:8000/oauth/callback"
    step_limit: int = 8
    sign_in_redirect_url: str = "http://localhost:8000/login/callback"
    session_key: SecretStr | None = None
    session_hours: int = 8
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=MAX_PORT)

    @model_validator(mode="before")
    @classmethod
    def _audience_is_the_client(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("audience"):
            values["audience"] = values.get("client_id") or cls.model_fields["client_id"].default
        return values

    @field_validator("session_key")
    @classmethod
    def _session_key_is_long(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < MIN_SESSION_KEY_CHARS:
            raise ValueError(
                f"DESKCLERK_SESSION_KEY needs at least {MIN_SESSION_KEY_CHARS} characters"
            )
        return value

    @field_validator("model")
    @classmethod
    def _provider_and_model(cls, value: str) -> str:
        provider, _, name = value.partition(":")
        if not provider or not name:
            raise ValueError(
                f"DESKCLERK_MODEL is provider:model, such as anthropic:claude-haiku-5-5, "
                f"not {value!r}"
            )
        return value


def load_settings() -> Settings:
    return Settings()
