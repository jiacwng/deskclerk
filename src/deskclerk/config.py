# Settings read from DESKCLERK_* environment variables by pydantic-settings: how employees sign
# in, the model and the budget, the audit log and the request quota, the MCP server list, the
# Postgres database that holds all state, how long conversations are kept and the session of the
# web front, the SCIM endpoints that offboard a leaver and the group allowed to open /admin.

from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SESSION_KEY_CHARS = 32
MAX_PORT = 65535
DEFAULT_ASK_SECONDS = 180
DEFAULT_TOOL_OUTPUT_CHARS = 20000
ASK_SECONDS_PER_LISTING_SECOND = 10
MIN_LISTING_SECONDS = 5
DEFAULT_LISTING_SECONDS = DEFAULT_ASK_SECONDS / ASK_SECONDS_PER_LISTING_SECOND
DEFAULT_HISTORY_TOKENS = 40000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DESKCLERK_", frozen=True)

    issuer: str = "http://localhost:8080/realms/deskclerk"
    client_id: str = "deskclerk-api"
    client_secret: SecretStr | None = None
    audience: str
    # The claim that names the employee. It must never change or be reused for someone else: sub,
    # or oid on Entra ID. Changing it later orphans every stored token and conversation.
    user_claim: str = "sub"
    groups_claim: str = "groups"
    admin_group: str | None = None
    model: str = "anthropic:claude-haiku-5-5"
    price_model: str = ""
    budget_usd: float = 2.0
    audit_text: bool = False
    audit_key: SecretStr | None = None
    requests_per_minute: int = 20
    servers_file: Path = Path("servers.toml")
    database_url: SecretStr
    token_key: SecretStr | None = None
    step_limit: int = 8
    ask_seconds: int = Field(default=DEFAULT_ASK_SECONDS, gt=0)
    tool_output_chars: int = Field(default=DEFAULT_TOOL_OUTPUT_CHARS, gt=0)
    history_tokens: int = Field(default=DEFAULT_HISTORY_TOKENS, gt=0)
    tool_selection: bool = False
    allow_unscreened: bool = False
    public_url: str = "http://localhost:8000"
    session_key: SecretStr | None = None
    session_hours: int = 8
    scim_token: SecretStr | None = None
    retention_days: int = Field(default=0, ge=0)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=MAX_PORT)

    @model_validator(mode="before")
    @classmethod
    def _audience_is_the_client(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("audience"):
            values["audience"] = values.get("client_id") or cls.model_fields["client_id"].default
        if not values.get("price_model"):
            values["price_model"] = values.get("model") or cls.model_fields["model"].default
        return values

    @field_validator("public_url")
    @classmethod
    def _public_url_has_a_scheme(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                f"DESKCLERK_PUBLIC_URL starts with http:// or https://, such as "
                f"https://deskclerk.example.com, not {value!r}"
            )
        return value.rstrip("/")

    @property
    def listing_seconds(self) -> float:
        return max(MIN_LISTING_SECONDS, self.ask_seconds / ASK_SECONDS_PER_LISTING_SECOND)

    @property
    def sign_in_redirect_url(self) -> str:
        return self.public_url + "/login/callback"

    @property
    def client_metadata_url(self) -> str:
        return self.public_url + "/oauth/client.json"

    @property
    def redirect_url(self) -> str:
        return self.public_url + "/oauth/callback"

    @field_validator("session_key")
    @classmethod
    def _session_key_is_long(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < MIN_SESSION_KEY_CHARS:
            raise ValueError(
                f"DESKCLERK_SESSION_KEY needs at least {MIN_SESSION_KEY_CHARS} characters"
            )
        return value

    @field_validator("model", "price_model")
    @classmethod
    def _provider_and_model(cls, value: str, info: ValidationInfo) -> str:
        if info.field_name == "price_model" and value == "none":
            return value
        provider, _, name = value.partition(":")
        if not provider or not name:
            raise ValueError(
                f"DESKCLERK_{str(info.field_name).upper()} is provider:model, such as "
                f"anthropic:claude-haiku-5-5, not {value!r}"
            )
        return value


def load_settings() -> Settings:
    return Settings()
