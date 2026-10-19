# Settings read from NEEDTOKNOW_* environment variables by pydantic-settings: how employees sign
# in, the Entra ID Graph credentials, the model and its prices, the budget, the audit log and the
# request quota, the MCP server list, the encrypted store of each employee's MCP tokens and the
# address and session of the web front.

from pathlib import Path
from typing import Any

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SESSION_KEY_CHARS = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEEDTOKNOW_", frozen=True)

    issuer: str = "http://localhost:8080/realms/needtoknow"
    # Left unset, the signing keys are found from the issuer's OpenID configuration. Set it where
    # the API reaches the login server by another address than the issuer, as in a container.
    jwks_url: str | None = None
    client_id: str = "needtoknow-api"
    audience: str
    login_scope: str = "openid"
    # The claim that names the employee. Keycloak's username is readable; on Entra ID use oid
    # and on Okta sub, the identifiers that never change when a person is renamed.
    user_claim: str = "preferred_username"
    groups_claim: str = "groups"
    graph_tenant_id: str | None = None
    graph_client_id: str | None = None
    graph_client_secret: SecretStr | None = None
    model: str = "anthropic:claude-haiku-5-5"
    price_input: float | None = None
    price_output: float | None = None
    budget_usd: float = 2.0
    jev_budget_usd: float = 1.0
    audit_text: bool = False
    audit_key: SecretStr | None = None
    requests_per_minute: int = 20
    servers_file: Path = Path("servers.toml")
    token_file: Path = Path("data/tokens.sqlite")
    token_key: SecretStr | None = None
    redirect_url: str = "http://localhost:8000/oauth/callback"
    mcp_client_name: str = "needtoknow"
    step_limit: int = 8
    host: str = "127.0.0.1"
    port: int = 8000
    sign_in_redirect_url: str = "http://localhost:8000/login/callback"
    session_key: SecretStr | None = None
    session_hours: int = 8

    @model_validator(mode="before")
    @classmethod
    def _audience_is_the_client(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("audience"):
            values["audience"] = values.get("client_id") or cls.model_fields["client_id"].default
        return values

    @model_validator(mode="after")
    def _all_graph_credentials_or_none(self) -> "Settings":
        given = [self.graph_tenant_id, self.graph_client_id, self.graph_client_secret]
        if any(value is None for value in given) and any(value is not None for value in given):
            raise ValueError(
                "set NEEDTOKNOW_GRAPH_TENANT_ID, NEEDTOKNOW_GRAPH_CLIENT_ID and "
                "NEEDTOKNOW_GRAPH_CLIENT_SECRET together or none of them"
            )
        return self

    @model_validator(mode="after")
    def _both_prices_or_neither(self) -> "Settings":
        if (self.price_input is None) != (self.price_output is None):
            raise ValueError(
                "set both NEEDTOKNOW_PRICE_INPUT and NEEDTOKNOW_PRICE_OUTPUT or neither"
            )
        return self

    @field_validator("session_key")
    @classmethod
    def _session_key_is_long(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < MIN_SESSION_KEY_CHARS:
            raise ValueError(
                f"NEEDTOKNOW_SESSION_KEY needs at least {MIN_SESSION_KEY_CHARS} characters"
            )
        return value

    @field_validator("model")
    @classmethod
    def _provider_and_model(cls, value: str) -> str:
        provider, _, name = value.partition(":")
        if not provider or not name:
            raise ValueError(
                f"NEEDTOKNOW_MODEL is provider:model, such as anthropic:claude-haiku-5-5, "
                f"not {value!r}"
            )
        return value


def load_settings() -> Settings:
    return Settings()
