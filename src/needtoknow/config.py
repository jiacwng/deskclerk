# Settings read from NEEDTOKNOW_* environment variables by pydantic-settings. The defaults are
# the development values from compose.yaml, and a database password left at its default is
# refused for any database that is not on this machine or in the compose network. The API
# process loads only Settings; the admin, owner and reader passwords live in ProvisionSettings,
# which only ingestion, the evaluation and the tests load.

import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The signing keys sit under the issuer, except where the API reaches the login server by a
# different address than the one stamped into the tokens, as it does inside a container.
JWKS_PATH = "/protocol/openid-connect/certs"

# Half the cores embeds as fast as all of them on a mixed corpus and leaves the machine usable.
# A bulk load that owns the machine raises it.
DEFAULT_EMBED_THREADS = max(1, (os.cpu_count() or 2) // 2)

# The schemas and relations every employee may read in full, so the startup check lets the app
# role read them without row-level security: the nine BIRD databases that carry no policy and
# financial.district, the list of districts.
PUBLIC_RELATIONS = frozenset(
    {
        "card_games",
        "codebase_community",
        "debit_card_specializing",
        "european_football_2",
        "financial.district",
        "formula_1",
        "student_club",
        "superhero",
        "thrombosis_prediction",
        "toxicology",
    }
)

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "db"})

_FROM_ENVIRONMENT = SettingsConfigDict(env_prefix="NEEDTOKNOW_", frozen=True)


class Settings(BaseSettings):
    model_config = _FROM_ENVIRONMENT

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "needtoknow"
    app_password: str = "needtoknow_app"
    model_cache: Path = Path("~/.cache/needtoknow/models")
    embed_threads: int = DEFAULT_EMBED_THREADS
    issuer: str = "http://localhost:8080/realms/needtoknow"
    jwks_url: str
    client_id: str = "needtoknow-api"
    model: str = "anthropic:claude-haiku-5-5"
    price_input: float | None = None
    price_output: float | None = None
    budget_usd: float = 2.0
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001
    mcp_url: str = "http://127.0.0.1:8001/mcp"
    screen: bool = True
    audit_text: bool = False
    requests_per_minute: int = 20
    pool_size: int = 10
    public_relations: Annotated[frozenset[str], NoDecode] = PUBLIC_RELATIONS

    @model_validator(mode="before")
    @classmethod
    def _jwks_under_issuer(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("jwks_url"):
            issuer = values.get("issuer") or cls.model_fields["issuer"].default
            values["jwks_url"] = f"{issuer}{JWKS_PATH}"
        return values

    @model_validator(mode="after")
    def _no_default_password_off_this_machine(self) -> "Settings":
        default = type(self).model_fields["app_password"].default
        refuse_default_password(self.db_host, "NEEDTOKNOW_APP_PASSWORD", self.app_password, default)
        return self

    @model_validator(mode="after")
    def _both_prices_or_neither(self) -> "Settings":
        if (self.price_input is None) != (self.price_output is None):
            raise ValueError(
                "set both NEEDTOKNOW_PRICE_INPUT and NEEDTOKNOW_PRICE_OUTPUT or neither"
            )
        return self

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

    @field_validator("public_relations", mode="before")
    @classmethod
    def _comma_list(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        names = frozenset(name.strip() for name in value.split(",") if name.strip())
        for name in names:
            if name.count(".") > 1:
                raise ValueError(
                    f"NEEDTOKNOW_PUBLIC_RELATIONS takes schema or schema.relation, not {name!r}"
                )
        return names

    @field_validator("model_cache")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        return value.expanduser()


class ProvisionSettings(BaseSettings):
    model_config = _FROM_ENVIRONMENT

    admin_user: str = "needtoknow"
    admin_password: str = "needtoknow"
    owner_password: str = "needtoknow_owner"
    reader_password: str = "needtoknow_reader"

    def check_for(self, db_host: str) -> None:
        for name in ("admin_password", "owner_password", "reader_password"):
            default = type(self).model_fields[name].default
            variable = f"NEEDTOKNOW_{name.upper()}"
            refuse_default_password(db_host, variable, getattr(self, name), default)


class DefaultPassword(ValueError):
    pass


def refuse_default_password(db_host: str, variable: str, value: str, default: str) -> None:
    if db_host not in LOCAL_HOSTS and value == default:
        raise DefaultPassword(f"{variable} is the development default; set it for {db_host}")


def load_settings() -> Settings:
    return Settings()


def load_provision_settings() -> ProvisionSettings:
    return ProvisionSettings()
