# Settings refuse a development password for a database that is not on this machine, a model
# named without its provider, and one price setting without the other.

import pytest
from pydantic import ValidationError

from needtoknow.config import DefaultPassword, ProvisionSettings, Settings


def test_a_remote_database_needs_a_real_app_password() -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_APP_PASSWORD"):
        Settings(db_host="db.example.com")

    assert Settings(db_host="db.example.com", app_password="s3cret").db_host


@pytest.mark.parametrize("model_name", ["claude-haiku-5-5", ":claude-haiku-5-5", "anthropic:"])
def test_the_model_needs_a_provider_and_a_name(model_name: str) -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_MODEL is provider:model"):
        Settings(model=model_name)


def test_one_price_without_the_other_is_refused() -> None:
    with pytest.raises(ValidationError, match="set both NEEDTOKNOW_PRICE_INPUT"):
        Settings(price_input=1.0)
    with pytest.raises(ValidationError, match="set both NEEDTOKNOW_PRICE_INPUT"):
        Settings(price_output=1.0)


def test_a_remote_database_needs_real_provision_passwords() -> None:
    with pytest.raises(DefaultPassword, match="NEEDTOKNOW_ADMIN_PASSWORD"):
        ProvisionSettings().check_for("db.example.com")

    ProvisionSettings().check_for("localhost")
