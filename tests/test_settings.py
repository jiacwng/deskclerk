# Settings refuse a development password for a database that is not on this machine.

import pytest
from pydantic import ValidationError

from needtoknow.config import DefaultPassword, ProvisionSettings, Settings


def test_a_remote_database_needs_a_real_app_password() -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_APP_PASSWORD"):
        Settings(db_host="db.example.com")

    assert Settings(db_host="db.example.com", app_password="s3cret").db_host


def test_a_remote_database_needs_real_provision_passwords() -> None:
    with pytest.raises(DefaultPassword, match="NEEDTOKNOW_ADMIN_PASSWORD"):
        ProvisionSettings().check_for("db.example.com")

    ProvisionSettings().check_for("localhost")
