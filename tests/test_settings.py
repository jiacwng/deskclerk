# Settings refuse a development password for a database that is not on this machine, a model
# named without its provider, one price setting without the other, Graph credentials or the
# company database given in part, and a public relation named with more than a schema and a
# relation.

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


def test_the_audience_is_the_client_unless_set() -> None:
    assert Settings(client_id="page").audience == "page"
    assert Settings(client_id="page", audience="api://default").audience == "api://default"


@pytest.mark.parametrize(
    "given",
    [
        {"data_host": "sales.example.com"},
        {"data_host": "sales.example.com", "data_name": "sales"},
        {"data_name": "sales", "data_password": "s3cret"},
        {"data_port": 5433},
    ],
)
def test_the_company_database_is_set_in_full_or_not_at_all(given: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_DATA_PASSWORD together"):
        Settings.model_validate(given)

    full = Settings(data_host="sales.example.com", data_name="sales", data_password="s3cret")
    assert full.data_port == 5432


def test_a_remote_company_database_needs_a_real_app_password() -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_DATA_PASSWORD is the development"):
        Settings(data_host="sales.example.com", data_name="sales", data_password="needtoknow_app")

    Settings(data_host="localhost", data_name="sales", data_password="needtoknow_app")


def test_graph_credentials_come_all_three_or_none() -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_GRAPH_CLIENT_SECRET together"):
        Settings(graph_tenant_id="tenant", graph_client_id="client")

    complete = Settings(
        graph_tenant_id="tenant", graph_client_id="client", graph_client_secret="hunter2"
    )
    assert "hunter2" not in repr(complete)


def test_a_remote_database_needs_real_provision_passwords() -> None:
    with pytest.raises(DefaultPassword, match="NEEDTOKNOW_ADMIN_PASSWORD"):
        ProvisionSettings().check_for("db.example.com", evaluation=False)

    ProvisionSettings().check_for("localhost", evaluation=True)


def test_the_reader_password_is_checked_only_for_the_evaluation() -> None:
    provision = ProvisionSettings(
        admin_password="real", owner_password="real", reader_password="needtoknow_reader"
    )

    provision.check_for("db.example.com", evaluation=False)
    with pytest.raises(DefaultPassword, match="NEEDTOKNOW_READER_PASSWORD"):
        provision.check_for("db.example.com", evaluation=True)


def test_public_relations_are_a_comma_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEEDTOKNOW_PUBLIC_RELATIONS", " handbook, sales.price_list ,")

    assert Settings().public_relations == frozenset({"handbook", "sales.price_list"})

    monkeypatch.setenv("NEEDTOKNOW_PUBLIC_RELATIONS", "")
    assert Settings().public_relations == frozenset()


def test_a_public_relation_takes_at_most_one_dot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEEDTOKNOW_PUBLIC_RELATIONS", "db.sales.price_list")

    with pytest.raises(ValidationError, match="schema or schema.relation"):
        Settings()
