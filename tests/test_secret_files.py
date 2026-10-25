# NAME_FILE supplies the secret NAME from a real file, and the server list adds its own variables.

from pathlib import Path

import pytest

from deskclerk.secret_files import SecretFileError, read_secret_files


def test_the_file_sets_the_variable_and_a_trailing_newline_is_stripped(tmp_path: Path) -> None:
    secret = tmp_path / "token_key"
    secret.write_text("the-key\n")
    environ = {"DESKCLERK_TOKEN_KEY_FILE": str(secret)}
    read_secret_files(environ)
    assert environ["DESKCLERK_TOKEN_KEY"] == "the-key"


def test_setting_both_is_an_error_naming_both(tmp_path: Path) -> None:
    secret = tmp_path / "token_key"
    secret.write_text("the-key")
    environ = {"DESKCLERK_TOKEN_KEY": "x", "DESKCLERK_TOKEN_KEY_FILE": str(secret)}
    with pytest.raises(SecretFileError, match="DESKCLERK_TOKEN_KEY and DESKCLERK_TOKEN_KEY_FILE"):
        read_secret_files(environ)


def test_a_missing_file_is_an_error_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(SecretFileError, match="absent"):
        read_secret_files({"OPENAI_API_KEY_FILE": str(missing)})


def test_variables_named_in_the_server_list_can_come_from_files(tmp_path: Path) -> None:
    (tmp_path / "pass").write_text("s3cret\n")
    (tmp_path / "bearer").write_text("tok\n")
    servers = tmp_path / "servers.toml"
    servers.write_text(
        '[[server]]\nname = "crm"\nurl = "https://crm.example.com/mcp"\nclient_id = "c"\n'
        'client_secret_env = "CRM_SECRET"\n'
        'headers = { X-Api-Key = "${env:CRM_BEARER}" }\n'
    )
    environ = {
        "DESKCLERK_SERVERS_FILE": str(servers),
        "CRM_SECRET_FILE": str(tmp_path / "pass"),
        "CRM_BEARER_FILE": str(tmp_path / "bearer"),
    }
    read_secret_files(environ)
    assert environ["CRM_SECRET"] == "s3cret"
    assert environ["CRM_BEARER"] == "tok"
