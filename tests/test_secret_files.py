# NAME_FILE supplies the secret NAME from a real file.

from pathlib import Path

import pytest

from deskclerk.secret_files import SecretFileError, read_secret_files


def test_the_file_sets_the_variable_and_a_trailing_newline_is_stripped(tmp_path: Path) -> None:
    secret = tmp_path / "token_key"
    secret.write_text("the-key\n")
    environ = {"DESKCLERK_TOKEN_KEY_FILE": str(secret)}
    read_secret_files(environ)
    assert environ["DESKCLERK_TOKEN_KEY"] == "the-key"


def test_the_audit_chain_key_can_come_from_a_file(tmp_path: Path) -> None:
    secret = tmp_path / "audit_chain_key"
    secret.write_text("the-chain-key\n")
    environ = {"DESKCLERK_AUDIT_CHAIN_KEY_FILE": str(secret)}
    read_secret_files(environ)
    assert environ["DESKCLERK_AUDIT_CHAIN_KEY"] == "the-chain-key"


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
