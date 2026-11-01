# The token key, the session key and the audit chain key are made on the first start, kept in the
# data directory and read again later, and a variable that is set always wins.

import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from deskclerk.config import Settings
from deskclerk.generated_keys import fill_missing_keys
from deskclerk.tokens import multi_fernet


def environment(data_dir: Path, **variables: str) -> dict[str, str]:
    return {"DESKCLERK_DATA_DIR": str(data_dir), **variables}


def test_missing_keys_are_made_and_kept_in_files_only_the_owner_can_read(tmp_path: Path) -> None:
    environ = environment(tmp_path / "keys")
    fill_missing_keys(environ)
    for variable, file_name in (
        ("DESKCLERK_TOKEN_KEY", "token_key"),
        ("DESKCLERK_SESSION_KEY", "session_key"),
        ("DESKCLERK_AUDIT_CHAIN_KEY", "audit_chain_key"),
    ):
        path = tmp_path / "keys" / file_name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.read_text().strip() == environ[variable]
    Fernet(environ["DESKCLERK_TOKEN_KEY"])
    assert Settings(session_key=SecretStr(environ["DESKCLERK_SESSION_KEY"])).session_key


def test_the_audit_chain_key_is_made_once_and_read_again(tmp_path: Path) -> None:
    first = environment(tmp_path)
    fill_missing_keys(first)
    second = environment(tmp_path)
    fill_missing_keys(second)
    key = (tmp_path / "audit_chain_key").read_text().strip()
    assert len(key) >= 32
    assert first["DESKCLERK_AUDIT_CHAIN_KEY"] == key == second["DESKCLERK_AUDIT_CHAIN_KEY"]


def test_the_next_start_reads_the_files_it_made(tmp_path: Path) -> None:
    first = environment(tmp_path)
    fill_missing_keys(first)
    second = environment(tmp_path)
    fill_missing_keys(second)
    assert second == first


def test_a_variable_that_is_set_wins_and_leaves_no_file(tmp_path: Path) -> None:
    key = Fernet.generate_key().decode()
    environ = environment(tmp_path, DESKCLERK_TOKEN_KEY=key)
    fill_missing_keys(environ)
    assert environ["DESKCLERK_TOKEN_KEY"] == key
    assert not (tmp_path / "token_key").exists()
    assert (tmp_path / "session_key").exists()
    assert (tmp_path / "audit_chain_key").exists()


def test_an_empty_variable_counts_as_unset(tmp_path: Path) -> None:
    environ = environment(tmp_path, DESKCLERK_TOKEN_KEY="")
    fill_missing_keys(environ)
    assert environ["DESKCLERK_TOKEN_KEY"] == (tmp_path / "token_key").read_text().strip()


def test_starts_at_the_same_moment_end_with_one_key_that_all_read(tmp_path: Path) -> None:
    def start(_: int) -> str:
        environ = environment(tmp_path)
        fill_missing_keys(environ)
        return environ["DESKCLERK_TOKEN_KEY"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        keys = list(pool.map(start, range(64)))
    assert len(set(keys)) == 1
    assert keys[0] == (tmp_path / "token_key").read_text().strip()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "audit_chain_key",
        "session_key",
        "token_key",
    ]


def test_a_made_key_is_one_key_and_a_list_in_the_variable_still_wins(tmp_path: Path) -> None:
    environ = environment(tmp_path)
    fill_missing_keys(environ)
    assert "," not in environ["DESKCLERK_TOKEN_KEY"]
    keys = f"{Fernet.generate_key().decode()},{environ['DESKCLERK_TOKEN_KEY']}"
    listed = environment(tmp_path, DESKCLERK_TOKEN_KEY=keys)
    fill_missing_keys(listed)
    assert listed["DESKCLERK_TOKEN_KEY"] == keys
    multi_fernet(keys)


def test_the_data_directory_defaults_to_data() -> None:
    assert Settings.model_fields["data_dir"].default == Path("data")


def test_the_unscreened_allowance_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKCLERK_ALLOW_UNSCREENED", "true")
    assert not hasattr(Settings(), "allow_unscreened")
