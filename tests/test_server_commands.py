# The server list lives in Postgres: import-servers replaces it from a TOML or mcpServers JSON file
# in one step, export-servers prints it as TOML, and a copy keeps its previous list when a stored
# entry cannot be loaded.

from pathlib import Path

import pytest

from deskclerk import __main__ as entry
from deskclerk.database import Database
from deskclerk.servers import ServerEntry, read_entries
from deskclerk.serverstore import ServerCatalog, ServerStore

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "servers.example.toml"


def export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> list[ServerEntry]:
    capsys.readouterr()
    entry.main(["export-servers"])
    exported = tmp_path / "exported.toml"
    exported.write_text(capsys.readouterr().out)
    return read_entries(exported)


@pytest.mark.usefixtures("database")
def test_importing_the_example_file_and_exporting_it_gives_the_same_servers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry.main(["import-servers", str(EXAMPLE)])
    assert export(tmp_path, capsys) == read_entries(EXAMPLE)


@pytest.mark.usefixtures("database")
def test_an_import_replaces_the_whole_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry.main(["import-servers", str(EXAMPLE)])
    replacement = tmp_path / "mcp.json"
    replacement.write_text('{"mcpServers": {"wiki": {"url": "https://wiki.test/mcp"}}}')
    entry.main(["import-servers", str(replacement)])
    assert [found.name for found in export(tmp_path, capsys)] == ["wiki"]


@pytest.mark.usefixtures("database")
def test_a_file_with_an_invalid_entry_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry.main(["import-servers", str(EXAMPLE)])
    invalid = tmp_path / "servers.toml"
    invalid.write_text('[[server]]\nname = "wiki"\nurl = "http://wiki.example.com/mcp"\n')
    with pytest.raises(SystemExit) as refused:
        entry.main(["import-servers", str(invalid)])
    assert "must be https" in str(refused.value.code)
    assert export(tmp_path, capsys) == read_entries(EXAMPLE)


@pytest.mark.anyio
async def test_a_stored_entry_that_cannot_be_loaded_leaves_the_previous_list(
    database: Database,
) -> None:
    store = ServerStore(database)
    catalog = ServerCatalog(database, store, {})
    store.add(ServerEntry(name="first", url="https://first.test/mcp"))
    await catalog.refresh()
    broken = ServerEntry(
        name="second", url="https://second.test/mcp", client_id="c", client_secret_env="NOT_SET"
    )
    store.add(broken)
    await catalog.refresh()
    assert [server.name for server in catalog.servers] == ["first"]
