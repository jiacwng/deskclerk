# The server list is read from TOML and refused when a name repeats, a remote URL is not https, a
# secret has no client or its environment variable is missing.

from pathlib import Path

import pytest

from needtoknow.servers import McpServer, ServersError, load_servers


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "servers.toml"
    path.write_text(text)
    return path


def test_servers_are_read_with_the_secret_from_the_environment(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
[[server]]
name = "github"
url = "https://example.com/mcp"
client_id = "abc"
client_secret_env = "GITHUB_SECRET"

[[server]]
name = "local"
url = "http://127.0.0.1:9000/mcp"
""",
    )
    servers = load_servers(path, {"GITHUB_SECRET": "shh"})
    assert servers == [
        McpServer("github", "https://example.com/mcp", "abc", "shh"),
        McpServer("local", "http://127.0.0.1:9000/mcp", None, None),
    ]


def test_a_repeated_name_is_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
        '[[server]]\nname = "a"\nurl = "https://b.test/mcp"\n',
    )
    with pytest.raises(ServersError, match="listed twice"):
        load_servers(path, {})


@pytest.mark.parametrize("url", ["http://example.com/mcp", "ftp://example.com/mcp", "example.com"])
def test_a_remote_url_must_be_https(tmp_path: Path, url: str) -> None:
    path = write(tmp_path, f'[[server]]\nname = "a"\nurl = "{url}"\n')
    with pytest.raises(ServersError, match="must be https"):
        load_servers(path, {})


def test_a_missing_secret_variable_is_named(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
        'client_id = "abc"\nclient_secret_env = "NO_SUCH_SECRET"\n',
    )
    with pytest.raises(ServersError, match="NO_SUCH_SECRET"):
        load_servers(path, {})


def test_a_secret_without_a_client_id_is_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\nclient_secret_env = "X"\n',
    )
    with pytest.raises(ServersError, match="without client_id"):
        load_servers(path, {"X": "y"})


def test_a_missing_file_is_an_error_naming_it(tmp_path: Path) -> None:
    with pytest.raises(ServersError, match="nothing.toml"):
        load_servers(tmp_path / "nothing.toml", {})
