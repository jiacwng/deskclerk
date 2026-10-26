# The server list is read from TOML or mcpServers JSON and refused when a name repeats, a remote
# URL is not https, a secret has no client or its environment variable is missing.

import json
from pathlib import Path

import pytest

from deskclerk.servers import McpServer, ServersError, load_servers


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "servers.toml"
    path.write_text(text)
    return path


def write_json(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "servers.json"
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


@pytest.mark.parametrize("name", ["google drive", "jira.cloud", "a__b", ""])
def test_a_server_name_the_model_cannot_take_is_refused(tmp_path: Path, name: str) -> None:
    path = write(tmp_path, f'[[server]]\nname = "{name}"\nurl = "https://a.test/mcp"\n')
    with pytest.raises(ServersError, match="must use only letters"):
        load_servers(path, {})


@pytest.mark.parametrize("name", ["linear", "google-drive", "jira_cloud"])
def test_a_server_name_with_letters_hyphens_and_single_underscores_is_read(
    tmp_path: Path, name: str
) -> None:
    path = write(tmp_path, f'[[server]]\nname = "{name}"\nurl = "https://a.test/mcp"\n')
    assert load_servers(path, {})[0].name == name


def test_a_repeated_name_is_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
        '[[server]]\nname = "a"\nurl = "https://b.test/mcp"\n',
    )
    with pytest.raises(ServersError, match="listed twice"):
        load_servers(path, {})


@pytest.mark.parametrize("url", ["http://example.com/mcp", "example.com"])
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


def test_auth_none_is_read_and_refuses_a_client(tmp_path: Path) -> None:
    entry = '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\nauth = "none"\n'
    assert load_servers(write(tmp_path, entry), {}) == [
        McpServer("a", "https://a.test/mcp", None, None, "none")
    ]
    with pytest.raises(ServersError, match="and a client"):
        load_servers(write(tmp_path, entry + 'client_id = "x"\n'), {})


def test_a_tool_list_is_read_and_stays_absent_when_not_set(tmp_path: Path) -> None:
    entry = '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
    assert load_servers(write(tmp_path, entry + 'tools = ["x", "y"]\n'), {}) == [
        McpServer("a", "https://a.test/mcp", None, None, "oauth", ("x", "y"))
    ]
    assert load_servers(write(tmp_path, entry), {})[0].tools is None


def test_a_tool_name_must_not_be_empty(tmp_path: Path) -> None:
    entry = '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\ntools = ["x", " "]\n'
    with pytest.raises(ServersError, match="needs a name"):
        load_servers(write(tmp_path, entry), {})


def test_a_repeated_tool_name_is_refused(tmp_path: Path) -> None:
    entry = '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\ntools = ["x", "x"]\n'
    with pytest.raises(ServersError, match="tool name"):
        load_servers(write(tmp_path, entry), {})


def test_an_mcp_json_file_gives_the_same_servers_as_toml(tmp_path: Path) -> None:
    toml = write(
        tmp_path,
        '[[server]]\nname = "github"\nurl = "https://example.com/mcp"\nclient_id = "abc"\n'
        'client_secret_env = "GITHUB_SECRET"\ntools = ["x"]\n'
        '[[server]]\nname = "local"\nurl = "http://127.0.0.1:9000/mcp"\nauth = "none"\n',
    )
    json_file = write_json(
        tmp_path,
        json.dumps(
            {
                "mcpServers": {
                    "github": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "client_id": "abc",
                        "client_secret_env": "GITHUB_SECRET",
                        "tools": ["x"],
                    },
                    "local": {"url": "http://127.0.0.1:9000/mcp", "auth": "none"},
                }
            }
        ),
    )
    from_toml = load_servers(toml, {"GITHUB_SECRET": "shh"})
    assert from_toml == [
        McpServer("github", "https://example.com/mcp", "abc", "shh", "oauth", ("x",)),
        McpServer("local", "http://127.0.0.1:9000/mcp", None, None, "none"),
    ]
    assert load_servers(json_file, {"GITHUB_SECRET": "shh"}) == from_toml


def test_a_stdio_server_is_refused_in_mcp_json(tmp_path: Path) -> None:
    path = write_json(
        tmp_path,
        json.dumps({"mcpServers": {"fs": {"command": "npx", "args": ["server-fs"]}}}),
    )
    with pytest.raises(ServersError, match="over HTTP only"):
        load_servers(path, {})


def test_an_invalid_mcp_json_entry_gives_the_same_error_as_toml(tmp_path: Path) -> None:
    toml = write(tmp_path, '[[server]]\nname = "a"\nurl = "http://example.com/mcp"\n')
    json_file = write_json(
        tmp_path,
        json.dumps({"mcpServers": {"a": {"url": "http://example.com/mcp"}}}),
    )
    with pytest.raises(ServersError) as toml_error:
        load_servers(toml, {})
    with pytest.raises(ServersError) as json_error:
        load_servers(json_file, {})
    assert str(json_error.value).replace("servers.json", "servers.toml") == str(toml_error.value)


def test_invalid_mcp_json_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ServersError, match="not valid JSON"):
        load_servers(write_json(tmp_path, "{not json"), {})


def test_headers_are_read_with_environment_values_filled_in(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\nauth = "none"\n'
        '[server.headers]\nX-Api-Key = "${env:A_KEY}"\nX-Team = "ops-${env:A_TEAM}"\n',
    )
    servers = load_servers(path, {"A_KEY": "k1", "A_TEAM": "red"})
    assert servers[0].headers == (("X-Api-Key", "k1"), ("X-Team", "ops-red"))


def test_headers_in_mcp_json_match_toml(tmp_path: Path) -> None:
    path = write_json(
        tmp_path,
        json.dumps(
            {
                "mcpServers": {
                    "a": {
                        "url": "https://a.test/mcp",
                        "auth": "none",
                        "headers": {"X-Api-Key": "${env:A_KEY}"},
                    }
                }
            }
        ),
    )
    assert load_servers(path, {"A_KEY": "k1"})[0].headers == (("X-Api-Key", "k1"),)


def test_a_missing_header_variable_is_named(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\nauth = "none"\n'
        '[server.headers]\nX-Api-Key = "${env:A_KEY}"\n',
    )
    with pytest.raises(ServersError, match="A_KEY"):
        load_servers(path, {})


@pytest.mark.parametrize("name", ["Authorization", "authorization"])
def test_an_authorization_header_is_refused_with_oauth(tmp_path: Path, name: str) -> None:
    path = write(
        tmp_path,
        f'[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n[server.headers]\n{name} = "x"\n',
    )
    with pytest.raises(ServersError, match="Authorization"):
        load_servers(path, {})


def test_an_authorization_header_is_allowed_without_oauth(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\nauth = "none"\n'
        '[server.headers]\nAuthorization = "Bearer x"\n',
    )
    assert load_servers(path, {})[0].headers == (("Authorization", "Bearer x"),)


def test_plain_http_allows_http_on_a_remote_host(tmp_path: Path) -> None:
    refused = write(tmp_path, '[[server]]\nname = "a"\nurl = "http://mesh.internal/mcp"\n')
    with pytest.raises(ServersError, match="must be https"):
        load_servers(refused, {})
    allowed = write(
        tmp_path, '[[server]]\nname = "a"\nurl = "http://mesh.internal/mcp"\nplain_http = true\n'
    )
    assert load_servers(allowed, {})[0].url == "http://mesh.internal/mcp"


def test_a_certificate_without_a_key_is_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\nclient_cert = "c.pem"\n',
    )
    with pytest.raises(ServersError, match="together"):
        load_servers(path, {})


def test_an_unreadable_certificate_is_named(tmp_path: Path) -> None:
    key = tmp_path / "key.pem"
    key.write_text("x")
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
        f'client_cert = "{(tmp_path / "missing.pem").as_posix()}"\n'
        f'client_key = "{key.as_posix()}"\n',
    )
    with pytest.raises(ServersError, match="missing.pem"):
        load_servers(path, {})


def test_a_readable_certificate_pair_is_kept(tmp_path: Path) -> None:
    cert = tmp_path / "c.pem"
    key = tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    path = write(
        tmp_path,
        '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
        f'client_cert = "{cert.as_posix()}"\nclient_key = "{key.as_posix()}"\n',
    )
    server = load_servers(path, {})[0]
    assert (server.client_cert, server.client_key) == (cert.as_posix(), key.as_posix())


def test_a_server_lists_the_groups_that_may_use_it_and_without_them_it_is_for_everyone(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "ledger"\nurl = "https://a.test/mcp"\ngroups = ["finance", "ops"]\n'
        '[[server]]\nname = "wiki"\nurl = "https://b.test/mcp"\n',
    )
    ledger, wiki = load_servers(path, {})
    assert ledger.groups == ("finance", "ops")
    assert (ledger.open_to(frozenset({"ops"})), ledger.open_to(frozenset({"hr"}))) == (True, False)
    assert ledger.open_to(frozenset()) is False
    assert wiki.groups is None
    assert wiki.open_to(frozenset()) is True


def test_groups_are_read_from_the_mcp_json_file(tmp_path: Path) -> None:
    path = write_json(
        tmp_path,
        '{"mcpServers": {"ledger": {"url": "https://a.test/mcp", "groups": ["finance"]}}}',
    )
    assert load_servers(path, {})[0].groups == ("finance",)


@pytest.mark.parametrize("groups", ["[]", '[" "]'])
def test_a_server_with_an_empty_group_list_is_refused(tmp_path: Path, groups: str) -> None:
    path = write(
        tmp_path, f'[[server]]\nname = "a"\nurl = "https://a.test/mcp"\ngroups = {groups}\n'
    )
    with pytest.raises(ServersError, match="groups"):
        load_servers(path, {})


def test_a_server_allows_always_allow_unless_the_file_forbids_it(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[server]]\nname = "ledger"\nurl = "https://a.test/mcp"\nalways_allow = false\n'
        '[[server]]\nname = "wiki"\nurl = "https://b.test/mcp"\n',
    )
    ledger, wiki = load_servers(path, {})
    assert (ledger.always_allow, wiki.always_allow) == (False, True)


def test_confirm_tools_are_read_and_default_to_none(tmp_path: Path) -> None:
    entry = '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
    forced = load_servers(write(tmp_path, entry + 'confirm_tools = ["send", "wipe"]\n'), {})[0]
    assert forced.confirm_tools == ("send", "wipe")
    assert load_servers(write(tmp_path, entry), {})[0].confirm_tools == ()


def test_a_confirm_tool_name_must_not_be_empty_or_repeated(tmp_path: Path) -> None:
    entry = '[[server]]\nname = "a"\nurl = "https://a.test/mcp"\n'
    with pytest.raises(ServersError, match="needs a name"):
        load_servers(write(tmp_path, entry + 'confirm_tools = [" "]\n'), {})
    with pytest.raises(ServersError, match="listed twice"):
        load_servers(write(tmp_path, entry + 'confirm_tools = ["x", "x"]\n'), {})
