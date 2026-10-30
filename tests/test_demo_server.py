# The demo server (the official MCP git server on a clone of this repository): the example server
# list names it, and, when DESKCLERK_DEMO_URL points at a running copy, its tools carry the hints
# that decide which calls need a confirmation.

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from deskclerk.connections import OAuthConnections
from deskclerk.database import Database
from deskclerk.servers import McpServer, load_servers
from test_connections import serving_connections

ROOT = Path(__file__).parent.parent
SANDBOX = "/sandbox/deskclerk"


def test_the_example_server_list_names_the_demo_server() -> None:
    [server] = load_servers(ROOT / "servers.example.toml", {})
    assert (server.name, server.url, server.auth) == ("repo", "http://localhost:8100/mcp", "none")


def test_the_demo_image_serves_the_git_server_on_the_clone_only() -> None:
    dockerfile = (ROOT / "demo" / "Dockerfile").read_text()
    assert "git clone" in dockerfile
    assert '"mcp-server-git", "--repository", "/sandbox/deskclerk"' in dockerfile
    assert "--require-hashes -r /tmp/requirements.lock" in dockerfile
    lock = (ROOT / "demo" / "requirements.lock").read_text()
    assert "\nmcp-server-git==2026.10.10 \\\n    --hash=sha256:" in lock


@pytest.fixture
def demo_server() -> McpServer:
    url = os.environ.get("DESKCLERK_DEMO_URL")
    if url is None:
        pytest.skip("DESKCLERK_DEMO_URL is not set")
    return McpServer("repo", url, None, None, "none", trusted=True)


@pytest.fixture
async def connections(database: Database) -> AsyncIterator[OAuthConnections]:
    async with serving_connections(database, Fernet.generate_key().decode()) as running:
        yield running


@pytest.mark.anyio
async def test_reads_are_read_only_and_changes_are_not(
    demo_server: McpServer, connections: OAuthConnections
) -> None:
    tools = {tool.name: tool for tool in await connections.tools("ada", demo_server)}
    for name in ("git_log", "git_show", "git_diff", "git_status"):
        assert tools[name].read_only
    for name in ("git_create_branch", "git_add", "git_commit", "git_checkout", "git_reset"):
        assert not tools[name].read_only
    assert tools["git_reset"].destructive


@pytest.mark.anyio
async def test_the_log_is_read_from_the_clone_and_other_paths_are_refused(
    demo_server: McpServer, connections: OAuthConnections
) -> None:
    log = await connections.call(
        "ada", demo_server, "git_log", {"repo_path": SANDBOX, "max_count": 3}
    )
    assert log.text.startswith("Commit history:")
    outside = await connections.call("ada", demo_server, "git_status", {"repo_path": "/etc"})
    assert "outside the allowed repository" in outside.text
