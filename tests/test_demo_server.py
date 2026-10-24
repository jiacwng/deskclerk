# The demo task server driven through deskclerk's own client and agent: a task is created,
# listed and updated, and a delete of an id nobody has seen is refused by the write check.

import re
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from cryptography.fernet import Fernet
from langgraph.checkpoint.memory import InMemorySaver

from deskclerk.agent import run_agent
from deskclerk.connections import OAuthConnections
from deskclerk.screen import pass_through
from deskclerk.servers import McpServer
from deskclerk.writecheck import write_check
from tasks_server import build_server
from test_agent import answer, calls, script
from test_connections import make_connections


@pytest.fixture
def demo_server() -> Iterator[McpServer]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    app = build_server().streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield McpServer("tasks", url, None, None, "none")
    finally:
        server.should_exit = True
        thread.join()
        sock.close()


@pytest.fixture
def connections(tmp_path: Path) -> OAuthConnections:
    return make_connections(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())


@pytest.mark.anyio
async def test_a_task_is_created_listed_and_updated(
    demo_server: McpServer, connections: OAuthConnections
) -> None:
    tools = await connections.tools("ada", demo_server)
    assert {tool.name: (tool.read_only, tool.destructive) for tool in tools} == {
        "list_tasks": (True, False),
        "get_task": (True, False),
        "create_task": (False, False),
        "update_task": (False, False),
        "delete_task": (False, True),
    }
    empty = await connections.call("ada", demo_server, "list_tasks", {})
    assert empty.text == "There are no tasks."
    created = await connections.call("ada", demo_server, "create_task", {"title": "pay rent"})
    match = re.fullmatch(r"created ([0-9a-f]{6}): pay rent \[todo\]", created.text)
    assert match
    task_id = match.group(1)
    updated = await connections.call(
        "ada", demo_server, "update_task", {"task_id": task_id, "status": "done"}
    )
    assert updated.text == f"updated {task_id}: pay rent [done]"
    listed = await connections.call("ada", demo_server, "list_tasks", {})
    assert listed.text == f"{task_id}: pay rent [done]"


@pytest.mark.anyio
async def test_a_delete_of_an_unseen_id_is_refused_by_the_write_check(
    demo_server: McpServer, connections: OAuthConnections
) -> None:
    model = script(calls(("tasks__delete_task", {"task_id": "abc123"})), answer("not deleted"))
    result = await run_agent(
        "ada",
        "delete my task",
        "Be careful.",
        [demo_server],
        connections,
        connections.tokens,
        InMemorySaver(),
        "ada/demo",
        [],
        model,
        write_check(),
        pass_through,
        5,
        lambda usage: None,
    )
    assert result.executed == ()
    [blocked] = result.blocked
    assert blocked.tool == "delete_task"
    assert "id it has not seen" in blocked.reason
