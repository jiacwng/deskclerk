# A demo MCP server for deskclerk: a task list held in memory, empty at start, over Streamable HTTP
# on port 8100 at /mcp with no sign-in.

import secrets
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

Status = Literal["todo", "doing", "done"]


def build_server() -> MCPServer:
    mcp = MCPServer("tasks")
    tasks: dict[str, dict[str, str]] = {}

    def describe(task_id: str) -> str:
        task = tasks[task_id]
        return f"{task_id}: {task['title']} [{task['status']}]"

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    def list_tasks() -> str:
        if not tasks:
            return "There are no tasks."
        return "\n".join(describe(task_id) for task_id in tasks)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    def get_task(task_id: str) -> str:
        if task_id not in tasks:
            raise ValueError(f"no task with id {task_id}")
        return describe(task_id)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
    def create_task(title: str) -> str:
        task_id = secrets.token_hex(3)
        tasks[task_id] = {"title": title, "status": "todo"}
        return f"created {describe(task_id)}"

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
    def update_task(task_id: str, title: str | None = None, status: Status | None = None) -> str:
        if task_id not in tasks:
            raise ValueError(f"no task with id {task_id}")
        if title is not None:
            tasks[task_id]["title"] = title
        if status is not None:
            tasks[task_id]["status"] = status
        return f"updated {describe(task_id)}"

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
    def delete_task(task_id: str) -> str:
        if task_id not in tasks:
            raise ValueError(f"no task with id {task_id}")
        del tasks[task_id]
        return f"deleted {task_id}"

    return mcp


if __name__ == "__main__":
    build_server().run("streamable-http", host="0.0.0.0", port=8100)  # noqa: S104
