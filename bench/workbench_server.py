# Serves the WorkBench tools over MCP (streamable HTTP, no auth) as five servers, one per domain,
# at /<server>/mcp on one port, and logs each executed call in the string format WorkBench's
# grader replays. The five servers share WorkBench's state and one call log.
import argparse
import importlib
import inspect
import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from workbench import SERVERS, server_of

TOOL_ERRORS = (KeyError, ValueError, TypeError, IndexError, AttributeError)


def _annotations(workbench_tool: Any, side_effect_names: set[str]) -> ToolAnnotations:
    writes = workbench_tool.name in side_effect_names
    return ToolAnnotations(
        readOnlyHint=not writes,
        destructiveHint=("delete_" in workbench_tool.name) if writes else None,
        openWorldHint=False,
    )


def _make_handler(workbench_tool: Any, calls: list[str], to_call_string: Any) -> Any:
    signature = inspect.signature(workbench_tool.func)
    defaults = {name: parameter.default for name, parameter in signature.parameters.items()}

    async def handler(**arguments: Any) -> str:
        given = {name: value for name, value in arguments.items() if value != defaults[name]}
        try:
            result = str(workbench_tool(**given))
        except TOOL_ERRORS as error:
            raise ToolError(f"{type(error).__name__}: {error}") from error
        calls.append(to_call_string(workbench_tool.name, given))
        return result

    handler.__signature__ = signature  # type: ignore[attr-defined]
    return handler


def create_app(workbench_path: str) -> Starlette:
    workbench_path = os.path.abspath(workbench_path)
    sys.path.insert(0, workbench_path)
    os.chdir(workbench_path)
    actions = importlib.import_module("src.evals.actions")
    state = importlib.import_module("src.tools.state")
    toolkits = importlib.import_module("src.tools.toolkits")

    side_effect_names = {tool.name for tool in toolkits.tools_with_side_effects}
    calls: list[str] = []
    servers = {name: MCPServer(name) for name in SERVERS}

    for workbench_tool in toolkits.all_tools:
        handler = _make_handler(
            workbench_tool, calls, actions.convert_intermediate_step_to_function_call
        )
        servers[server_of(workbench_tool.name)].add_tool(
            handler,
            name=workbench_tool.name.split(".", 1)[1],
            description=workbench_tool.description,
            annotations=_annotations(workbench_tool, side_effect_names),
            structured_output=False,
        )

    async def reset(request: Request) -> Response:
        state.reset_state()
        calls.clear()
        return JSONResponse({"calls": 0})

    async def get_calls(request: Request) -> Response:
        return JSONResponse(calls)

    apps = {name: server.streamable_http_app() for name, server in servers.items()}

    @asynccontextmanager
    async def lifespan(parent: Starlette) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for app in apps.values():
                await stack.enter_async_context(app.router.lifespan_context(app))
            yield

    routes = [
        Route("/bench/reset", reset, methods=["POST"]),
        Route("/bench/calls", get_calls, methods=["GET"]),
        *(Mount(f"/{name}", app=app) for name, app in apps.items()),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbench", required=True, help="path to the WorkBench checkout")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    uvicorn.run(create_app(arguments.workbench), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
