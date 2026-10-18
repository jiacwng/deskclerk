# Serves the WorkBench tools over MCP (streamable HTTP, no auth) and logs each executed call
# in the string format WorkBench's grader replays.
import argparse
import importlib
import inspect
import os
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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


def create_server(workbench_path: str) -> MCPServer:
    workbench_path = os.path.abspath(workbench_path)
    sys.path.insert(0, workbench_path)
    os.chdir(workbench_path)
    actions = importlib.import_module("src.evals.actions")
    state = importlib.import_module("src.tools.state")
    toolkits = importlib.import_module("src.tools.toolkits")

    side_effect_names = {tool.name for tool in toolkits.tools_with_side_effects}
    calls: list[str] = []
    server = MCPServer("workbench")

    for workbench_tool in toolkits.all_tools:
        handler = _make_handler(
            workbench_tool, calls, actions.convert_intermediate_step_to_function_call
        )
        server.add_tool(
            handler,
            name=workbench_tool.name.replace(".", "__"),
            description=workbench_tool.description,
            annotations=_annotations(workbench_tool, side_effect_names),
            structured_output=False,
        )

    @server.custom_route("/bench/reset", methods=["POST"])
    async def reset(request: Request) -> Response:
        state.reset_state()
        calls.clear()
        return JSONResponse({"calls": 0})

    @server.custom_route("/bench/calls", methods=["GET"])
    async def get_calls(request: Request) -> Response:
        return JSONResponse(calls)

    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbench", required=True, help="path to the WorkBench checkout")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    server = create_server(arguments.workbench)
    server.run("streamable-http", host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
