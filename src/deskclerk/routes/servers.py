# The routes for connecting the employee's MCP servers: the list of servers they may use, the
# OAuth sign-in to a server and its callback, disconnecting, and revoking an always-allowed tool.

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from deskclerk.connections import (
    UPSTREAM_ERRORS,
    AlreadyConnected,
    FlowExpired,
    NoSignInNeeded,
    UnknownState,
    WrongUser,
    first_leaf,
)
from deskclerk.pinning import MenuEntry, allowed_tool_names, read_menu, revoke_tool
from deskclerk.routes.services import MAX_REASON_CHARS, Caller, Services, clip, link_failure

CONNECT_ERRORS: dict[type[Exception], int] = {
    UnknownState: 400,
    WrongUser: 403,
    FlowExpired: 410,
    AlreadyConnected: 409,
    NoSignInNeeded: 400,
}

_log = logging.getLogger(__name__)


class ServerOut(BaseModel):
    name: str
    url: str
    title: str | None
    description: str | None
    examples: list[str]
    connected: bool
    needs_sign_in: bool
    tools: list[MenuEntry]
    always_allowed: list[str]


class ConnectOut(BaseModel):
    authorization_url: str


def _upstream_reason(error: BaseException) -> str:
    leaf = first_leaf(error)
    return f"{type(leaf).__name__}: {str(leaf)[:MAX_REASON_CHARS]}"


def _connect_failed(failure: Exception) -> RedirectResponse:
    _log.warning("connect failed: %s", failure)
    expired = isinstance(failure, FlowExpired | UnknownState)
    return link_failure("connect_expired" if expired else "connect_failed")


def router(services: Services) -> APIRouter:
    routes = APIRouter()
    tokens = services.tokens
    connections = services.connections

    @routes.get("/servers")
    async def list_servers(who: Caller = Depends(services.caller)) -> list[ServerOut]:
        user = who.identity.user_id
        listed = []
        for server in services.catalog.offered:
            if not server.open_to(who.identity.groups):
                continue
            connected = (
                server.auth == "none"
                or await run_in_threadpool(tokens.get, user, server.name, "tokens") is not None
            )
            menu = (
                await run_in_threadpool(read_menu, tokens, user, server.name) if connected else []
            )
            allowed = [t for t in menu if server.tools is None or t.name in server.tools]
            menu_names = {t.name for t in allowed}
            always = (
                [
                    name
                    for name in await run_in_threadpool(
                        allowed_tool_names, tokens, user, server.name
                    )
                    if name in menu_names
                ]
                if connected
                else []
            )
            listed.append(
                ServerOut(
                    name=server.name,
                    url=server.url,
                    title=server.title,
                    description=server.description,
                    examples=list(server.examples),
                    connected=connected,
                    needs_sign_in=server.auth != "none",
                    tools=allowed,
                    always_allowed=always,
                )
            )
        return listed

    @routes.post("/servers/{name}/connect")
    async def connect(name: str, who: Caller = Depends(services.posting_caller)) -> ConnectOut:
        server = services.known_server(name, who.identity)
        try:
            url = await connections.begin(who.identity.user_id, server)
        except tuple(CONNECT_ERRORS) as error:
            raise HTTPException(CONNECT_ERRORS[type(error)], str(error)) from None
        except UPSTREAM_ERRORS as failure:
            _log.warning("connect to %s failed to start: %s", name, _upstream_reason(failure))
            raise HTTPException(502, f"cannot start the sign-in to {name}") from None
        return ConnectOut(authorization_url=url)

    @routes.get("/oauth/client.json")
    async def oauth_client_document() -> JSONResponse:
        document = connections.client_metadata().model_dump(mode="json", exclude_none=True)
        return JSONResponse(
            document | {"client_id": services.settings.client_metadata_url},
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @routes.get("/oauth/callback")
    async def oauth_callback(
        state: str,
        code: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
        iss: str | None = None,
        who: Caller = Depends(services.caller),
    ) -> Response:
        if error is not None or code is None:
            try:
                await connections.cancel(who.identity.user_id, state)
            except tuple(CONNECT_ERRORS) as refused:
                return _connect_failed(refused)
            _log.warning(
                "connect refused by the server: error=%s error_description=%s",
                clip(error),
                clip(error_description),
            )
            return link_failure("connect_failed")
        try:
            server_name = await connections.complete(who.identity.user_id, state, code, iss)
        except tuple(CONNECT_ERRORS) as failure:
            return _connect_failed(failure)
        except UPSTREAM_ERRORS as failure:
            _log.warning("connect failed to complete: %s", _upstream_reason(failure))
            return link_failure("connect_failed")
        await run_in_threadpool(services.audit.connection, who.identity, "connect", server_name)
        return RedirectResponse("/", status_code=303)

    @routes.post("/servers/{name}/disconnect", status_code=204)
    async def disconnect(name: str, who: Caller = Depends(services.posting_caller)) -> Response:
        server = services.known_server(name, who.identity)
        await connections.cancel_for(who.identity.user_id, server.name)
        await run_in_threadpool(tokens.delete_connection, who.identity.user_id, server.name)
        await run_in_threadpool(services.audit.connection, who.identity, "disconnect", server.name)
        return Response(status_code=204)

    @routes.delete("/servers/{name}/always/{tool}", status_code=204)
    async def revoke_always_allowed(
        name: str, tool: str, who: Caller = Depends(services.posting_caller)
    ) -> Response:
        server = services.known_server(name, who.identity)
        user = who.identity.user_id
        async with services.database.user_lock(user):
            revoked = await run_in_threadpool(revoke_tool, tokens, user, server.name, tool)
        if not revoked:
            raise HTTPException(404, f"{tool} is not always allowed")
        return Response(status_code=204)

    return routes
