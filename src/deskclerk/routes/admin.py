# The admin page and its JSON routes: usage and recent asks, the searchable audit log with its
# chain check and download, server changes, and revoking an employee. Only the admins group may use
# them, and create_app includes this router only when that group is configured.

from collections.abc import Callable
from datetime import UTC, datetime
from importlib import resources

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from deskclerk.admin import (
    AdminAuditPage,
    AdminChain,
    AdminSummary,
    audit_page,
    audit_row,
    recent_rows,
    server_rows,
    usage_rows,
)
from deskclerk.auditstore import AuditFilter
from deskclerk.routes.health import page_headers
from deskclerk.routes.services import Caller, Services
from deskclerk.servers import ServerEntry, ServersError
from deskclerk.serverstore import NoSuchServer, ServerExists
from deskclerk.spending import month_of

AUDIT_PAGE_SIZE = 50
AUDIT_PAGE_MAXIMUM = 200


class RevokeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)


class RevokeOut(BaseModel):
    tokens: int
    sessions: int


def audit_filter(
    employee: str | None = None,
    entry: str | None = None,
    server: str | None = None,
    tool: str | None = None,
    decision: str | None = None,
    flagged: bool | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> AuditFilter:
    return AuditFilter(employee, entry, server, tool, decision, flagged, since, until)


def router(services: Services) -> APIRouter:
    routes = APIRouter()
    catalog = services.catalog
    admin_page = resources.files("deskclerk").joinpath("static/admin.html").read_text("utf-8")
    headers = page_headers(admin_page, {})

    @routes.get("/admin", response_class=HTMLResponse)
    async def admin(request: Request) -> Response:
        try:
            await services.admin_caller(request)
        except HTTPException as refused:
            if refused.status_code == 401:
                return RedirectResponse("/login", status_code=302)
            raise
        return HTMLResponse(admin_page, headers=headers)

    @routes.get("/admin/summary")
    async def admin_summary(_: Caller = Depends(services.admin_caller)) -> AdminSummary:
        month = month_of(datetime.now(UTC))
        return AdminSummary(
            month=month,
            servers=await run_in_threadpool(server_rows, list(catalog.servers), services.tokens),
            usage=await run_in_threadpool(
                usage_rows, services.spending_store, month, services.settings.budget_usd
            ),
            recent=await recent_rows(services.database, services.settings.audit_text),
        )

    @routes.get("/admin/audit")
    async def admin_audit(
        found: AuditFilter = Depends(audit_filter),
        limit: int = Query(AUDIT_PAGE_SIZE, ge=1, le=AUDIT_PAGE_MAXIMUM),
        cursor: int | None = None,
        _: Caller = Depends(services.admin_caller),
    ) -> AdminAuditPage:
        rows = await run_in_threadpool(services.audit_store.search, found, limit, cursor)
        return audit_page(rows)

    @routes.get("/admin/audit/verify")
    async def admin_audit_verify(_: Caller = Depends(services.admin_caller)) -> AdminChain:
        report = await run_in_threadpool(services.audit_store.verify)
        return AdminChain(
            ok=report.broken_at is None, rows=report.rows, first_broken=report.broken_at
        )

    @routes.get("/admin/audit/download")
    async def admin_audit_download(
        found: AuditFilter = Depends(audit_filter), _: Caller = Depends(services.admin_caller)
    ) -> StreamingResponse:
        rows = services.audit_store.rows(found)
        lines = (audit_row(row).model_dump_json() + "\n" for row in rows)
        return StreamingResponse(
            lines,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="audit.jsonl"'},
        )

    def refuse_unusable(entry: ServerEntry) -> None:
        try:
            catalog.check(entry)
        except ServersError as error:
            raise HTTPException(422, str(error)) from None

    async def change_servers(
        who: Caller, action: str, target: str, change: Callable[[], None]
    ) -> Response:
        try:
            await run_in_threadpool(change)
        except ServerExists as error:
            raise HTTPException(409, str(error)) from None
        except NoSuchServer as error:
            raise HTTPException(404, str(error)) from None
        await catalog.refresh()
        await run_in_threadpool(services.audit.admin_action, who.identity, action, target)
        return Response(status_code=204)

    @routes.get("/admin/servers/{name}")
    async def admin_server(name: str, _: Caller = Depends(services.admin_caller)) -> ServerEntry:
        try:
            return await run_in_threadpool(catalog.store.entry, name)
        except NoSuchServer as error:
            raise HTTPException(404, str(error)) from None

    @routes.post("/admin/servers", status_code=204)
    async def admin_add_server(
        entry: ServerEntry, who: Caller = Depends(services.admin_posting_caller)
    ) -> Response:
        refuse_unusable(entry)
        return await change_servers(who, "add_server", entry.name, lambda: catalog.store.add(entry))

    @routes.put("/admin/servers/{name}", status_code=204)
    async def admin_edit_server(
        name: str, entry: ServerEntry, who: Caller = Depends(services.admin_posting_caller)
    ) -> Response:
        if entry.name != name:
            raise HTTPException(422, f"the body names {entry.name}, not {name}")
        refuse_unusable(entry)
        return await change_servers(who, "edit_server", name, lambda: catalog.store.update(entry))

    @routes.post("/admin/servers/{name}/pause", status_code=204)
    async def admin_pause_server(
        name: str, who: Caller = Depends(services.admin_posting_caller)
    ) -> Response:
        return await change_servers(
            who, "pause_server", name, lambda: catalog.store.set_paused(name, True)
        )

    @routes.post("/admin/servers/{name}/resume", status_code=204)
    async def admin_resume_server(
        name: str, who: Caller = Depends(services.admin_posting_caller)
    ) -> Response:
        return await change_servers(
            who, "resume_server", name, lambda: catalog.store.set_paused(name, False)
        )

    @routes.delete("/admin/servers/{name}", status_code=204)
    async def admin_remove_server(
        name: str, who: Caller = Depends(services.admin_posting_caller)
    ) -> Response:
        return await change_servers(who, "remove_server", name, lambda: catalog.store.remove(name))

    @routes.post("/admin/employees/revoke")
    async def admin_revoke_employee(
        body: RevokeBody, who: Caller = Depends(services.admin_posting_caller)
    ) -> RevokeOut:
        async with services.database.user_lock(body.user_id):
            await services.connections.cancel_all_for(body.user_id)
            revoked = await run_in_threadpool(services.tokens.delete_user_tokens, body.user_id)
            ended = await run_in_threadpool(services.sessions.delete_user, body.user_id)
        await run_in_threadpool(
            services.audit.admin_action, who.identity, "revoke_employee", body.user_id
        )
        return RevokeOut(tokens=revoked, sessions=ended)

    return routes
