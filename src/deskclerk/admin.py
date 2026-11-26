# The summary behind /admin: each configured server, paused or not, with how many employees have
# connected it, what each employee spent this month, and the conversations most recently active.
# Recent activity comes from the conversations table, which holds the employee, the title and the
# time of the last message. Each employee row also carries the display name recorded at sign-in,
# or none when the employee has not been seen since names were stored. A title is the start of a
# request, so it is shown only when DESKCLERK_AUDIT_TEXT allows request text in the logs. The audit
# routes read the audit table. A server's tools are listed live, as the admin, to show whether the
# server answers and which of its tools write.

from datetime import datetime

from pydantic import BaseModel

from deskclerk.auditstore import AuditPage, AuditRow
from deskclerk.connections import ToolInfo
from deskclerk.conversations import recent_conversations
from deskclerk.database import Database
from deskclerk.employees import Employee
from deskclerk.servers import McpServer
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore

RECENT_LIMIT = 50


class AdminServer(BaseModel):
    name: str
    url: str
    auth: str
    has_headers: bool
    trusted: bool
    paused: bool
    employees_connected: int | None
    allowlist_size: int | None


class AdminTool(BaseModel):
    name: str
    title: str
    read_only: bool
    allowlisted: bool
    asks_first: bool


class AdminServerTools(BaseModel):
    reachable: bool | None
    error: str | None
    tools: list[AdminTool]


class AdminEmployee(BaseModel):
    user_id: str
    display_name: str


class AdminUsage(BaseModel):
    user_id: str
    name: str | None
    calls: int
    spent_usd: float
    budget_usd: float


class AdminActivity(BaseModel):
    user_id: str
    name: str | None
    title: str | None
    updated_at: datetime


class AdminSummary(BaseModel):
    month: str
    servers: list[AdminServer]
    usage: list[AdminUsage]
    recent: list[AdminActivity]
    employees: list[AdminEmployee]


class AdminAuditRow(BaseModel):
    id: int
    time: datetime
    user: str
    name: str | None
    entry: str
    servers: list[str]
    tools: list[str]
    decisions: list[str]
    flagged: bool
    hash: str
    line: str


class AdminAuditPage(BaseModel):
    rows: list[AdminAuditRow]
    next_cursor: int | None


class AdminChain(BaseModel):
    ok: bool
    rows: int
    first_broken: int | None


def server_rows(servers: list[McpServer], tokens: TokenStore) -> list[AdminServer]:
    counts = tokens.connected_counts()
    return [
        AdminServer(
            name=server.name,
            url=server.url,
            auth=server.auth,
            has_headers=bool(server.headers),
            trusted=server.trusted,
            paused=server.paused,
            employees_connected=None if server.auth == "none" else counts.get(server.name, 0),
            allowlist_size=None if server.tools is None else len(server.tools),
        )
        for server in servers
    ]


def tool_rows(server: McpServer, listed: list[ToolInfo]) -> list[AdminTool]:
    return [
        AdminTool(
            name=tool.name,
            title=tool.title,
            read_only=tool.read_only,
            allowlisted=server.tools is None or tool.name in server.tools,
            asks_first=not tool.read_only
            or not server.trusted
            or tool.name in server.confirm_tools,
        )
        for tool in listed
    ]


def employee_rows(employees: list[Employee]) -> list[AdminEmployee]:
    return [AdminEmployee(user_id=e.user_id, display_name=e.display_name) for e in employees]


def usage_rows(
    spending: SpendingStore, month: str, budget_usd: float, names: dict[str, str]
) -> list[AdminUsage]:
    return [
        AdminUsage(
            user_id=row.user_id,
            name=names.get(row.user_id),
            calls=row.calls,
            spent_usd=row.total_usd,
            budget_usd=budget_usd,
        )
        for row in spending.report(month)
    ]


async def recent_rows(
    database: Database, show_titles: bool, names: dict[str, str]
) -> list[AdminActivity]:
    found = await recent_conversations(database, RECENT_LIMIT)
    return [
        AdminActivity(
            user_id=c.user_id,
            name=names.get(c.user_id),
            title=c.title if show_titles else None,
            updated_at=c.updated_at,
        )
        for c in found
    ]


def audit_row(row: AuditRow, names: dict[str, str]) -> AdminAuditRow:
    return AdminAuditRow(**vars(row), name=names.get(row.user))


def audit_page(page: AuditPage, names: dict[str, str]) -> AdminAuditPage:
    return AdminAuditPage(
        rows=[audit_row(row, names) for row in page.rows], next_cursor=page.next_cursor
    )
